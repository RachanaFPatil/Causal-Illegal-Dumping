"""
dashboard/app.py  — FIXED VERSION
===================================
Fixes applied:
  1. PDF download: resolve pdf_challan_path to absolute path before send_file
     (was failing because Flask resolved it relative to dashboard/ not project root)
  2. Evidence images: normalise ALL path formats — Windows backslash, absolute,
     relative — and search evidence/faces + evidence root + project-root-relative
  3. Removed broken /api/violations/<id>/notes route (was causing 500 on 📋 click)
  4. Added /api/db/clear endpoint for resetting all data between demo runs
  5. Fix: fid_conn().commit() → with fid_conn() as c: c.execute().commit()
     (SQLite context manager doesn't auto-commit on __exit__ for WAL connections)

Run:
    cd <project_root>
    python dashboard/app.py
"""

from __future__ import annotations
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file, abort

# ── Path setup: project root is parent of dashboard/ ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from penalty_manager import PenaltyManager, _get_connection, DB_PATH
from hotspot.hotspot_manager import HotspotManager, _get_conn as _hs_conn

app = Flask(__name__, static_folder="static", template_folder="templates")
pm = PenaltyManager()
hm = HotspotManager()

# Resolve dirs relative to PROJECT_ROOT, not the cwd of whoever starts Flask
CHALLAN_DIR  = PROJECT_ROOT / "challans"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"


# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def _cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r


# ── Static / root ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(
        str(Path(__file__).parent / "templates"), "index.html"
    )

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(str(Path(__file__).parent / "static"), path)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/summary")
def api_summary():
    s  = pm.summary()
    hs = hm.summary()
    try:
        from FaceID.database import get_connection as fid_conn
        with fid_conn() as c:
            unknown = c.execute(
                "SELECT COUNT(*) FROM unknown_violations WHERE status='pending'"
            ).fetchone()[0]
    except Exception:
        unknown = 0
    return jsonify({**s, **hs, "unknown_faces_pending": unknown})


# ══════════════════════════════════════════════════════════════════════════════
#  VIOLATIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/violations")
def api_violations():
    status = request.args.get("status")
    ctype  = request.args.get("type")
    search = request.args.get("search", "")
    limit  = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    q = "SELECT * FROM violations WHERE 1=1"
    params = []
    if status:
        q += " AND status=?"; params.append(status)
    if ctype:
        q += " AND challan_type=?"; params.append(ctype)
    if search:
        q += " AND (challan_id LIKE ? OR plate_number LIKE ? OR owner_name LIKE ?)"
        params += [f"%{search}%", f"%{search}%", f"%{search}%"]
    q += " ORDER BY violation_timestamp DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    with _get_connection() as conn:
        rows = conn.execute(q, params).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/violations/<challan_id>")
def api_violation_detail(challan_id):
    row = pm.get_violation_by_challan(challan_id)
    if row is None:
        abort(404)
    return jsonify(dict(row))


@app.route("/api/violations/<challan_id>/mark-paid", methods=["POST"])
def api_mark_paid(challan_id):
    txn = (request.json or {}).get("transaction_id", "")
    ok  = pm.mark_as_paid(challan_id, txn or None)
    return jsonify({"success": ok})


@app.route("/api/violations/<challan_id>/escalate", methods=["POST"])
def api_force_escalate(challan_id):
    try:
        days = int((request.json or {}).get("days", 2))
        if hasattr(pm, "simulate_days_passed"):
            pm.simulate_days_passed(challan_id, days)
        pm.check_and_escalate()
        row = pm.get_violation_by_challan(challan_id)
        return jsonify({
            "success":    True,
            "new_amount": dict(row).get("penalty_amount") if row else None,
            "status":     dict(row).get("status") if row else None,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/violations/simulate-escalation", methods=["POST"])
def api_simulate_escalation():
    try:
        days = int((request.json or {}).get("days", 0))
        if days > 0 and hasattr(pm, "simulate_days_passed"):
            with _get_connection() as conn:
                rows = conn.execute(
                    "SELECT challan_id FROM violations "
                    "WHERE status IN ('pending','notified','escalated')"
                ).fetchall()
            for row in rows:
                pm.simulate_days_passed(row["challan_id"], days)
        n = pm.check_and_escalate()
        return jsonify({"success": True, "escalated": n, "days_simulated": days})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/violations/escalate-check", methods=["POST"])
def api_escalate_check():
    n = pm.check_and_escalate()
    return jsonify({"escalated": n})


# ══════════════════════════════════════════════════════════════════════════════
#  CHALLANS — PDF download  (FIX 1: absolute path resolution)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_pdf_path(pdf_path_str: str) -> Path | None:
    """
    Resolve pdf_challan_path to an absolute Path.
    The DB stores it as a relative path like 'challans/BBMP-VH-xxx.pdf'.
    Flask's cwd may differ from PROJECT_ROOT, so we must absolutise.
    """
    if not pdf_path_str:
        return None
    p = Path(pdf_path_str)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to project root
    p2 = PROJECT_ROOT / p
    if p2.exists():
        return p2
    # Try relative to current working directory
    p3 = Path.cwd() / p
    if p3.exists():
        return p3
    return None


@app.route("/api/challan/<challan_id>/pdf")
def api_challan_pdf(challan_id):
    row = pm.get_violation_by_challan(challan_id)
    if row is None:
        return jsonify({"error": f"Challan {challan_id} not found in database"}), 404

    row_dict  = dict(row)
    pdf_path  = _resolve_pdf_path(row_dict.get("pdf_challan_path") or "")

    # If the stored path doesn't exist, regenerate
    if pdf_path is None:
        regen = pm.generate_challan(challan_id)
        if regen:
            pdf_path = _resolve_pdf_path(regen)

    if pdf_path is None or not pdf_path.exists():
        return jsonify({
            "error": (
                f"PDF for {challan_id} not found. "
                "Click '↻ Regen' in the Challans tab to regenerate it."
            )
        }), 404

    return send_file(
        str(pdf_path),
        as_attachment=True,
        download_name=f"{challan_id}.pdf",
        mimetype="application/pdf",
    )


@app.route("/api/challan/<challan_id>/regenerate", methods=["POST"])
def api_regen_pdf(challan_id):
    path = pm.generate_challan(challan_id)
    return jsonify({"pdf_path": path, "success": path is not None})


# ══════════════════════════════════════════════════════════════════════════════
#  EVIDENCE images  (FIX 2: robust path normalisation)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_evidence_path(raw: str) -> Path | None:
    """
    Accept any of:
      - evidence/faces/dumping_evidence_xxx.jpg   (Linux relative)
      - evidence\\faces\\dumping_evidence_xxx.jpg  (Windows relative)
      - C:\\Users\\...\\evidence\\faces\\file.jpg  (Windows absolute)
      - /home/.../evidence/faces/file.jpg          (Linux absolute)
      - just a filename like dumping_evidence_xxx.jpg

    Returns the first existing Path, or None.
    """
    if not raw:
        return None

    # Normalise slashes
    norm = raw.replace("\\", "/")

    # Try 1: as-is (already absolute and exists)
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    # Try 2: relative to project root
    p2 = PROJECT_ROOT / norm
    if p2.exists():
        return p2

    # Try 3: strip any leading path components until 'evidence'
    parts = norm.replace("\\", "/").split("/")
    try:
        ev_idx = next(i for i, x in enumerate(parts) if x == "evidence")
        rel = "/".join(parts[ev_idx:])
        p3 = PROJECT_ROOT / rel
        if p3.exists():
            return p3
    except StopIteration:
        pass

    # Try 4: just the filename in evidence/faces or evidence/
    fname = Path(norm).name
    for d in [EVIDENCE_DIR / "faces", EVIDENCE_DIR]:
        p4 = d / fname
        if p4.exists():
            return p4

    # Try 5: relative to cwd
    p5 = Path.cwd() / norm
    if p5.exists():
        return p5

    return None


@app.route("/api/evidence/<path:filename>")
def api_evidence(filename):
    p = _resolve_evidence_path(filename)
    if p and p.exists():
        return send_file(str(p))
    # Last resort: the raw filename as given by the DB (may be an absolute path)
    p2 = _resolve_evidence_path(filename.replace("/", os.sep))
    if p2 and p2.exists():
        return send_file(str(p2))
    return jsonify({"error": f"Evidence file not found: {filename}"}), 404


# New: serve evidence by its full stored path (used by Face ID view buttons)
@app.route("/api/evidence-by-path")
def api_evidence_by_path():
    raw = request.args.get("path", "")
    if not raw:
        return jsonify({"error": "No path provided"}), 400
    p = _resolve_evidence_path(raw)
    if p and p.exists():
        return send_file(str(p))
    return jsonify({"error": f"Not found: {raw}"}), 404


# ══════════════════════════════════════════════════════════════════════════════
#  HOTSPOTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/hotspots")
def api_hotspots():
    return jsonify(hm.get_all_hotspots())

@app.route("/api/hotspots/events")
def api_hotspot_events():
    return jsonify(hm.get_all_events())

@app.route("/api/hotspots/run-check", methods=["POST"])
def api_hotspot_run():
    escalated = hm.run_hotspot_check()
    return jsonify({"escalated_count": len(escalated), "escalated": escalated})

@app.route("/api/hotspots/<hotspot_id>/report")
def api_hotspot_report(hotspot_id):
    path = PROJECT_ROOT / "hotspot_reports" / f"{hotspot_id}_report.json"
    if path.exists():
        return send_file(str(path), mimetype="application/json")
    # Generate live from DB
    try:
        from hotspot.hotspot_manager import _get_conn
        with _get_conn() as conn:
            hs  = conn.execute("SELECT * FROM hotspots WHERE hotspot_id=?",
                               (hotspot_id,)).fetchone()
            evs = conn.execute(
                "SELECT ve.* FROM violation_events ve "
                "JOIN hotspot_events he ON ve.id=he.event_id "
                "WHERE he.hotspot_id=?", (hotspot_id,)
            ).fetchall()
        if hs is None:
            return jsonify({"error": "Hotspot not found"}), 404
        report = {
            "hotspot_id":      hs["hotspot_id"],
            "location_name":   hs["location_name"],
            "violation_count": hs["violation_count"],
            "status":          hs["status"],
            "first_seen":      hs["first_seen"],
            "last_seen":       hs["last_seen"],
            "latitude":        hs["latitude"],
            "longitude":       hs["longitude"],
            "events":          [dict(e) for e in evs],
        }
        return app.response_class(
            response=json.dumps(report, default=str),
            mimetype="application/json",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/hotspots/<hotspot_id>/resolve", methods=["POST"])
def api_hotspot_resolve(hotspot_id):
    with _hs_conn() as conn:
        conn.execute("UPDATE hotspots SET status='resolved' WHERE hotspot_id=?",
                     (hotspot_id,))
        conn.commit()
    return jsonify({"success": True})


# ══════════════════════════════════════════════════════════════════════════════
#  FACE ID
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/faceid/unknown")
def api_faceid_unknown():
    try:
        from FaceID.database import get_connection as fid_conn
        status = request.args.get("status", "pending")
        with fid_conn() as c:
            rows = c.execute(
                "SELECT * FROM unknown_violations WHERE status=? "
                "ORDER BY timestamp DESC", (status,)
            ).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/faceid/unknown/<int:uid>/status", methods=["POST"])
def api_faceid_update_status(uid):
    try:
        from FaceID.database import get_connection as fid_conn
        new_status = (request.json or {}).get("status", "reviewed")
        with fid_conn() as c:
            c.execute("UPDATE unknown_violations SET status=? WHERE id=?",
                      (new_status, uid))
            # Explicit commit — WAL mode requires it
            c.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/faceid/offenders")
def api_faceid_offenders():
    try:
        from FaceID.database import get_connection as fid_conn
        with fid_conn() as c:
            rows = c.execute(
                "SELECT id, name, email, phone, address, uid_ref, "
                "violation_count, registered_on FROM offenders"
            ).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/notifications")
def api_notifications():
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT challan_id, owner_name, email, notification_status,
                   status, violation_timestamp, created_at, penalty_amount
            FROM violations
            WHERE email IS NOT NULL AND email != ''
            ORDER BY violation_timestamp DESC LIMIT 200
        """).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/notifications/<challan_id>/resend", methods=["POST"])
def api_resend(challan_id):
    try:
        from delivery_agent import DeliveryAgent
        da = DeliveryAgent()
        ok = da.send_violation_notification(challan_id)
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
#  DB RESET  (FIX 4: clear all data for a fresh demo run)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/db/clear", methods=["POST"])
def api_db_clear():
    """
    Clears all violations, hotspot data, and unknown face violations.
    Keeps the owners table and offenders table (registered people) intact.
    Use before a fresh demo run.
    """
    try:
        with _get_connection() as conn:
            conn.execute("DELETE FROM violations")
            conn.commit()
        with _hs_conn() as conn:
            conn.execute("DELETE FROM violation_events")
            conn.execute("DELETE FROM hotspot_events")
            conn.execute("DELETE FROM hotspots")
            conn.commit()
        try:
            from FaceID.database import get_connection as fid_conn
            with fid_conn() as c:
                c.execute("DELETE FROM unknown_violations")
                c.commit()
        except Exception:
            pass
        # Delete generated PDFs
        pdf_count = 0
        for pdf in CHALLAN_DIR.glob("*.pdf"):
            try:
                pdf.unlink()
                pdf_count += 1
            except Exception:
                pass
        return jsonify({
            "success": True,
            "message": f"All violations, hotspots, and unknown faces cleared. "
                       f"{pdf_count} PDFs deleted.",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS / ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/charts/violations-by-day")
def api_chart_by_day():
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT DATE(violation_timestamp) as day, COUNT(*) as count
            FROM violations GROUP BY day ORDER BY day DESC LIMIT 30
        """).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/charts/violations-by-type")
def api_chart_by_type():
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT challan_type, COUNT(*) as count FROM violations
            GROUP BY challan_type
        """).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/charts/status-distribution")
def api_chart_status():
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT status, COUNT(*) as count FROM violations GROUP BY status
        """).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/charts/revenue")
def api_chart_revenue():
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT DATE(violation_timestamp) as day,
                   SUM(penalty_amount) as total,
                   SUM(CASE WHEN status='paid' THEN penalty_amount ELSE 0 END) as collected
            FROM violations GROUP BY day ORDER BY day DESC LIMIT 30
        """).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/charts/hotspot-map")
def api_hotspot_map():
    hotspots = hm.get_all_hotspots()
    features = []
    for hs in hotspots:
        if hs["latitude"] or hs["longitude"]:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [hs["longitude"], hs["latitude"]]},
                "properties": {
                    "hotspot_id": hs["hotspot_id"],
                    "location":   hs["location_name"],
                    "count":      hs["violation_count"],
                    "status":     hs["status"],
                    "first_seen": hs["first_seen"],
                    "last_seen":  hs["last_seen"],
                },
            })
    return jsonify({"type": "FeatureCollection", "features": features})


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  VidTrace Admin Dashboard")
    print(f"  Project root: {PROJECT_ROOT}")
    print("  http://localhost:5050")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5050, debug=False)