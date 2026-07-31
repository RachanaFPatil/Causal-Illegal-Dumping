"""
hotspot/hotspot_manager.py
==========================
Hotspot monitoring subsystem for VidTrace.

Every confirmed violation (vehicle or face-based) is logged here with
coordinates. When a location accumulates more than HOTSPOT_THRESHOLD
violations, a hotspot case is created and escalated to municipal
authorities by email.

Since VidTrace is post-event (not live), hotspot detection runs at the
end of each pipeline run. Multiple runs of different videos accumulate
in the same DB — if two videos show dumping at the same location, the
hotspot fires.

Spatial clustering: Two violation events are considered the "same location"
if they are within CLUSTER_RADIUS_METERS of each other (Haversine distance).
For CCTV-based detection where lat/lon comes from the camera location tag,
events from the same camera are always co-located — cluster by camera_id
when lat/lon is identical.
"""

from __future__ import annotations

import json
import logging
import math
import smtplib
import sqlite3
import uuid
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HOTSPOT_DB_PATH      = Path("penalties.db")   # shared with PenaltyManager
CLUSTER_RADIUS_M     = 100.0                  # metres — events within this = same hotspot
HOTSPOT_THRESHOLD    = 2                      # distinct videos at same GPS location before escalation
HOTSPOT_DIR          = Path("hotspot_reports")
HOTSPOT_DIR.mkdir(parents=True, exist_ok=True)

# BBMP escalation contacts — replace with real ward email in production
BBMP_WARD_EMAIL      = "rachfpatil@gmail.com"  # prototype - use real ward email in production
BBMP_HELPLINE        = "1533"               # BBMP solid waste helpline
BBMP_SWACHHA_PORTAL  = "https://bbmp.gov.in/swachha-bengaluru"

# SMTP — reused from delivery_agent config (import it)
try:
    from delivery_agent import SMTP_USER, SMTP_PASSWORD, SMTP_HOST, SMTP_PORT, SENDER_NAME
except ImportError:
    SMTP_USER = SMTP_PASSWORD = ""
    SMTP_HOST = "smtp.gmail.com"; SMTP_PORT = 587
    SENDER_NAME = "VidTrace BBMP Enforcement"


# ══════════════════════════════════════════════════════════════════════════════
#  DB schema helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(HOTSPOT_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_hotspot_tables() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violation_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                challan_id      TEXT,
                source_video    TEXT,
                location_name   TEXT DEFAULT '',
                latitude        REAL DEFAULT 0.0,
                longitude       REAL DEFAULT 0.0,
                camera_id       TEXT DEFAULT '',
                timestamp       TEXT NOT NULL,
                evidence_path   TEXT DEFAULT '',
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hotspots (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                hotspot_id          TEXT UNIQUE NOT NULL,
                location_name       TEXT DEFAULT '',
                latitude            REAL DEFAULT 0.0,
                longitude           REAL DEFAULT 0.0,
                camera_id           TEXT DEFAULT '',
                violation_count     INTEGER DEFAULT 0,
                first_seen          TEXT,
                last_seen           TEXT,
                status              TEXT DEFAULT 'active',
                report_path         TEXT DEFAULT '',
                escalated_at        TEXT,
                created_at          TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hotspot_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                hotspot_id      TEXT NOT NULL,
                event_id        INTEGER NOT NULL,
                FOREIGN KEY (hotspot_id) REFERENCES hotspots(hotspot_id)
            )
        """)
        conn.commit()
    logger.info("Hotspot tables ready.")


# ══════════════════════════════════════════════════════════════════════════════
#  Haversine distance
# ══════════════════════════════════════════════════════════════════════════════

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ══════════════════════════════════════════════════════════════════════════════
#  HotspotManager
# ══════════════════════════════════════════════════════════════════════════════

class HotspotManager:

    def __init__(self):
        init_hotspot_tables()

    # ── Public: log one violation event ──────────────────────────────────

    def log_event(
        self,
        challan_id:    Optional[str],
        source_video:  str   = "",
        location_name: str   = "",
        latitude:      float = 0.0,
        longitude:     float = 0.0,
        camera_id:     str   = "",
        timestamp:     Optional[str] = None,
        evidence_path: str   = "",
    ) -> int:
        """Log one confirmed violation event. Returns the event row id."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _get_conn() as conn:
            cur = conn.execute("""
                INSERT INTO violation_events
                    (challan_id, source_video, location_name, latitude,
                     longitude, camera_id, timestamp, evidence_path)
                VALUES (?,?,?,?,?,?,?,?)
            """, (challan_id, source_video, location_name,
                  latitude, longitude, camera_id, ts, evidence_path))
            conn.commit()
            event_id = cur.lastrowid
        logger.info("Event logged id=%d loc=%s", event_id, location_name)
        return event_id

    # ── Public: run clustering + escalation ──────────────────────────────

    def run_hotspot_check(self) -> list[dict]:
        """
        Cluster all events, update hotspot counts, escalate any that cross
        the threshold. Returns list of newly escalated hotspot dicts.
        """
        events = self._load_all_events()
        escalated = []

        for event in events:
            hs = self._find_nearby_hotspot(event)
            if hs:
                self._add_event_to_hotspot(hs["hotspot_id"], event["id"],
                                           event["timestamp"])
                hs = self._get_hotspot(hs["hotspot_id"])
            else:
                hs = self._create_hotspot(event)

            # Check threshold
            if (hs["violation_count"] > HOTSPOT_THRESHOLD
                    and hs["status"] == "active"):
                result = self._escalate(hs)
                if result:
                    escalated.append(result)

        return escalated

    # ── Public: get all hotspots for dashboard ────────────────────────────

    def get_all_hotspots(self) -> list[dict]:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM hotspots ORDER BY violation_count DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_events(self) -> list[dict]:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM violation_events ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        with _get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as total_events,
                       (SELECT COUNT(*) FROM hotspots) as total_hotspots,
                       (SELECT COUNT(*) FROM hotspots WHERE status='escalated') as escalated
                FROM violation_events
            """).fetchone()
        return dict(row)

    # ── Internal clustering ───────────────────────────────────────────────

    def _load_all_events(self) -> list[dict]:
        """Load events not yet assigned to any hotspot."""
        with _get_conn() as conn:
            assigned_ids = {r[0] for r in conn.execute(
                "SELECT event_id FROM hotspot_events"
            ).fetchall()}
            rows = conn.execute(
                "SELECT * FROM violation_events ORDER BY timestamp ASC"
            ).fetchall()
        return [dict(r) for r in rows if r["id"] not in assigned_ids]

    def _find_nearby_hotspot(self, event: dict) -> Optional[dict]:
        lat, lon, cam = event["latitude"], event["longitude"], event["camera_id"]
        with _get_conn() as conn:
            hotspots = conn.execute(
                "SELECT * FROM hotspots WHERE status != 'resolved'"
            ).fetchall()
        for hs in hotspots:
            hs = dict(hs)
            # Same camera = same location (CCTV fixed mount)
            if cam and cam == hs["camera_id"]:
                return hs
            # Or within cluster radius
            if lat and lon and hs["latitude"] and hs["longitude"]:
                d = _haversine_m(lat, lon, hs["latitude"], hs["longitude"])
                if d <= CLUSTER_RADIUS_M:
                    return hs
            # Fallback: same location name
            if (event["location_name"] and
                    event["location_name"] == hs["location_name"]):
                return hs
        return None

    def _create_hotspot(self, event: dict) -> dict:
        hsid = f"HS-{uuid.uuid4().hex[:8].upper()}"
        ts   = event["timestamp"]
        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO hotspots
                    (hotspot_id, location_name, latitude, longitude,
                     camera_id, violation_count, first_seen, last_seen)
                VALUES (?,?,?,?,?,1,?,?)
            """, (hsid, event["location_name"], event["latitude"],
                  event["longitude"], event["camera_id"], ts, ts))
            conn.execute(
                "INSERT INTO hotspot_events (hotspot_id, event_id) VALUES (?,?)",
                (hsid, event["id"]),
            )
            conn.commit()
        return self._get_hotspot(hsid)

    def _add_event_to_hotspot(self, hsid: str, event_id: int, ts: str) -> None:
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO hotspot_events (hotspot_id, event_id) VALUES (?,?)",
                (hsid, event_id),
            )
            conn.execute("""
                UPDATE hotspots
                SET violation_count = violation_count + 1, last_seen = ?
                WHERE hotspot_id = ?
            """, (ts, hsid))
            conn.commit()

    def _get_hotspot(self, hsid: str) -> dict:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM hotspots WHERE hotspot_id = ?", (hsid,)
            ).fetchone()
        return dict(row)

    # ── Escalation ────────────────────────────────────────────────────────

    def _escalate(self, hs: dict) -> Optional[dict]:
        """Generate report and email BBMP. Returns hotspot dict."""
        try:
            report_path = self._generate_report(hs)
            self._send_escalation_email(hs, report_path)
            with _get_conn() as conn:
                conn.execute("""
                    UPDATE hotspots SET status='escalated',
                    escalated_at=?, report_path=? WHERE hotspot_id=?
                """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      report_path, hs["hotspot_id"]))
                conn.commit()
            logger.warning("HOTSPOT ESCALATED: %s at %s (%d violations)",
                           hs["hotspot_id"], hs["location_name"],
                           hs["violation_count"])
            return self._get_hotspot(hs["hotspot_id"])
        except Exception as e:
            logger.error("Hotspot escalation failed: %s", e)
            return None

    def _generate_report(self, hs: dict) -> str:
        """Generate a plain-text + JSON hotspot report."""
        with _get_conn() as conn:
            event_ids = [r[0] for r in conn.execute(
                "SELECT event_id FROM hotspot_events WHERE hotspot_id=?",
                (hs["hotspot_id"],)
            ).fetchall()]
            events = []
            for eid in event_ids:
                row = conn.execute(
                    "SELECT * FROM violation_events WHERE id=?", (eid,)
                ).fetchone()
                if row:
                    events.append(dict(row))

        report = {
            "hotspot_id":        hs["hotspot_id"],
            "location":          hs["location_name"],
            "coordinates":       {"lat": hs["latitude"], "lon": hs["longitude"]},
            "camera_id":         hs["camera_id"],
            "violation_count":   hs["violation_count"],
            "first_seen":        hs["first_seen"],
            "last_seen":         hs["last_seen"],
            "generated_at":      datetime.now().isoformat(),
            "events":            events,
            "trend_summary":     (
                f"{hs['violation_count']} illegal dumping events detected at this "
                f"location between {hs['first_seen']} and {hs['last_seen']}. "
                "Immediate clearance and preventive measures requested."
            ),
            "bbmp_action_requested": [
                "Immediate waste clearance",
                "Temporary CCTV signage/warning boards installation",
                "Increase patrol frequency",
                "Consider installing permanent bin infrastructure",
            ],
        }

        path = str(HOTSPOT_DIR / f"{hs['hotspot_id']}_report.json")
        Path(path).write_text(json.dumps(report, indent=2, default=str))
        logger.info("Hotspot report saved: %s", path)
        return path

    def _send_escalation_email(self, hs: dict, report_path: str) -> bool:
        if not SMTP_USER or not SMTP_PASSWORD:
            logger.warning("SMTP not configured — hotspot email skipped.")
            return False

        msg = MIMEMultipart("mixed")
        msg["From"]    = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"]      = BBMP_WARD_EMAIL
        msg["Subject"] = (
            f"[VidTrace ALERT] Illegal Dumping Hotspot Detected — "
            f"{hs['location_name']} ({hs['violation_count']} violations)"
        )

        html = f"""
        <html><body style="font-family:Arial,sans-serif;">
        <div style="background:#CC0000;padding:16px;color:white;">
          <h2>⚠ Illegal Dumping Hotspot Alert</h2>
          <p>VidTrace Municipal Enforcement System — BBMP</p>
        </div>
        <div style="padding:20px;border:1px solid #ddd;">
          <p>This is an automated alert from the VidTrace Illegal Dumping
          Detection System. A dumping hotspot has been identified that
          requires immediate municipal action.</p>

          <table style="border-collapse:collapse;width:100%;margin:16px 0;">
            <tr style="background:#f5f5f5;">
              <td style="padding:8px;font-weight:bold;border:1px solid #ddd;">Hotspot ID</td>
              <td style="padding:8px;border:1px solid #ddd;">{hs['hotspot_id']}</td>
            </tr>
            <tr>
              <td style="padding:8px;font-weight:bold;border:1px solid #ddd;">Location</td>
              <td style="padding:8px;border:1px solid #ddd;color:#CC0000;">
                <b>{hs['location_name'] or 'Not specified'}</b></td>
            </tr>
            <tr style="background:#f5f5f5;">
              <td style="padding:8px;font-weight:bold;border:1px solid #ddd;">Violations</td>
              <td style="padding:8px;border:1px solid #ddd;font-size:18px;color:#CC0000;">
                <b>{hs['violation_count']}</b></td>
            </tr>
            <tr>
              <td style="padding:8px;font-weight:bold;border:1px solid #ddd;">First Seen</td>
              <td style="padding:8px;border:1px solid #ddd;">{hs['first_seen']}</td>
            </tr>
            <tr style="background:#f5f5f5;">
              <td style="padding:8px;font-weight:bold;border:1px solid #ddd;">Last Seen</td>
              <td style="padding:8px;border:1px solid #ddd;">{hs['last_seen']}</td>
            </tr>
          </table>

          <div style="background:#FFF8E1;border-left:4px solid #E8500A;padding:12px;">
            <b>Requested Actions:</b>
            <ul>
              <li>Immediate waste clearance at this location</li>
              <li>Install warning/deterrent signage</li>
              <li>Increase patrol frequency</li>
              <li>Consider permanent bin infrastructure</li>
              <li>Register complaint on Swachha Bengaluru portal</li>
            </ul>
            <p>Portal: <a href="{BBMP_SWACHHA_PORTAL}">{BBMP_SWACHHA_PORTAL}</a><br/>
            Helpline: <b>{BBMP_HELPLINE}</b></p>
          </div>

          <p>Full hotspot report is attached as JSON.</p>
          <p style="color:#888;font-size:11px;">Auto-generated by VidTrace.</p>
        </div>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        # Attach JSON report
        if report_path and Path(report_path).exists():
            with open(report_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            f'attachment; filename="{Path(report_path).name}"')
            msg.attach(part)

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.ehlo(); server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, BBMP_WARD_EMAIL, msg.as_string())
            logger.info("Hotspot escalation email sent to %s", BBMP_WARD_EMAIL)
            return True
        except Exception as e:
            logger.error("Hotspot email failed: %s", e)
            return False