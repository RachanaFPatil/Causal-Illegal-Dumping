"""
enroll_navya.py
===============
Run this once on your Windows machine to enroll Navya S into faceid.db.
InsightFace buffalo_l model is already downloaded from your previous run.

Usage:
    python enroll_navya.py

Make sure faceid.db is in the same folder as this script (project root).
"""

import cv2
import numpy as np
import sqlite3
import sys
from pathlib import Path

DB_PATH   = Path("faceid.db")
VIDEO     = "testn.mp4"    # put testn.mp4 in project root

NAME      = "Navya S"
EMAIL     = "navyaspnk26@gmail.com"
PHONE     = "9620946682"
UID       = "NAVYA001"
ADDRESS   = ""

# ── Load InsightFace ──────────────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[Enroll] InsightFace loaded.")
except ImportError:
    print("[Enroll] ERROR: insightface not installed. Run: pip install insightface onnxruntime")
    sys.exit(1)

# ── Extract best face from video ──────────────────────────────────────────────
if not Path(VIDEO).exists():
    print(f"[Enroll] ERROR: {VIDEO} not found. Put the video in the project root.")
    sys.exit(1)

cap   = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[Enroll] Scanning {total} frames in {VIDEO} ...")

best_embedding = None
best_score     = 0.0
best_frame_n   = -1

for i in range(0, total, 2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    faces = app.get(frame)
    if not faces:
        continue
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    area      = (face.bbox[2]-face.bbox[0]) * (face.bbox[3]-face.bbox[1])
    det_score = getattr(face, "det_score", 0.5)
    score     = float(area) * float(det_score)
    if score > best_score:
        best_score     = score
        best_embedding = face.embedding.astype(np.float32)
        best_frame_n   = i
        print(f"  Frame {i}: area={area:.0f} det={det_score:.2f} "
              f"norm={np.linalg.norm(best_embedding):.3f}")

cap.release()

if best_embedding is None:
    print("[Enroll] ERROR: No face found in video.")
    sys.exit(1)

print(f"\n[Enroll] Best face at frame {best_frame_n} "
      f"norm={np.linalg.norm(best_embedding):.3f}")

# ── Connect to faceid.db ──────────────────────────────────────────────────────
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

# Ensure address column exists (added in newer version)
cols = [r[1] for r in conn.execute("PRAGMA table_info(offenders)").fetchall()]
if "address" not in cols:
    conn.execute("ALTER TABLE offenders ADD COLUMN address TEXT DEFAULT ''")
    conn.commit()
    print("[Enroll] Added 'address' column to offenders table.")

# Check if already enrolled
existing = conn.execute(
    "SELECT id FROM offenders WHERE name=? OR uid_ref=?", (NAME, UID)
).fetchone()

if existing:
    conn.execute(
        "UPDATE offenders SET embedding=?, email=?, phone=?, uid_ref=?, address=? "
        "WHERE name=? OR uid_ref=?",
        (best_embedding.tobytes(), EMAIL, PHONE, UID, ADDRESS, NAME, UID)
    )
    conn.commit()
    print(f"[Enroll] Updated existing record for '{NAME}'.")
else:
    conn.execute(
        "INSERT INTO offenders (name, email, phone, uid_ref, embedding, address) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (NAME, EMAIL, PHONE, UID, best_embedding.tobytes(), ADDRESS)
    )
    conn.commit()
    print(f"[Enroll] Enrolled '{NAME}' successfully.")

# ── Verify ────────────────────────────────────────────────────────────────────
rows = conn.execute("SELECT id, name, email, phone FROM offenders").fetchall()
print("\n[Enroll] Current offenders in DB:")
for r in rows:
    print(f"  id={r['id']}  name={r['name']!r:20s}  "
          f"email={r['email']!r}  phone={r['phone']!r}")
conn.close()

print(f"\n[Enroll] Done. '{NAME}' will now be matched in future FaceID runs.")