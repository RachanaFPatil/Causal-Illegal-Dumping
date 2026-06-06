"""
FaceID/database.py
Manages the SQLite offender database.
Stores name, email, phone, UID reference, and face embedding (numpy vector).
"""

import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path("faceid.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS offenders (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT NOT NULL,
                email            TEXT,
                phone            TEXT,
                uid_ref          TEXT,
                embedding        BLOB NOT NULL,
                violation_count  INTEGER DEFAULT 0,
                registered_on    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS unknown_violations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT NOT NULL,
                evidence_frame_path TEXT,
                face_crop_path      TEXT,
                zone                TEXT DEFAULT '',
                status              TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
    print("[FaceID] Database ready at faceid.db")


def save_offender(name, email, phone, uid_ref, embedding: np.ndarray) -> int:
    blob = embedding.tobytes()
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO offenders (name, email, phone, uid_ref, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (name, email, phone, uid_ref, blob))
        conn.commit()
        return cur.lastrowid


def load_all_offenders() -> list:
    """Returns list of dicts with embedding as np.ndarray."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM offenders").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        result.append(d)
    return result


def increment_violation_count(offender_id: int) -> None:
    with get_connection() as conn:
        conn.execute("""
            UPDATE offenders SET violation_count = violation_count + 1
            WHERE id = ?
        """, (offender_id,))
        conn.commit()


def log_unknown_violation(timestamp, evidence_frame_path, face_crop_path, zone="") -> int:
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO unknown_violations
                (timestamp, evidence_frame_path, face_crop_path, zone)
            VALUES (?, ?, ?, ?)
        """, (timestamp, evidence_frame_path, face_crop_path, zone))
        conn.commit()
        return cur.lastrowid