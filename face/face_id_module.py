"""
FaceID/face_id_module.py
========================
Main controller. Called by run_pipeline.py when Layer 5 fires a
pedestrian violation (no plate detected).

Flow:
    frame → detect face → extract embedding → match against DB
    → if matched: create challan + send email
    → if unknown: log to unknown_violations table + save face crop
"""

import cv2
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from insightface.app import FaceAnalysis

from FaceID.database import init_db, log_unknown_violation
from FaceID.matcher  import get_embedding, match_face
from FaceID.notifier import send_violation_email
from penalty_manager import PenaltyManager

logger = logging.getLogger(__name__)

EVIDENCE_DIR = Path("evidence/faces")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


class FaceIDModule:
    """
    Instantiate once at pipeline startup. Call process() per violation frame.
    """

    def __init__(self):
        init_db()
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        self._pm = PenaltyManager()
        print("[FaceID] Module ready.")

    def process(
        self,
        frame:      np.ndarray,
        location:   str  = "Unknown Location",
        confidence: float = 0.0,
    ) -> Optional[dict]:
        """
        Run FaceID on a violation frame.

        Parameters
        ----------
        frame      : BGR numpy array from OpenCV
        location   : string location for challan
        confidence : Layer 5 violation confidence score

        Returns
        -------
        dict with match info + challan_id, or None if no face detected.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_tag    = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Step 1: Detect face + extract embedding ───────────────────────
        result = get_embedding(self._app, frame)
        if result is None:
            print("[FaceID] No face detected in violation frame.")
            return None

        embedding, bbox, face_crop = result

        # Save face crop as evidence
        face_path = str(EVIDENCE_DIR / f"face_{ts_tag}.jpg")
        cv2.imwrite(face_path, face_crop)

        # ── Step 2: Match against DB ──────────────────────────────────────
        match = match_face(embedding)

        if match:
            print(f"[FaceID] ✅ Match: {match['name']} "
                  f"(similarity={match['similarity']:.3f})")

            # ── Step 3a: Known person → issue challan ─────────────────────
            challan_id = self._pm.create_violation(
                plate_number              = None,
                evidence_plate_image_path = face_path,
                location                  = location,
                confidence                = confidence,
            )

            # Override owner details with FaceID match
            self._override_owner(challan_id, match)

            pdf_path = self._pm.generate_challan(challan_id)

            # ── Step 4: Send email ────────────────────────────────────────
            if match["email"]:
                violation = self._pm.get_violation_by_challan(challan_id)
                send_violation_email(
                    recipient_email = match["email"],
                    recipient_name  = match["name"],
                    challan_id      = challan_id,
                    penalty_amount  = float(violation["penalty_amount"]),
                    location        = location,
                    pdf_path        = pdf_path,
                )

            return {
                "status":     "matched",
                "name":       match["name"],
                "email":      match["email"],
                "challan_id": challan_id,
                "pdf_path":   pdf_path,
                "similarity": match["similarity"],
                "face_path":  face_path,
            }

        else:
            # ── Step 3b: Unknown person → log for manual review ───────────
            print("[FaceID] ❓ No match — logging unknown violation.")
            log_unknown_violation(
                timestamp           = timestamp,
                evidence_frame_path = face_path,
                face_crop_path      = face_path,
                zone                = location,
            )
            return {
                "status":    "unknown",
                "face_path": face_path,
            }

    def _override_owner(self, challan_id: str, match: dict) -> None:
        """Update the violation record with FaceID-matched owner details."""
        from FaceID.database import get_connection
        with get_connection() as conn:
            conn.execute("""
                UPDATE violations
                SET owner_name   = ?,
                    email        = ?,
                    phone_number = ?,
                    notes        = notes || ' | identified_by=FaceID'
                WHERE challan_id = ?
            """, (match["name"], match["email"],
                  match["phone"], challan_id))
            conn.commit()
        logger.info("[FaceID] Owner updated in DB for challan %s", challan_id)