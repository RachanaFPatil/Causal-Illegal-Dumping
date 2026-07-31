"""
FaceID/face_id_module.py
========================
Called by run_pipeline.py when Layer 5 fires a violation AND no plate is detected.

Decision logic:
  plate found  → existing vehicle workflow (UNCHANGED)
  no plate     → FaceIDModule.process(dumping_frame, best_frame)
                 → face detected + matched → pedestrian challan Rs.300
                 → face detected + unmatched → log unknown_violations
                 → no face detected → log unknown_violations

IMPORTANT: challan evidence uses the dumping_frame (the action frame),
NOT the face crop. Face is used only for identity lookup.
"""

import cv2
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EVIDENCE_DIR = Path("evidence/faces")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

PEDESTRIAN_PENALTY = 300.0   # Rs.


class FaceIDModule:
    """Instantiate once at pipeline startup. Call process() per violation."""

    def __init__(self):
        from FaceID.database import init_db
        init_db()
        # Lazy-load insightface — only if actually needed
        self._app = None
        # Penalty manager reused from main pipeline (passed in or created here)
        from penalty_manager import PenaltyManager
        self._pm = PenaltyManager()
        print("[FaceID] Module ready.")

    def _ensure_model(self):
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
                self._app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                self._app.prepare(ctx_id=0, det_size=(640, 640))
                print("[FaceID] InsightFace buffalo_l loaded.")
            except ImportError:
                print("[FaceID] insightface not installed — face ID disabled.")
                print("         pip install insightface onnxruntime")
                raise

    def process(
        self,
        dumping_frame:  np.ndarray,   # the frame showing the actual dumping act
        best_frame:     np.ndarray,   # sharpest frame for face detection
        location:       str  = "Unknown Location",
        confidence:     float = 0.0,
        pair_id:        str  = "",
    ) -> Optional[dict]:
        """
        Run FaceID on a violation.

        Parameters
        ----------
        dumping_frame : BGR frame showing the dumping action — used as challan evidence
        best_frame    : sharpest available frame — used for face detection only
        location      : location string for challan
        confidence    : Layer 5 confidence
        pair_id       : event pair_id for deduplication

        Returns dict with status + challan info, or None.
        """
        ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save dumping_frame as the challan evidence image
        evidence_path = str(EVIDENCE_DIR / f"dumping_evidence_{ts_tag}.jpg")
        cv2.imwrite(evidence_path, dumping_frame)

        # Try face detection on best_frame
        try:
            self._ensure_model()
        except ImportError:
            return self._log_unknown(ts, evidence_path, location, "insightface_unavailable")

        from FaceID.matcher import get_embedding, match_face
        result = get_embedding(self._app, best_frame)

        if result is None:
            print("[FaceID] No face detected in violation frame.")
            return self._log_unknown(ts, evidence_path, location, "no_face_detected")

        embedding, bbox, face_crop = result

        # Save face crop separately (identification only — NOT the challan evidence)
        face_path = str(EVIDENCE_DIR / f"face_crop_{ts_tag}.jpg")
        cv2.imwrite(face_path, face_crop)

        # Match against DB
        match = match_face(embedding)

        if match:
            print(f"[FaceID] ✅ Match: {match['name']} (sim={match['similarity']:.3f})")
            return self._issue_challan(match, evidence_path, location, confidence, ts_tag)
        else:
            print("[FaceID] ❓ No match — logging unknown violation.")
            return self._log_unknown(ts, evidence_path, location, "no_db_match",
                                     face_path=face_path)

    def _issue_challan(self, match: dict, evidence_path: str,
                       location: str, confidence: float,
                       ts_tag: str) -> dict:
        """Create a pedestrian challan with FaceID-matched owner and Rs.300 penalty."""
        try:
            # Create violation with plate_number=None → pedestrian type
            # Then override owner details with face match
            challan_id = self._pm.create_violation(
                plate_number              = None,
                evidence_video_path       = None,
                evidence_plate_image_path = evidence_path,
                location                  = location,
                confidence                = confidence,
                override_penalty          = PEDESTRIAN_PENALTY,
                override_owner            = {
                    "name":  match["name"],
                    "email": match["email"],
                    "phone": match["phone"],
                },
            )
            pdf_path = self._pm.generate_challan(challan_id)
            print(f"[FaceID] 📄 Challan {challan_id} → {pdf_path}")

            # Send email via delivery agent
            if match.get("email"):
                try:
                    from delivery_agent import DeliveryAgent
                    da = DeliveryAgent()
                    da.send_violation_notification(challan_id)
                except Exception as e:
                    print(f"[FaceID] Email send failed: {e}")

            return {
                "status":     "matched",
                "name":       match["name"],
                "email":      match["email"],
                "challan_id": challan_id,
                "pdf_path":   pdf_path,
                "similarity": match["similarity"],
                "evidence":   evidence_path,
            }
        except Exception as e:
            logger.error("[FaceID] Challan creation failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _log_unknown(self, ts: str, evidence_path: str,
                     location: str, reason: str,
                     face_path: str = "") -> dict:
        from FaceID.database import log_unknown_violation
        uid = log_unknown_violation(
            timestamp           = ts,
            evidence_frame_path = evidence_path,
            face_crop_path      = face_path,
            zone                = location,
        )
        return {"status": "unknown", "reason": reason,
                "unknown_id": uid, "evidence": evidence_path}