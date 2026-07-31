"""
penalty_manager_patch.py
========================
Monkey-patches PenaltyManager.create_violation() to accept two new
optional parameters WITHOUT modifying the original file:

    override_penalty : float  — replaces base penalty (used for face-ID Rs.300)
    override_owner   : dict   — {"name", "email", "phone"} skips DB owner lookup

Drop this file next to penalty_manager.py and import it ONCE at startup.
run_pipeline.py already does: from penalty_manager_patch import apply_patch; apply_patch()
"""

from __future__ import annotations
from typing import Optional
import sqlite3
from datetime import datetime, timedelta
import uuid
import logging

logger = logging.getLogger(__name__)


def apply_patch():
    """Patch PenaltyManager.create_violation in-place."""
    import penalty_manager as pm_mod
    PenaltyManager = pm_mod.PenaltyManager

    _original_create = PenaltyManager.create_violation

    def create_violation_patched(
        self,
        plate_number:              Optional[str] = None,
        evidence_video_path:       Optional[str] = None,
        evidence_plate_image_path: Optional[str] = None,
        location:                  str            = "",
        confidence:                float          = 0.0,
        override_penalty:          Optional[float] = None,
        override_owner:            Optional[dict]  = None,
    ) -> str:
        # If no overrides, delegate to original
        if override_penalty is None and override_owner is None:
            return _original_create(
                self,
                plate_number              = plate_number,
                evidence_video_path       = evidence_video_path,
                evidence_plate_image_path = evidence_plate_image_path,
                location                  = location,
                confidence                = confidence,
            )

        # Patched path: used by FaceID pedestrian challans
        now        = datetime.now()
        challan_id = self._make_challan_id(plate_number)
        is_vehicle = plate_number is not None and plate_number.strip() != ""
        ctype      = "vehicle" if is_vehicle else "pedestrian"
        base       = (pm_mod.VEHICLE_PENALTY if is_vehicle
                      else pm_mod.PEDESTRIAN_PENALTY)
        penalty    = override_penalty if override_penalty is not None else base
        due_date   = (now + timedelta(days=pm_mod.DUE_DAYS)).strftime("%Y-%m-%d")
        notes      = f"conf={confidence:.2f}"

        if override_owner:
            owner_name   = override_owner.get("name",  "Unknown Person")
            phone_number = override_owner.get("phone", None)
            email        = override_owner.get("email", None)
        elif is_vehicle:
            row = self._lookup_owner(plate_number.strip().upper())
            if row:
                owner_name, phone_number, email = (
                    row["owner_name"], row["phone_number"], row["email"])
            else:
                owner_name, phone_number, email = "Owner Not Found", None, None
        else:
            owner_name, phone_number, email = "Unknown Person", None, None

        from penalty_manager import _get_connection
        with _get_connection() as conn:
            conn.execute("""
                INSERT INTO violations (
                    challan_id, plate_number, owner_name, phone_number,
                    email, challan_type, violation_timestamp, location,
                    penalty_amount, base_penalty_amount, due_date, status,
                    evidence_plate_image_path, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                challan_id, plate_number, owner_name, phone_number,
                email, ctype, now.strftime("%Y-%m-%d %H:%M:%S"),
                location, penalty, penalty, due_date, "pending",
                evidence_plate_image_path, notes,
            ))
            conn.commit()
        logger.info("Patched create_violation | %s | penalty=Rs.%.0f", challan_id, penalty)
        return challan_id

    PenaltyManager.create_violation = create_violation_patched
    logger.info("PenaltyManager.create_violation patched with override support.")