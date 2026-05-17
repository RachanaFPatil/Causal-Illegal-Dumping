"""
Penalty & Challan Management System
=====================================
Integrates with the VidTrace illegal dumping detection pipeline
(Layer1–Layer5 + enhancer.py) to automatically issue challans when a
violation is confirmed.

Usage (from run_pipeline.py or manually):
    from penalty_manager import PenaltyManager

    pm = PenaltyManager()
    challan_id = pm.create_violation(
        plate_number          = "KA05KK5546",
        evidence_video_path   = "vidtrace_output.mp4",
        evidence_plate_image  = "evidence/pair_id_evidence.jpg",
        location              = "Bangalore, KA",
        confidence            = 0.72,
    )
    pm.generate_challan(challan_id)   # creates PDF in challans/
    pm.check_and_escalate()           # run daily / after every pipeline run

Requirements (install once):
    pip install reportlab pillow

Database: penalties.db   (auto-created on first run)
Challans:  challans/     (auto-created folder)
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import io

# ── ReportLab imports ─────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[PenaltyMgr] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH             = Path("penalties.db")
CHALLAN_DIR         = Path("challans")
VEHICLE_PENALTY     = 500.0   # INR
PEDESTRIAN_PENALTY  = 200.0   # INR
DUE_DAYS            = 7       # days until due date
ESCALATION_DAYS     = 2       # escalate every N days after due date
ESCALATION_RATE     = 0.10    # +10% per escalation cycle
AUTHORITY_NAME      = "Bruhat Bengaluru Mahanagara Palike (BBMP)"
AUTHORITY_DEPT      = "Solid Waste Management & Environment Division"
AUTHORITY_ADDRESS   = "N.R. Square, Hudson Circle, Bengaluru – 560 002"
AUTHORITY_PHONE     = "080-22221188"
AUTHORITY_EMAIL     = "swm@bbmp.gov.in"
UPI_ID              = "bbmp.swm@upi"


# ══════════════════════════════════════════════════════════════════════════════
#  Database helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection with row-factory set."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    """Create tables and seed mock owner data if the DB doesn't exist yet."""
    CHALLAN_DIR.mkdir(parents=True, exist_ok=True)

    with _get_connection() as conn:
        # ── owners table ──────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS owners (
                plate_number TEXT PRIMARY KEY,
                owner_name   TEXT NOT NULL,
                phone_number TEXT,
                email        TEXT,
                address      TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── violations table ──────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                challan_id               TEXT UNIQUE NOT NULL,
                plate_number             TEXT,
                owner_name               TEXT,
                phone_number             TEXT,
                email                    TEXT,
                challan_type             TEXT NOT NULL DEFAULT 'vehicle',
                violation_timestamp      TIMESTAMP NOT NULL,
                location                 TEXT DEFAULT '',
                penalty_amount           REAL NOT NULL,
                due_date                 DATE NOT NULL,
                status                   TEXT NOT NULL DEFAULT 'pending',
                escalation_count         INTEGER NOT NULL DEFAULT 0,
                evidence_video_path      TEXT,
                evidence_plate_image_path TEXT,
                pdf_challan_path         TEXT,
                notes                    TEXT DEFAULT '',
                created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Seed mock owner (only this one record) ────────────────────────
        conn.execute("""
            INSERT OR IGNORE INTO owners
                (plate_number, owner_name, phone_number, email, address)
            VALUES
                ('KA05KK5546', 'Offender', '9901967521',
                 'rachfpatil@gmail.com', 'Bangalore, Karnataka')
        """)
        conn.commit()

    logger.info(f"Database ready at '{DB_PATH}'")


# ══════════════════════════════════════════════════════════════════════════════
#  QR Code helper (pure PIL — no qrcode library required)
# ══════════════════════════════════════════════════════════════════════════════

def _make_upi_qr_bytes(challan_id: str, amount: float) -> Optional[bytes]:
    """
    Generate a minimal UPI deep-link QR code as PNG bytes.
    Falls back gracefully to None if any dependency is missing.

    UPI URL format:
        upi://pay?pa=<vpa>&pn=<name>&am=<amount>&tn=<note>&cu=INR
    """
    upi_url = (
        f"upi://pay"
        f"?pa={UPI_ID}"
        f"&pn=BBMP+SWM"
        f"&am={amount:.2f}"
        f"&tn=Challan+{challan_id}"
        f"&cu=INR"
    )
    try:
        import qrcode                                       # optional
        qr = qrcode.QRCode(box_size=4, border=2)
        qr.add_data(upi_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pass

    # ── Fallback: draw a simple placeholder with PIL ──────────────────────
    try:
        from PIL import Image, ImageDraw, ImageFont
        size = 160
        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, size-1, size-1], outline="black", width=3)
        # Draw a basic QR-like pattern border
        for i in range(0, size, 10):
            shade = 200 if (i // 10) % 2 == 0 else 240
            draw.line([(i, 0), (i, size)], fill=(shade, shade, shade))
        draw.text((10, 65), "SCAN TO PAY", fill="black")
        draw.text((10, 80), f"Rs.{amount:.0f}", fill="black")
        draw.text((10, 95), "UPI", fill="gray")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  PDF Challan Generator
# ══════════════════════════════════════════════════════════════════════════════

class _ChallanPDF:
    """Builds a professional municipal challan PDF using ReportLab Platypus."""

    # ── Colour palette ────────────────────────────────────────────────────
    _NAVY   = colors.HexColor("#1B2A6B")   # header background
    _ORANGE = colors.HexColor("#E8500A")   # accent / violation banner
    _GREEN  = colors.HexColor("#1A7A4A")   # paid / legal
    _LGRAY  = colors.HexColor("#F4F4F4")   # alternating row fill
    _DGRAY  = colors.HexColor("#444444")   # body text
    _RED    = colors.HexColor("#CC0000")   # warning text

    def __init__(self, violation: sqlite3.Row):
        self._v   = dict(violation)
        self._buf = io.BytesIO()

    # ── Public entry point ────────────────────────────────────────────────

    def build(self) -> bytes:
        doc = SimpleDocTemplate(
            self._buf,
            pagesize=A4,
            leftMargin=1.5*cm, rightMargin=1.5*cm,
            topMargin=1.5*cm,  bottomMargin=1.5*cm,
            title=f"Traffic Challan {self._v['challan_id']}",
            author=AUTHORITY_NAME,
        )
        story = []
        styles = getSampleStyleSheet()

        story += self._header(styles)
        story.append(Spacer(1, 6*mm))
        story += self._violation_banner(styles)
        story.append(Spacer(1, 5*mm))
        story += self._challan_details_table(styles)
        story.append(Spacer(1, 5*mm))
        story += self._owner_details_table(styles)
        story.append(Spacer(1, 5*mm))
        story += self._penalty_table(styles)
        story.append(Spacer(1, 5*mm))
        story += self._escalation_note(styles)
        story.append(Spacer(1, 5*mm))
        story += self._evidence_section(styles)
        story.append(Spacer(1, 5*mm))
        story += self._payment_section(styles)
        story.append(Spacer(1, 5*mm))
        story += self._signature_section(styles)
        story.append(Spacer(1, 3*mm))
        story += self._footer_note(styles)

        doc.build(story)
        return self._buf.getvalue()

    # ── Shared style helper ───────────────────────────────────────────────

    def _s(self, name: str, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, **kw)

    # ── Section builders ──────────────────────────────────────────────────

    def _header(self, styles) -> list:
        """Tri-column header: logo placeholder | authority text | challan id."""
        header_style = self._s(
            "Header",
            fontSize=9, textColor=colors.white,
            fontName="Helvetica", leading=13,
        )
        title_style = self._s(
            "HTitle",
            fontSize=14, textColor=colors.white,
            fontName="Helvetica-Bold", leading=18,
        )
        id_style = self._s(
            "HID",
            fontSize=8, textColor=colors.HexColor("#AAAAFF"),
            fontName="Helvetica", leading=12, alignment=TA_RIGHT,
        )
        # Left cell — emblem placeholder
        emblem = Paragraph(
            "<b>🏛</b>",
            self._s("E", fontSize=30, textColor=colors.white,
                    fontName="Helvetica", leading=40),
        )
        # Middle cell — authority info
        centre_cell = [
            Paragraph(AUTHORITY_NAME, title_style),
            Paragraph(AUTHORITY_DEPT, header_style),
            Paragraph(AUTHORITY_ADDRESS, header_style),
            Paragraph(
                f"Ph: {AUTHORITY_PHONE}   Email: {AUTHORITY_EMAIL}",
                header_style,
            ),
        ]
        # Right cell — challan reference
        right_cell = [
            Paragraph("<b>CHALLAN</b>", self._s(
                "CR", fontSize=16, textColor=colors.white,
                fontName="Helvetica-Bold", alignment=TA_RIGHT)),
            Paragraph(f"ID: {self._v['challan_id']}", id_style),
            Paragraph(
                f"Date: {self._v['violation_timestamp'][:10]}", id_style),
        ]

        tbl = Table([[emblem, centre_cell, right_cell]],
                    colWidths=[2*cm, 12*cm, 4.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), self._NAVY),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ]))
        return [tbl]

    def _violation_banner(self, styles) -> list:
        """Orange banner with violation type."""
        label = "VEHICLE VIOLATION CHALLAN" \
            if self._v["challan_type"] == "vehicle" \
            else "PEDESTRIAN VIOLATION CHALLAN"
        banner = Paragraph(
            f"⚠  {label}  ⚠",
            self._s("Banner", fontSize=13, textColor=colors.white,
                    fontName="Helvetica-Bold", alignment=TA_CENTER, leading=18),
        )
        tbl = Table([[banner]], colWidths=[18.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), self._ORANGE),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return [tbl]

    def _section_heading(self, text: str) -> Paragraph:
        return Paragraph(
            text.upper(),
            ParagraphStyle("SH", fontSize=9, fontName="Helvetica-Bold",
                           textColor=self._NAVY, spaceAfter=3),
        )

    def _row_style(self) -> TableStyle:
        return TableStyle([
            ("BACKGROUND",   (0, 0), (0, -1), self._LGRAY),
            ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",     (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",    (0, 0), (-1, -1), self._DGRAY),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ])

    def _challan_details_table(self, styles) -> list:
        v = self._v
        rows = [
            ["Challan ID",          v["challan_id"]],
            ["Violation Date/Time", v["violation_timestamp"]],
            ["Location",            v["location"] or "—"],
            ["Number Plate",        v["plate_number"] or "NOT DETECTED"],
            ["Detection Confidence",
             f"{float(v.get('notes', '0') or 0):.0%}"
             if "conf=" not in str(v.get("notes", ""))
             else str(v.get("notes", ""))],
        ]
        tbl = Table(rows, colWidths=[5*cm, 13.5*cm])
        tbl.setStyle(self._row_style())
        return [self._section_heading("Violation Details"), tbl]

    def _owner_details_table(self, styles) -> list:
        v = self._v
        owner_found = (v["owner_name"] not in
                       ("Owner Not Found", "Unknown Person", None))
        rows = [
            ["Owner Name",    v["owner_name"] or "Unknown Person"],
            ["Phone Number",  v["phone_number"] or "Not Available"],
            ["Email Address", v["email"] or "Not Available"],
        ]
        tbl = Table(rows, colWidths=[5*cm, 13.5*cm])
        style = self._row_style()
        if not owner_found:
            style.add("TEXTCOLOR", (1, 0), (1, 0), self._RED)
        tbl.setStyle(style)
        elements = [self._section_heading("Owner / Offender Details"), tbl]
        if not owner_found:
            note = Paragraph(
                "<i>* Owner details could not be verified from the "
                "registered vehicle database. Physical verification required.</i>",
                ParagraphStyle("Note", fontSize=8, textColor=self._RED, leading=11),
            )
            elements.append(note)
        return elements

    def _penalty_table(self, styles) -> list:
        v       = self._v
        amount  = v["penalty_amount"]
        due     = v["due_date"]
        esc     = v["escalation_count"]

        base_row  = ["Base Penalty Amount", f"Rs. {amount:.2f}"]
        esc_rows  = []
        if esc > 0:
            extra = amount - (VEHICLE_PENALTY if v["challan_type"] == "vehicle"
                              else PEDESTRIAN_PENALTY)
            esc_rows = [
                [f"Escalation Charges ({esc}x)", f"+ Rs. {extra:.2f}"],
                ["Total Amount Due",             f"Rs. {amount:.2f}"],
            ]
        due_row    = ["Payment Due Date",  due]
        status_row = ["Current Status",    v["status"].upper()]

        rows = [base_row] + esc_rows + [due_row, status_row]
        tbl  = Table(rows, colWidths=[5*cm, 13.5*cm])
        style = self._row_style()
        # Highlight total / amount in bold orange
        for ri, row in enumerate(rows):
            if row[0] in ("Total Amount Due", "Base Penalty Amount"):
                style.add("FONTNAME",  (1, ri), (1, ri), "Helvetica-Bold")
                style.add("TEXTCOLOR", (1, ri), (1, ri), self._ORANGE)
        tbl.setStyle(style)
        return [self._section_heading("Penalty Details"), tbl]

    def _escalation_note(self, styles) -> list:
        note_style = ParagraphStyle(
            "EscNote", fontSize=8, leading=12,
            textColor=self._DGRAY, backColor=colors.HexColor("#FFF8E1"),
            borderColor=self._ORANGE, borderWidth=1,
            borderPadding=6,
        )
        text = (
            "<b>ESCALATION POLICY:</b> If the challan is not paid by the due "
            f"date, an additional <b>{int(ESCALATION_RATE*100)}% penalty</b> "
            f"will be added every <b>{ESCALATION_DAYS} days</b> until the "
            "amount is paid. Repeated non-payment may result in vehicle "
            "impoundment or legal action under the Solid Waste Management "
            "Rules 2016 and relevant municipal bye-laws."
        )
        return [Paragraph(text, note_style)]

    def _evidence_section(self, styles) -> list:
        v    = self._v
        rows = [
            ["Evidence Video",   v["evidence_video_path"] or "—"],
            ["Plate Image",      v["evidence_plate_image_path"] or "—"],
        ]
        tbl = Table(rows, colWidths=[5*cm, 13.5*cm])
        tbl.setStyle(self._row_style())
        elements: list = [self._section_heading("Evidence Reference"), tbl]

        # Embed plate image thumbnail if file exists
        plate_img_path = v.get("evidence_plate_image_path") or ""
        if plate_img_path and Path(plate_img_path).exists():
            try:
                img = RLImage(plate_img_path, width=5*cm, height=3*cm)
                caption = Paragraph(
                    "<i>Plate Evidence Image</i>",
                    ParagraphStyle("C", fontSize=7, textColor=colors.gray,
                                   alignment=TA_LEFT),
                )
                elements += [Spacer(1, 3*mm), img, caption]
            except Exception:
                pass
        return elements

    def _payment_section(self, styles) -> list:
        v      = self._v
        amount = v["penalty_amount"]

        # QR code
        qr_bytes = _make_upi_qr_bytes(v["challan_id"], amount)
        elements = [self._section_heading("Payment Instructions")]

        pay_style = ParagraphStyle(
            "Pay", fontSize=9, leading=14, textColor=self._DGRAY)

        instructions = Paragraph(
            f"Pay <b>Rs. {amount:.2f}</b> via UPI to <b>{UPI_ID}</b> "
            f"or at any BBMP ward office.<br/>"
            f"Please mention Challan ID <b>{v['challan_id']}</b> "
            "in the payment reference/remarks.<br/>"
            "Online payment portal: <b>https://bbmp.gov.in/payments</b>",
            pay_style,
        )

        if qr_bytes:
            qr_img   = RLImage(io.BytesIO(qr_bytes), width=3.5*cm, height=3.5*cm)
            qr_label = Paragraph(
                "<i>Scan to Pay (UPI)</i>",
                ParagraphStyle("QL", fontSize=7, textColor=colors.gray,
                               alignment=TA_CENTER),
            )
            # Side-by-side layout: instructions | QR
            inner = Table(
                [[instructions, [qr_img, qr_label]]],
                colWidths=[13*cm, 5.5*cm],
            )
            inner.setStyle(TableStyle([
                ("VALIGN",  (0, 0), (-1, -1), "TOP"),
                ("ALIGN",   (1, 0), (1, 0),   "CENTER"),
            ]))
            elements.append(inner)
        else:
            elements.append(instructions)

        return elements

    def _signature_section(self, styles) -> list:
        sig_style = ParagraphStyle(
            "Sig", fontSize=8, leading=12,
            textColor=self._DGRAY, alignment=TA_CENTER,
        )
        left  = [
            Spacer(1, 1.5*cm),
            HRFlowable(width=6*cm, thickness=0.5, color=colors.black),
            Paragraph("Signature of Violator / Recipient", sig_style),
        ]
        right = [
            Spacer(1, 1.5*cm),
            HRFlowable(width=6*cm, thickness=0.5, color=colors.black),
            Paragraph(
                f"Authorised Signatory<br/>{AUTHORITY_DEPT}", sig_style),
        ]
        tbl = Table([[left, right]], colWidths=[9*cm, 9.5*cm])
        tbl.setStyle(TableStyle([
            ("VALIGN",  (0, 0), (-1, -1), "TOP"),
            ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
        ]))
        return [HRFlowable(width="100%", thickness=0.8,
                           color=self._NAVY, spaceAfter=6),
                tbl]

    def _footer_note(self, styles) -> list:
        foot_style = ParagraphStyle(
            "Foot", fontSize=7, leading=10,
            textColor=colors.gray, alignment=TA_CENTER,
        )
        return [
            HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey),
            Spacer(1, 2*mm),
            Paragraph(
                "This is a computer-generated challan and is legally valid "
                "without a physical signature under the Information Technology "
                "Act 2000. "
                "For grievances contact: grievance.swm@bbmp.gov.in",
                foot_style,
            ),
            Paragraph(
                "Generated by VidTrace Illegal Dumping Detection System — "
                f"Challan ID: {self._v['challan_id']}",
                foot_style,
            ),
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  PenaltyManager — main public API
# ══════════════════════════════════════════════════════════════════════════════

class PenaltyManager:
    """
    Penalty & Challan Management System for VidTrace.

    Typical call sequence after pipeline detects a violation:
        pm = PenaltyManager()
        challan_id = pm.create_violation(plate_number, video_path, plate_img, loc, conf)
        pm.generate_challan(challan_id)
        pm.check_and_escalate()   # call periodically
    """

    def __init__(self):
        _init_db()

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_challan_id(plate: Optional[str]) -> str:
        """Generate a unique, determinism-free challan ID."""
        raw = f"{plate or 'PED'}-{uuid.uuid4().hex[:8].upper()}"
        prefix = "VH" if plate else "PD"
        return f"BBMP-{prefix}-{raw}"

    @staticmethod
    def _lookup_owner(plate: str) -> Optional[sqlite3.Row]:
        """Look up owner in the owners table. Returns None if not found."""
        with _get_connection() as conn:
            return conn.execute(
                "SELECT * FROM owners WHERE plate_number = ?", (plate,)
            ).fetchone()

    # ── Public API ────────────────────────────────────────────────────────

    def create_violation(
        self,
        plate_number:              Optional[str],
        evidence_video_path:       Optional[str] = None,
        evidence_plate_image_path: Optional[str] = None,
        location:                  str            = "",
        confidence:                float          = 0.0,
    ) -> str:
        """
        Record a new violation in the database.

        Parameters
        ----------
        plate_number            : Detected plate text, or None for pedestrian.
        evidence_video_path     : Path to saved output video file.
        evidence_plate_image_path: Path to saved plate/evidence image.
        location                : Free-text location description.
        confidence              : L5 violation confidence score (0–1).

        Returns
        -------
        challan_id : str — The unique challan ID for this violation.
        """
        now         = datetime.now()
        challan_id  = self._make_challan_id(plate_number)
        is_vehicle  = plate_number is not None and plate_number.strip() != ""
        challan_type = "vehicle" if is_vehicle else "pedestrian"
        base_penalty = VEHICLE_PENALTY if is_vehicle else PEDESTRIAN_PENALTY
        due_date     = (now + timedelta(days=DUE_DAYS)).strftime("%Y-%m-%d")
        notes        = f"conf={confidence:.2f}"

        # ── Owner lookup ──────────────────────────────────────────────────
        if is_vehicle:
            owner_row = self._lookup_owner(plate_number.strip().upper())
            if owner_row:
                owner_name   = owner_row["owner_name"]
                phone_number = owner_row["phone_number"]
                email        = owner_row["email"]
                logger.info(f"Owner found: {owner_name} for plate {plate_number}")
            else:
                owner_name   = "Owner Not Found"
                phone_number = None
                email        = None
                logger.warning(f"No owner record for plate '{plate_number}'")
        else:
            owner_name   = "Unknown Person"
            phone_number = None
            email        = None

        # ── Insert violation ──────────────────────────────────────────────
        with _get_connection() as conn:
            conn.execute("""
                INSERT INTO violations (
                    challan_id, plate_number, owner_name, phone_number,
                    email, challan_type, violation_timestamp, location,
                    penalty_amount, due_date, status,
                    evidence_video_path, evidence_plate_image_path, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                challan_id, plate_number, owner_name, phone_number,
                email, challan_type, now.strftime("%Y-%m-%d %H:%M:%S"),
                location, base_penalty, due_date, "pending",
                evidence_video_path, evidence_plate_image_path, notes,
            ))
            conn.commit()

        logger.info(
            f"Violation created | challan={challan_id} | "
            f"plate={plate_number} | type={challan_type} | "
            f"penalty=Rs.{base_penalty:.0f} | due={due_date}"
        )
        return challan_id

    # ─────────────────────────────────────────────────────────────────────

    def generate_challan(self, challan_id: str) -> Optional[str]:
        """
        Generate a PDF challan for the given challan_id.

        Returns the saved PDF path, or None on failure.
        """
        violation = self.get_violation_by_challan(challan_id)
        if violation is None:
            logger.error(f"generate_challan: challan '{challan_id}' not found")
            return None

        CHALLAN_DIR.mkdir(parents=True, exist_ok=True)
        safe_id  = challan_id.replace("/", "-").replace("\\", "-")
        pdf_path = CHALLAN_DIR / f"{safe_id}.pdf"

        try:
            pdf_bytes = _ChallanPDF(violation).build()
            pdf_path.write_bytes(pdf_bytes)
            logger.info(f"Challan PDF saved → '{pdf_path}'")
        except Exception as exc:
            logger.exception(f"PDF generation failed for {challan_id}: {exc}")
            return None

        # Update db with pdf path and status
        with _get_connection() as conn:
            conn.execute("""
                UPDATE violations
                SET pdf_challan_path = ?, status = 'notified'
                WHERE challan_id = ?
            """, (str(pdf_path), challan_id))
            conn.commit()

        return str(pdf_path)

    # ─────────────────────────────────────────────────────────────────────

    def check_and_escalate(self) -> int:
        """
        Check all pending/notified violations for overdue payment and
        apply escalation penalties (+10% per 2-day cycle after due date).

        Returns the number of violations escalated in this call.
        """
        today     = datetime.now().date()
        escalated = 0

        with _get_connection() as conn:
            rows = conn.execute("""
                SELECT challan_id, due_date, penalty_amount, escalation_count
                FROM   violations
                WHERE  status IN ('pending', 'notified', 'escalated')
            """).fetchall()

            for row in rows:
                due_date = datetime.strptime(row["due_date"], "%Y-%m-%d").date()
                if today <= due_date:
                    continue  # not overdue yet

                days_overdue     = (today - due_date).days
                cycles_due       = days_overdue // ESCALATION_DAYS
                cycles_applied   = row["escalation_count"]
                new_cycles       = cycles_due - cycles_applied

                if new_cycles <= 0:
                    continue

                # Recalculate full penalty from base to avoid floating drift
                base    = row["penalty_amount"] / (
                    (1 + ESCALATION_RATE) ** cycles_applied
                )
                new_amt = base * ((1 + ESCALATION_RATE) ** cycles_due)
                new_cnt = cycles_due

                conn.execute("""
                    UPDATE violations
                    SET penalty_amount    = ?,
                        escalation_count  = ?,
                        status            = 'escalated'
                    WHERE challan_id = ?
                """, (round(new_amt, 2), new_cnt, row["challan_id"]))

                logger.warning(
                    f"ESCALATED | {row['challan_id']} | "
                    f"{new_cycles} new cycle(s) | "
                    f"Rs.{row['penalty_amount']:.2f} → Rs.{new_amt:.2f}"
                )
                escalated += 1

            conn.commit()

        if escalated:
            logger.info(f"check_and_escalate: {escalated} violation(s) escalated")
        else:
            logger.info("check_and_escalate: no new escalations")
        return escalated

    # ─────────────────────────────────────────────────────────────────────

    def mark_as_paid(
        self,
        challan_id:     str,
        transaction_id: Optional[str] = None,
    ) -> bool:
        """
        Mark a challan as paid.

        Parameters
        ----------
        challan_id     : The BBMP challan ID.
        transaction_id : Optional UPI / bank reference number.

        Returns True on success, False if challan not found.
        """
        with _get_connection() as conn:
            note_suffix = (
                f" | txn={transaction_id}" if transaction_id else ""
            )
            cur = conn.execute("""
                UPDATE violations
                SET status = 'paid',
                    notes  = notes || ?
                WHERE challan_id = ? AND status != 'paid'
            """, (note_suffix, challan_id))
            conn.commit()

        if cur.rowcount == 0:
            logger.warning(
                f"mark_as_paid: '{challan_id}' not found or already paid"
            )
            return False

        logger.info(
            f"PAID | {challan_id}"
            + (f" | txn={transaction_id}" if transaction_id else "")
        )
        return True

    # ─────────────────────────────────────────────────────────────────────

    def get_violation_by_challan(
        self, challan_id: str
    ) -> Optional[sqlite3.Row]:
        """Return the full violation row for a given challan_id, or None."""
        with _get_connection() as conn:
            return conn.execute(
                "SELECT * FROM violations WHERE challan_id = ?", (challan_id,)
            ).fetchone()

    # ── Convenience helpers ───────────────────────────────────────────────

    def list_pending(self) -> list:
        """Return all pending/notified violations as a list of dicts."""
        with _get_connection() as conn:
            rows = conn.execute("""
                SELECT challan_id, plate_number, owner_name,
                       penalty_amount, due_date, status, violation_timestamp
                FROM   violations
                WHERE  status IN ('pending', 'notified', 'escalated')
                ORDER  BY violation_timestamp DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        """Return a high-level summary of all violations in the database."""
        with _get_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                   AS total,
                    SUM(CASE WHEN status='paid'      THEN 1 ELSE 0 END) AS paid,
                    SUM(CASE WHEN status='escalated' THEN 1 ELSE 0 END) AS escalated,
                    SUM(CASE WHEN status IN ('pending','notified')
                                             THEN 1 ELSE 0 END)         AS pending,
                    SUM(penalty_amount)                        AS total_amount,
                    SUM(CASE WHEN status='paid'
                        THEN penalty_amount ELSE 0 END)        AS collected
                FROM violations
            """).fetchone()
        return dict(row)


# ══════════════════════════════════════════════════════════════════════════════
#  Integration shim — called from run_pipeline.py after Layer 5 confirms
# ══════════════════════════════════════════════════════════════════════════════

def process_pipeline_violation(
    plate_number:     Optional[str],
    evidence_video:   Optional[str],
    evidence_plate:   Optional[str],
    location:         str   = "",
    confidence:       float = 0.0,
    auto_pdf:         bool  = True,
) -> Optional[str]:
    """
    Convenience wrapper — create a violation record and optionally
    generate the PDF challan in one call.

    Designed to be dropped into run_pipeline.py after Layer 5 fires
    a confirmed VIOLATION event:

        from penalty_manager import process_pipeline_violation
        ...
        if event.is_violation:
            challan_id = process_pipeline_violation(
                plate_number   = enhancer_result.plate_text,
                evidence_video = "vidtrace_output.mp4",
                evidence_plate = enhancer_result.saved_paths[0],
                location       = "Outer Ring Road, Bengaluru",
                confidence     = event.confidence,
            )
            print(f"[Challan] Issued: {challan_id}")

    Returns the challan_id string, or None on failure.
    """
    try:
        pm         = PenaltyManager()
        challan_id = pm.create_violation(
            plate_number              = plate_number,
            evidence_video_path       = evidence_video,
            evidence_plate_image_path = evidence_plate,
            location                  = location,
            confidence                = confidence,
        )
        if auto_pdf:
            pdf_path = pm.generate_challan(challan_id)
            if pdf_path:
                print(f"[Challan] PDF → {pdf_path}")
        return challan_id
    except Exception as exc:
        logger.exception(f"process_pipeline_violation failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CLI demo — python penalty_manager.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  VidTrace Penalty & Challan Management System — Demo")
    print("=" * 60)

    pm = PenaltyManager()

    # ── Test 1: known plate (owner in DB) ─────────────────────────────────
    print("\n[TEST 1] Known plate — KA05KK5546")
    cid1 = pm.create_violation(
        plate_number              = "KA05KK5546",
        evidence_video_path       = "vidtrace_output.mp4",
        evidence_plate_image_path = "evidence/plate.jpg",
        location                  = "Outer Ring Road, Bengaluru",
        confidence                = 0.72,
    )
    pdf1 = pm.generate_challan(cid1)
    print(f"  challan_id : {cid1}")
    print(f"  pdf        : {pdf1}")

    # ── Test 2: unknown plate (owner NOT in DB) ───────────────────────────
    print("\n[TEST 2] Unknown plate — MH12AB1234")
    cid2 = pm.create_violation(
        plate_number  = "MH12AB1234",
        location      = "Whitefield, Bengaluru",
        confidence    = 0.58,
    )
    pdf2 = pm.generate_challan(cid2)
    print(f"  challan_id : {cid2}")
    print(f"  pdf        : {pdf2}")

    # ── Test 3: pedestrian (no plate) ─────────────────────────────────────
    print("\n[TEST 3] Pedestrian — no plate")
    cid3 = pm.create_violation(
        plate_number  = None,
        location      = "MG Road, Bengaluru",
        confidence    = 0.63,
    )
    pdf3 = pm.generate_challan(cid3)
    print(f"  challan_id : {cid3}")
    print(f"  pdf        : {pdf3}")

    # ── Escalation check ──────────────────────────────────────────────────
    print("\n[CHECK] Escalation scan …")
    n = pm.check_and_escalate()
    print(f"  Escalated  : {n} violation(s)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[SUMMARY]")
    s = pm.summary()
    print(f"  Total violations : {s['total']}")
    print(f"  Pending          : {s['pending']}")
    print(f"  Paid             : {s['paid']}")
    print(f"  Escalated        : {s['escalated']}")
    print(f"  Total amount     : Rs. {s['total_amount'] or 0:.2f}")
    print(f"  Collected        : Rs. {s['collected'] or 0:.2f}")

    print("\n[DONE] Check the 'challans/' folder for generated PDFs.")
    print("=" * 60)
