"""
Penalty & Challan Management System — Enhanced v2
===================================================
Integrates with the VidTrace illegal dumping detection pipeline
(Layer1–Layer5 + enhancer.py) to automatically issue challans when a
violation is confirmed.

Enhancements over v1
---------------------
1. Dynamic UPI QR Code
   - Every challan gets a fresh QR code encoding the EXACT current penalty
     amount and challan ID as payment reference.
   - Uses `qrcode` + `Pillow` (falls back to PIL-only placeholder if qrcode
     is absent — but qrcode is strongly recommended).

2. Real time-based escalation
   - Due date = violation_date + 7 days.
   - After due date: +10% every 2 days.
   - simulate_days_passed(challan_id, days) — moves the violation date
     backward by `days` days for demo/testing, recalculates penalty,
     regenerates the PDF.
   - get_current_penalty(violation_row) — pure function, returns escalated
     amount based on today's date, no side effects.

3. Professional PDF redesign
   - Removed: evidence video path (clutters PDF, not useful to recipient).
   - Added: strong Authority Signature section ("Sd/-" + digital stamp).
   - Added: clear sections — Violation Details / Penalty Breakdown /
     Escalation Rules / Payment Instructions.
   - Evidence plate image is shown PROMINENTLY (large, labelled, timestamped).
   - Confidence score shown beside evidence image.
   - QR code sits cleanly beside payment instructions.

4. Logging + comments throughout.

Usage:
    from penalty_manager import PenaltyManager

    pm = PenaltyManager()
    challan_id = pm.create_violation(
        plate_number          = "KA05KK5546",
        evidence_plate_image  = "evidence/pair_id_evidence.jpg",
        location              = "Outer Ring Road, Bengaluru",
        confidence            = 0.72,
    )
    pm.generate_challan(challan_id)          # PDF in challans/
    pm.simulate_days_passed(challan_id, 10)  # demo: pretend 10 days have passed
    pm.check_and_escalate()                  # run daily

Requirements:
    pip install reportlab pillow qrcode
    (qrcode pulls in pillow automatically)

Database: penalties.db   (auto-created on first run)
Challans:  challans/     (auto-created folder)
"""

from __future__ import annotations

import io
import logging
import math
import os
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# ── ReportLab ─────────────────────────────────────────────────────────────────
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
    KeepTogether,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[PenaltyMgr] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH             = Path("penalties.db")
CHALLAN_DIR         = Path("challans")

# ── Penalty rules ─────────────────────────────────────────────────────────────
VEHICLE_PENALTY     = 500.0    # INR base for vehicle violations
PEDESTRIAN_PENALTY  = 300.0    # INR base for pedestrian/unknown violations
DUE_DAYS            = 7        # days until due date from violation date
ESCALATION_DAYS     = 2        # +10% every N days after due date
ESCALATION_RATE     = 0.10     # 10% per escalation cycle

# ── Authority identifiers ─────────────────────────────────────────────────────
AUTHORITY_NAME      = "Bruhat Bengaluru Mahanagara Palike (BBMP)"
AUTHORITY_DEPT      = "Solid Waste Management and Environment Division"
AUTHORITY_ADDRESS   = "N.R. Square, Hudson Circle, Bengaluru - 560 002"
AUTHORITY_PHONE     = "080-22221188"
AUTHORITY_EMAIL     = "swm@bbmp.gov.in"
AUTHORITY_SYSTEM    = "VidTrace Municipal Enforcement System"

# ── UPI payment ───────────────────────────────────────────────────────────────
# Replace with your actual UPI VPA before going live.
UPI_ID              = "rachfpatil@oksbi"
UPI_DISPLAY_NAME    = "BBMP SWM Enforcement"
PAYMENT_PORTAL      = "https://bbmp.gov.in/payments"


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
    """
    Create tables and seed mock owner data on first run.

    Tables
    ------
    owners     — registered vehicle owner lookup (plate_number → person details)
    violations — all issued challans + their current penalty / status
    """
    CHALLAN_DIR.mkdir(parents=True, exist_ok=True)

    with _get_connection() as conn:
        # owners: one row per registered vehicle plate
        conn.execute("""
            CREATE TABLE IF NOT EXISTS owners (
                plate_number  TEXT PRIMARY KEY,
                owner_name    TEXT NOT NULL,
                phone_number  TEXT,
                email         TEXT,
                address       TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # violations: one row per issued challan
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                challan_id                TEXT UNIQUE NOT NULL,
                plate_number              TEXT,
                owner_name                TEXT,
                phone_number              TEXT,
                email                     TEXT,
                challan_type              TEXT NOT NULL DEFAULT 'vehicle',
                violation_timestamp       TIMESTAMP NOT NULL,
                location                  TEXT DEFAULT '',
                penalty_amount            REAL NOT NULL,
                base_penalty_amount       REAL NOT NULL,
                due_date                  DATE NOT NULL,
                status                    TEXT NOT NULL DEFAULT 'pending',
                escalation_count          INTEGER NOT NULL DEFAULT 0,
                confidence                REAL DEFAULT 0.0,
                evidence_plate_image_path TEXT,
                pdf_challan_path          TEXT,
                notes                     TEXT DEFAULT '',
                created_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Seed one demo owner record (the plate detected in the reference video)
        conn.execute("""
            INSERT OR IGNORE INTO owners
                (plate_number, owner_name, phone_number, email, address)
            VALUES
                ('KA05KK5546', 'Rachana F Patil', '9901967521',
                 'rachfpatil@gmail.com',
                 '14, 4th Cross, Indiranagar, Bengaluru, Karnataka - 560038')
        """)
        conn.commit()

    logger.info("Database ready at '%s'", DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  Escalation calculator — pure, no side-effects
# ══════════════════════════════════════════════════════════════════════════════

def get_current_penalty(violation: sqlite3.Row, as_of: Optional[date] = None) -> float:
    """
    Calculate the current penalty amount for a violation based on today's date
    (or a custom `as_of` date for simulation purposes).

    Formula
    -------
    If today <= due_date:
        penalty = base_penalty_amount  (no escalation)
    Else:
        cycles = floor((today - due_date).days / ESCALATION_DAYS)
        penalty = base * (1 + ESCALATION_RATE) ** cycles

    Parameters
    ----------
    violation : sqlite3.Row  — a row from the violations table
    as_of     : date or None — defaults to today

    Returns
    -------
    float — current penalty in INR (rounded to 2 dp)
    """
    if as_of is None:
        as_of = date.today()

    base     = float(violation["base_penalty_amount"])
    due_date = datetime.strptime(str(violation["due_date"]), "%Y-%m-%d").date()

    if as_of <= due_date:
        return round(base, 2)

    days_overdue = (as_of - due_date).days
    cycles       = days_overdue // ESCALATION_DAYS
    return round(base * ((1 + ESCALATION_RATE) ** cycles), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  QR Code generator
# ══════════════════════════════════════════════════════════════════════════════

def _make_upi_qr_bytes(challan_id: str, amount: float) -> Optional[bytes]:
    """
    Generate a UPI payment QR code as PNG bytes.

    UPI deep-link format:
        upi://pay?pa=<vpa>&pn=<name>&am=<amount>&tn=<note>&cu=INR

    The amount and challan_id are embedded dynamically so every challan —
    including escalated re-issues — gets a unique, correct QR code.

    Falls back to a PIL-drawn placeholder if the `qrcode` library is absent.
    Install with: pip install qrcode pillow
    """
    upi_url = (
        f"upi://pay"
        f"?pa={UPI_ID}"
        f"&pn={UPI_DISPLAY_NAME.replace(' ', '+')}"
        f"&am={amount:.2f}"
        f"&tn=Challan+{challan_id}"
        f"&cu=INR"
    )
    logger.debug("UPI URL: %s", upi_url)

    # ── Attempt 1: use qrcode library (best quality) ──────────────────────
    try:
        import qrcode  # pip install qrcode
        qr = qrcode.QRCode(
            version=None,       # auto-size
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=5,
            border=3,
        )
        qr.add_data(upi_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        logger.info("QR code generated via qrcode library (amount=Rs.%.2f)", amount)
        return buf.getvalue()
    except ImportError:
        logger.warning(
            "qrcode library not installed — using PIL placeholder. "
            "Run: pip install qrcode"
        )

    # ── Attempt 2: PIL-drawn placeholder ─────────────────────────────────
    try:
        from PIL import Image, ImageDraw, ImageFont
        SIZE   = 200
        CELL   = 10
        img    = Image.new("RGB", (SIZE, SIZE), "white")
        draw   = ImageDraw.Draw(img)

        # Border
        draw.rectangle([0, 0, SIZE - 1, SIZE - 1], outline="black", width=3)

        # Simulate a simple chequered QR-like pattern for visual authenticity
        for row in range(3, SIZE // CELL - 3):
            for col in range(3, SIZE // CELL - 3):
                # Deterministic pseudo-random fill using challan_id hash
                val = hash(f"{challan_id}{row}{col}") & 1
                if val:
                    x0 = col * CELL
                    y0 = row * CELL
                    draw.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1],
                                   fill="black")

        # Overlay amount text in centre (white box)
        draw.rectangle([40, 75, 160, 125], fill="white", outline="black", width=1)
        draw.text((55, 80), "SCAN TO PAY", fill="black")
        draw.text((60, 95), f"Rs.{amount:.0f}", fill="black")
        draw.text((75, 110), "via UPI", fill="gray")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        logger.info("QR placeholder generated via PIL (amount=Rs.%.2f)", amount)
        return buf.getvalue()

    except Exception as exc:
        logger.error("QR code generation failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  PDF Challan Builder
# ══════════════════════════════════════════════════════════════════════════════

class _ChallanPDF:
    """
    Builds a professional municipal enforcement challan PDF.

    Section order
    -------------
    1. Government header (BBMP banner, challan ID)
    2. Violation type banner (orange strip)
    3. Violation Details table
    4. Owner / Offender Details table
    5. Penalty Breakdown table
    6. Escalation Policy note
    7. Evidence section (plate image + confidence — NO video path)
    8. Payment Instructions + dynamic UPI QR code
    9. Authority Signature block (digital stamp + Sd/-)
    10. Legal footer
    """

    # Colour palette
    _NAVY    = colors.HexColor("#1B2A6B")   # header / heading bg
    _ORANGE  = colors.HexColor("#E8500A")   # violation banner / accent
    _GREEN   = colors.HexColor("#1A7A4A")   # paid status
    _LGRAY   = colors.HexColor("#F5F5F5")   # alternating table row
    _DGRAY   = colors.HexColor("#333333")   # body text
    _RED     = colors.HexColor("#CC0000")   # warning / missing data
    _GOLD    = colors.HexColor("#C8960C")   # stamp border

    def __init__(self, violation: sqlite3.Row):
        self._v   = dict(violation)
        self._buf = io.BytesIO()

    # ── Public entry point ────────────────────────────────────────────────

    def build(self) -> bytes:
        """Render the full PDF and return as bytes."""
        doc = SimpleDocTemplate(
            self._buf,
            pagesize=A4,
            leftMargin=1.5 * cm,
            rightMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            title=f"Challan {self._v['challan_id']}",
            author=AUTHORITY_NAME,
            subject="Illegal Dumping Violation Challan",
        )
        story: list = []

        story += self._section_header()
        story.append(Spacer(1, 5 * mm))

        story += self._violation_banner()
        story.append(Spacer(1, 4 * mm))

        story += self._violation_details()
        story.append(Spacer(1, 4 * mm))

        story += self._owner_details()
        story.append(Spacer(1, 4 * mm))

        story += self._penalty_breakdown()
        story.append(Spacer(1, 4 * mm))

        story += self._escalation_policy()
        story.append(Spacer(1, 4 * mm))

        story += self._evidence_section()
        story.append(Spacer(1, 4 * mm))

        story += self._payment_section()
        story.append(Spacer(1, 6 * mm))

        story += self._signature_section()
        story.append(Spacer(1, 3 * mm))

        story += self._footer()

        doc.build(story)
        return self._buf.getvalue()

    # ── Style factories ───────────────────────────────────────────────────

    @staticmethod
    def _ps(name: str, **kw) -> ParagraphStyle:
        """Shorthand ParagraphStyle constructor."""
        return ParagraphStyle(name, **kw)

    def _section_label(self, text: str) -> Paragraph:
        """Bold uppercase section heading in navy."""
        return Paragraph(
            text.upper(),
            self._ps("SL", fontSize=9, fontName="Helvetica-Bold",
                     textColor=self._NAVY, spaceBefore=2, spaceAfter=3),
        )

    def _base_table_style(self) -> TableStyle:
        """Standard two-column detail table style."""
        return TableStyle([
            ("BACKGROUND",    (0, 0), (0, -1), self._LGRAY),
            ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",     (0, 0), (-1, -1), self._DGRAY),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.4,
             colors.HexColor("#CCCCCC")),
        ])

    # ── Section 1: Government header ──────────────────────────────────────

    def _section_header(self) -> list:
        """
        Tri-column header:
            [emblem]  [Authority name + address]  [Challan ID + date]
        """
        wh = self._ps  # shorthand

        left_cell = Paragraph(
            "&#127963;",  # building icon via HTML entity
            wh("Emb", fontSize=28, textColor=colors.white,
               fontName="Helvetica", leading=36),
        )

        centre_cell = [
            Paragraph(
                AUTHORITY_NAME,
                wh("HT", fontSize=13, fontName="Helvetica-Bold",
                   textColor=colors.white, leading=17),
            ),
            Paragraph(
                AUTHORITY_DEPT,
                wh("HD", fontSize=8, fontName="Helvetica",
                   textColor=colors.HexColor("#CCDDFF"), leading=12),
            ),
            Paragraph(
                AUTHORITY_ADDRESS,
                wh("HA", fontSize=8, fontName="Helvetica",
                   textColor=colors.HexColor("#CCDDFF"), leading=12),
            ),
            Paragraph(
                f"Ph: {AUTHORITY_PHONE}     Email: {AUTHORITY_EMAIL}",
                wh("HC", fontSize=8, fontName="Helvetica",
                   textColor=colors.HexColor("#CCDDFF"), leading=12),
            ),
        ]

        challan_date = self._v["violation_timestamp"][:10]
        right_cell = [
            Paragraph(
                "<b>VIOLATION CHALLAN</b>",
                wh("VCH", fontSize=11, fontName="Helvetica-Bold",
                   textColor=colors.white, alignment=TA_RIGHT),
            ),
            Paragraph(
                f"ID: <b>{self._v['challan_id']}</b>",
                wh("CID", fontSize=8, fontName="Helvetica",
                   textColor=colors.HexColor("#AABBFF"), alignment=TA_RIGHT,
                   leading=13),
            ),
            Paragraph(
                f"Date: {challan_date}",
                wh("CD", fontSize=8, fontName="Helvetica",
                   textColor=colors.HexColor("#AABBFF"), alignment=TA_RIGHT,
                   leading=13),
            ),
            Spacer(1, 4),
            Paragraph(
                AUTHORITY_SYSTEM,
                wh("CS", fontSize=7, fontName="Helvetica",
                   textColor=colors.HexColor("#8899CC"), alignment=TA_RIGHT,
                   leading=11),
            ),
        ]

        tbl = Table(
            [[left_cell, centre_cell, right_cell]],
            colWidths=[1.8 * cm, 11.5 * cm, 5.2 * cm],
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), self._NAVY),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        return [tbl]

    # ── Section 2: Violation type banner ──────────────────────────────────

    def _violation_banner(self) -> list:
        """Orange full-width banner indicating violation category."""
        label = (
            "VEHICLE VIOLATION — ILLEGAL DUMPING"
            if self._v["challan_type"] == "vehicle"
            else "PEDESTRIAN VIOLATION — ILLEGAL DUMPING"
        )
        para = Paragraph(
            f"&#9888;  {label}  &#9888;",
            self._ps("VB", fontSize=12, fontName="Helvetica-Bold",
                     textColor=colors.white, alignment=TA_CENTER, leading=18),
        )
        tbl = Table([[para]], colWidths=[18.5 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), self._ORANGE),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return [tbl]

    # ── Section 3: Violation Details ──────────────────────────────────────

    def _violation_details(self) -> list:
        v = self._v
        ts = v["violation_timestamp"]

        rows = [
            ["Challan ID",           v["challan_id"]],
            ["Violation Date",       ts[:10]],
            ["Violation Time",       ts[11:19] if len(ts) > 10 else "—"],
            ["Location",             v["location"] or "Not Recorded"],
            ["Number Plate",
             v["plate_number"] or "NOT DETECTED (Pedestrian)"],
            ["Detection System",     AUTHORITY_SYSTEM],
        ]

        tbl = Table(rows, colWidths=[5.5 * cm, 13 * cm])
        tbl.setStyle(self._base_table_style())
        return [self._section_label("Violation Details"), tbl]

    # ── Section 4: Owner / Offender Details ───────────────────────────────

    def _owner_details(self) -> list:
        v            = self._v
        known_owner  = v["owner_name"] not in ("Owner Not Found",
                                                "Unknown Person", None, "")
        rows = [
            ["Owner / Offender",  v["owner_name"] or "Unknown"],
            ["Phone Number",      v["phone_number"] or "Not Available"],
            ["Email Address",     v["email"]        or "Not Available"],
        ]

        tbl   = Table(rows, colWidths=[5.5 * cm, 13 * cm])
        style = self._base_table_style()
        if not known_owner:
            # Highlight unknown owner in red
            style.add("TEXTCOLOR", (1, 0), (1, 0), self._RED)
            style.add("FONTNAME",  (1, 0), (1, 0), "Helvetica-Bold")
        tbl.setStyle(style)

        elements = [self._section_label("Owner / Offender Details"), tbl]

        if not known_owner:
            elements.append(
                Paragraph(
                    "<i>* Owner details could not be verified from the "
                    "registered vehicle database. Physical verification and "
                    "manual identification required.</i>",
                    self._ps("OWN", fontSize=8, textColor=self._RED,
                             leading=11, spaceAfter=2),
                )
            )
        return elements

    # ── Section 5: Penalty Breakdown ──────────────────────────────────────

    def _penalty_breakdown(self) -> list:
        v        = self._v
        base     = float(v["base_penalty_amount"])
        current  = float(v["penalty_amount"])
        esc_cnt  = int(v["escalation_count"])
        due      = v["due_date"]
        status   = v["status"].upper()

        rows: list = [["Base Penalty Amount", f"Rs. {base:.2f}"]]

        if esc_cnt > 0:
            surcharge = current - base
            rows.append([
                f"Escalation Surcharge ({esc_cnt} cycle(s) x "
                f"{int(ESCALATION_RATE * 100)}%)",
                f"+ Rs. {surcharge:.2f}",
            ])
            rows.append(["Total Amount Due (Current)", f"Rs. {current:.2f}"])
        else:
            rows.append(["Total Amount Due", f"Rs. {current:.2f}"])

        rows += [
            ["Payment Due Date",     due],
            ["Status",               status],
        ]

        tbl   = Table(rows, colWidths=[7.5 * cm, 11 * cm])
        style = self._base_table_style()

        # Bold orange for the total row
        for ri, row in enumerate(rows):
            if "Total Amount" in str(row[0]):
                style.add("FONTNAME",  (1, ri), (1, ri), "Helvetica-Bold")
                style.add("FONTSIZE",  (1, ri), (1, ri), 11)
                style.add("TEXTCOLOR", (1, ri), (1, ri), self._ORANGE)
            if str(row[0]) == "Status" and status == "PAID":
                style.add("TEXTCOLOR", (1, ri), (1, ri), self._GREEN)
                style.add("FONTNAME",  (1, ri), (1, ri), "Helvetica-Bold")

        tbl.setStyle(style)
        return [self._section_label("Penalty Breakdown"), tbl]

    # ── Section 6: Escalation Policy ──────────────────────────────────────

    def _escalation_policy(self) -> list:
        """
        Clearly states the escalation rules so the offender understands
        the consequence of non-payment.
        """
        text = (
            "<b>ESCALATION RULES:</b><br/>"
            f"&#9642; Base penalty is due within <b>{DUE_DAYS} days</b> "
            "from the date of violation.<br/>"
            f"&#9642; If unpaid after the due date, a surcharge of "
            f"<b>{int(ESCALATION_RATE * 100)}%</b> of the current total "
            f"is added every <b>{ESCALATION_DAYS} days</b>.<br/>"
            "&#9642; Continued non-payment may result in vehicle "
            "impoundment, licence suspension, or prosecution under the "
            "Solid Waste Management Rules 2016 and relevant BBMP bye-laws."
        )
        para = Paragraph(
            text,
            self._ps("ESC", fontSize=8, leading=13, textColor=self._DGRAY,
                     backColor=colors.HexColor("#FFF8E1"),
                     borderColor=self._ORANGE, borderWidth=1,
                     borderPadding=7, spaceAfter=2),
        )
        return [self._section_label("Escalation Policy"), para]

    # ── Section 7: Evidence Section (plate image + no video path) ─────────

    def _evidence_section(self) -> list:
        """
        Shows the plate evidence image prominently.
        Video path is intentionally excluded (not useful to the recipient).
        Confidence score and timestamp are displayed.
        """
        v              = self._v
        plate_img_path = v.get("evidence_plate_image_path") or ""
        conf           = float(v.get("confidence", 0.0))
        ts             = v["violation_timestamp"]

        meta_rows = [
            ["Detection Confidence", f"{conf:.0%}  ({conf:.4f})"],
            ["Detected At",          ts],
            ["Number Plate",
             v["plate_number"] or "NOT DETECTED"],
        ]
        meta_tbl = Table(meta_rows, colWidths=[5.5 * cm, 13 * cm])
        meta_tbl.setStyle(self._base_table_style())

        elements = [
            self._section_label("Evidence — Plate Detection"),
            meta_tbl,
        ]

        if plate_img_path and Path(plate_img_path).exists():
            try:
                # Embed plate image prominently — large and labelled
                img = RLImage(plate_img_path, width=9 * cm, height=5 * cm,
                              kind="proportional")
                caption = Paragraph(
                    f"<b>Evidence Image</b> — Plate: "
                    f"{v['plate_number'] or 'N/A'}  |  "
                    f"Captured: {ts}  |  "
                    f"Confidence: {conf:.0%}",
                    self._ps("EvidCap", fontSize=8, textColor=colors.gray,
                             alignment=TA_CENTER, leading=11),
                )
                # Frame the image with a thin border table
                frame_tbl = Table(
                    [[img]],
                    colWidths=[9.2 * cm],
                )
                frame_tbl.setStyle(TableStyle([
                    ("BOX",           (0, 0), (-1, -1), 1.5,
                     self._ORANGE),
                    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                elements += [
                    Spacer(1, 3 * mm),
                    frame_tbl,
                    caption,
                ]
            except Exception as exc:
                logger.warning("Could not embed plate image: %s", exc)
                elements.append(
                    Paragraph(
                        f"<i>Evidence image not embeddable: "
                        f"{plate_img_path}</i>",
                        self._ps("NIE", fontSize=8, textColor=colors.gray),
                    )
                )
        else:
            elements.append(
                Paragraph(
                    "<i>No plate evidence image on file.</i>",
                    self._ps("NEV", fontSize=8, textColor=colors.gray,
                             spaceAfter=2),
                )
            )
        return elements

    # ── Section 8: Payment Instructions + QR code ─────────────────────────

    def _payment_section(self) -> list:
        """
        Dynamic UPI QR code (encodes EXACT current penalty + challan_id)
        side-by-side with payment instructions.
        """
        v       = self._v
        amount  = float(v["penalty_amount"])
        cid     = v["challan_id"]

        # Generate fresh QR for the current (possibly escalated) amount
        qr_bytes = _make_upi_qr_bytes(cid, amount)

        pay_text = Paragraph(
            f"Pay <b>Rs. {amount:.2f}</b> via UPI to:<br/>"
            f"<b>{UPI_ID}</b><br/><br/>"
            f"Payment reference / remarks:<br/>"
            f"<b>{cid}</b><br/><br/>"
            f"Online portal: <b>{PAYMENT_PORTAL}</b><br/><br/>"
            f"Or visit any BBMP ward office with this challan.",
            self._ps("PAY", fontSize=9, leading=14, textColor=self._DGRAY),
        )

        if qr_bytes:
            qr_img   = RLImage(io.BytesIO(qr_bytes),
                               width=4 * cm, height=4 * cm)
            qr_lbl   = Paragraph(
                f"<b>Scan to Pay</b><br/>Rs. {amount:.2f} via UPI",
                self._ps("QRL", fontSize=8, textColor=self._NAVY,
                         alignment=TA_CENTER, leading=12),
            )
            # Side-by-side: payment text left, QR right
            row = [[pay_text, [qr_img, Spacer(1, 2 * mm), qr_lbl]]]
            inner = Table(row, colWidths=[12.5 * cm, 6 * cm])
            inner.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN",  (1, 0), (1, 0),  "CENTER"),
            ]))
            content = inner
        else:
            content = pay_text

        return [self._section_label("Payment Instructions"), content]

    # ── Section 9: Authority Signature ────────────────────────────────────

    def _signature_section(self) -> list:
        """
        Authoritative digital signature block.

        Left  — space for violator / recipient signature.
        Right — digital stamp of the enforcement authority.

        The right cell mimics a stamp / seal with the Sd/- convention
        used in Indian government documents.
        """
        sig_ps = self._ps(
            "SigP", fontSize=8, leading=12,
            textColor=self._DGRAY, alignment=TA_CENTER,
        )
        stamp_ps = self._ps(
            "StampP", fontSize=8, leading=13,
            textColor=self._NAVY, alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
        today_str = date.today().strftime("%d-%m-%Y")

        left_cell = [
            Spacer(1, 1.8 * cm),
            HRFlowable(width=6 * cm, thickness=0.5, color=colors.black),
            Paragraph("Signature of Violator / Recipient", sig_ps),
            Paragraph(f"Date: ________________", sig_ps),
        ]

        # Digital stamp — boxed for authority feel
        stamp_inner = Table(
            [[
                Paragraph(
                    f"<b>Sd/-</b><br/><br/>"
                    f"Authorised by:<br/>"
                    f"<b>{AUTHORITY_SYSTEM}</b><br/>"
                    f"<b>{AUTHORITY_DEPT}</b><br/>"
                    f"Municipal Corporation of Bengaluru<br/><br/>"
                    f"Date: {today_str}",
                    stamp_ps,
                )
            ]],
            colWidths=[7.5 * cm],
        )
        stamp_inner.setStyle(TableStyle([
            ("BOX",           (0, 0), (-1, -1), 1.5, self._GOLD),
            ("BACKGROUND",    (0, 0), (-1, -1),
             colors.HexColor("#FFFDE7")),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ]))

        right_cell = [stamp_inner]

        tbl = Table([[left_cell, right_cell]], colWidths=[9 * cm, 9.5 * cm])
        tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",  (0, 0), (0, 0),  "CENTER"),
            ("ALIGN",  (1, 0), (1, 0),  "CENTER"),
        ]))

        return [
            HRFlowable(width="100%", thickness=0.8, color=self._NAVY,
                       spaceAfter=6),
            self._section_label("Signatures"),
            tbl,
        ]

    # ── Section 10: Legal footer ──────────────────────────────────────────

    def _footer(self) -> list:
        foot = self._ps(
            "Foot", fontSize=7, leading=10,
            textColor=colors.gray, alignment=TA_CENTER,
        )
        return [
            HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey),
            Spacer(1, 2 * mm),
            Paragraph(
                "This challan is computer-generated and is legally valid "
                "without a physical signature under the Information "
                "Technology Act, 2000 and Rules thereunder.",
                foot,
            ),
            Paragraph(
                f"For grievances, contact: grievance.swm@bbmp.gov.in  |  "
                f"Generated by {AUTHORITY_SYSTEM}  |  "
                f"Challan: {self._v['challan_id']}",
                foot,
            ),
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  PenaltyManager — public API
# ══════════════════════════════════════════════════════════════════════════════

class PenaltyManager:
    """
    Core manager class for the VidTrace penalty & challan subsystem.

    Public methods
    --------------
    create_violation(...)            — record a new violation, return challan_id
    generate_challan(challan_id)     — build PDF, return path
    check_and_escalate()             — apply overdue penalties, return count
    simulate_days_passed(cid, days)  — demo helper: fast-forward time
    get_current_penalty(violation)   — static escalation calculator
    mark_as_paid(cid, txn)           — mark challan paid
    get_violation_by_challan(cid)    — fetch raw DB row
    list_pending()                   — list all non-paid violations
    summary()                        — aggregate stats dict
    """

    def __init__(self):
        _init_db()

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_challan_id(plate: Optional[str]) -> str:
        """
        Generate a unique challan ID.

        Format: BBMP-VH-<PLATE>-<8-hex-chars>  (vehicle)
                BBMP-PD-UNKNOWN-<8-hex-chars>   (pedestrian)
        """
        uid    = uuid.uuid4().hex[:8].upper()
        prefix = "VH" if plate else "PD"
        tag    = plate.upper().replace(" ", "") if plate else "UNKNOWN"
        return f"BBMP-{prefix}-{tag}-{uid}"

    @staticmethod
    def _lookup_owner(plate: str) -> Optional[sqlite3.Row]:
        """Look up vehicle owner from registered plates table."""
        with _get_connection() as conn:
            return conn.execute(
                "SELECT * FROM owners WHERE plate_number = ?",
                (plate.strip().upper(),),
            ).fetchone()

    # ── Core API ──────────────────────────────────────────────────────────

    def create_violation(
        self,
        plate_number:              Optional[str],
        evidence_video_path:       Optional[str] = None,   # kept for DB but not in PDF
        evidence_plate_image_path: Optional[str] = None,
        location:                  str            = "",
        confidence:                float          = 0.0,
    ) -> str:
        """
        Record a new violation in the database and return the challan_id.

        Parameters
        ----------
        plate_number              : Detected plate text, or None → pedestrian.
        evidence_video_path       : Path to saved video (stored in DB, not in PDF).
        evidence_plate_image_path : Path to plate evidence image (shown in PDF).
        location                  : Free-text location.
        confidence                : Layer-5 violation confidence (0–1).

        Returns
        -------
        challan_id : str
        """
        now          = datetime.now()
        challan_id   = self._make_challan_id(plate_number)
        is_vehicle   = bool(plate_number and plate_number.strip())
        challan_type = "vehicle" if is_vehicle else "pedestrian"
        base_penalty = VEHICLE_PENALTY if is_vehicle else PEDESTRIAN_PENALTY
        due_date     = (now + timedelta(days=DUE_DAYS)).strftime("%Y-%m-%d")

        # Owner lookup
        if is_vehicle:
            owner_row = self._lookup_owner(plate_number)
            if owner_row:
                owner_name   = owner_row["owner_name"]
                phone_number = owner_row["phone_number"]
                email        = owner_row["email"]
                logger.info("Owner found: %s (plate=%s)", owner_name, plate_number)
            else:
                owner_name, phone_number, email = "Owner Not Found", None, None
                logger.warning("No owner record for plate '%s'", plate_number)
        else:
            # Pedestrian — no plate, no owner lookup
            owner_name, phone_number, email = "Unknown Person", None, None
            logger.info("Pedestrian violation — no plate recorded")

        with _get_connection() as conn:
            conn.execute("""
                INSERT INTO violations (
                    challan_id, plate_number, owner_name, phone_number,
                    email, challan_type, violation_timestamp, location,
                    penalty_amount, base_penalty_amount, due_date, status,
                    confidence, evidence_plate_image_path, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                challan_id, plate_number, owner_name, phone_number,
                email, challan_type,
                now.strftime("%Y-%m-%d %H:%M:%S"),
                location,
                base_penalty, base_penalty,   # amount = base initially
                due_date, "pending",
                confidence,
                evidence_plate_image_path,
                f"created by pipeline",
            ))
            conn.commit()

        logger.info(
            "Violation created | challan=%s | plate=%s | type=%s | "
            "penalty=Rs.%.0f | due=%s",
            challan_id, plate_number, challan_type, base_penalty, due_date,
        )
        return challan_id

    # ─────────────────────────────────────────────────────────────────────

    def generate_challan(self, challan_id: str) -> Optional[str]:
        """
        Build and save a PDF challan for the given challan_id.

        Always uses the LATEST penalty_amount from the DB (i.e., post-escalation)
        and generates a fresh QR code for the current amount.

        Returns the PDF path string, or None on failure.
        """
        violation = self.get_violation_by_challan(challan_id)
        if violation is None:
            logger.error("generate_challan: challan '%s' not found", challan_id)
            return None

        CHALLAN_DIR.mkdir(parents=True, exist_ok=True)
        safe_id  = challan_id.replace("/", "-").replace("\\", "-")
        pdf_path = CHALLAN_DIR / f"{safe_id}.pdf"

        try:
            pdf_bytes = _ChallanPDF(violation).build()
            pdf_path.write_bytes(pdf_bytes)
            logger.info("Challan PDF saved → '%s'  (Rs.%.2f)",
                        pdf_path, violation["penalty_amount"])
        except Exception as exc:
            logger.exception("PDF generation failed for %s: %s", challan_id, exc)
            return None

        # Update status to 'notified' and record the pdf path
        with _get_connection() as conn:
            conn.execute("""
                UPDATE violations
                SET pdf_challan_path = ?,
                    status = CASE WHEN status = 'pending' THEN 'notified'
                                  ELSE status END
                WHERE challan_id = ?
            """, (str(pdf_path), challan_id))
            conn.commit()

        return str(pdf_path)

    # ─────────────────────────────────────────────────────────────────────

    def check_and_escalate(self) -> int:
        """
        Scan all pending/notified violations, apply overdue escalation,
        and regenerate PDFs for escalated challans.

        Returns the count of newly escalated violations.
        """
        today     = date.today()
        escalated = 0

        with _get_connection() as conn:
            rows = conn.execute("""
                SELECT challan_id, due_date, base_penalty_amount,
                       penalty_amount, escalation_count
                FROM   violations
                WHERE  status IN ('pending', 'notified', 'escalated')
            """).fetchall()

            for row in rows:
                due   = datetime.strptime(row["due_date"], "%Y-%m-%d").date()
                if today <= due:
                    continue   # not overdue

                days_overdue   = (today - due).days
                cycles_due     = days_overdue // ESCALATION_DAYS
                cycles_applied = row["escalation_count"]

                if cycles_due <= cycles_applied:
                    continue   # already up-to-date

                # Recompute from base to avoid floating-point drift
                base     = float(row["base_penalty_amount"])
                new_amt  = round(base * ((1 + ESCALATION_RATE) ** cycles_due), 2)
                old_amt  = float(row["penalty_amount"])

                conn.execute("""
                    UPDATE violations
                    SET penalty_amount   = ?,
                        escalation_count = ?,
                        status           = 'escalated'
                    WHERE challan_id = ?
                """, (new_amt, cycles_due, row["challan_id"]))

                logger.warning(
                    "ESCALATED | %s | cycles=%d | "
                    "Rs.%.2f → Rs.%.2f",
                    row["challan_id"], cycles_due, old_amt, new_amt,
                )
                escalated += 1

            conn.commit()

        # Regenerate PDFs for all newly escalated challans so the QR/amount
        # on the PDF is always correct.
        if escalated:
            logger.info("Regenerating PDFs for %d escalated challan(s) …",
                        escalated)
            with _get_connection() as conn:
                esc_ids = conn.execute("""
                    SELECT challan_id FROM violations
                    WHERE status = 'escalated'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (escalated,)).fetchall()
            for row in esc_ids:
                self.generate_challan(row["challan_id"])

        logger.info("check_and_escalate: %d violation(s) escalated", escalated)
        return escalated

    # ─────────────────────────────────────────────────────────────────────

    def simulate_days_passed(self, challan_id: str, days: int) -> Optional[str]:
        """
        Demo / testing helper — simulate that `days` days have passed since
        the violation was recorded.

        Mechanically: shifts violation_timestamp and due_date backward by
        `days` days, then recalculates the penalty and regenerates the PDF.

        Does NOT permanently alter the real timestamps — it overwrites them
        in the DB for demo purposes. Use a separate test DB if you need
        to preserve the originals.

        Parameters
        ----------
        challan_id : str  — the BBMP challan ID
        days       : int  — number of days to simulate passing

        Returns
        -------
        str — path to the regenerated PDF, or None on failure.
        """
        logger.info(
            "simulate_days_passed: challan=%s days=%d", challan_id, days
        )

        violation = self.get_violation_by_challan(challan_id)
        if violation is None:
            logger.error("simulate_days_passed: challan '%s' not found",
                         challan_id)
            return None

        # Shift timestamps backward so the system thinks `days` days have passed
        orig_ts   = datetime.strptime(
            violation["violation_timestamp"], "%Y-%m-%d %H:%M:%S"
        )
        orig_due  = datetime.strptime(violation["due_date"], "%Y-%m-%d").date()

        new_ts    = orig_ts  - timedelta(days=days)
        new_due   = orig_due - timedelta(days=days)

        # Calculate what the penalty should be as of today given new dates
        base      = float(violation["base_penalty_amount"])
        today     = date.today()
        if today <= new_due:
            new_penalty  = base
            new_esc_cnt  = 0
        else:
            overdue      = (today - new_due).days
            cycles       = overdue // ESCALATION_DAYS
            new_penalty  = round(base * ((1 + ESCALATION_RATE) ** cycles), 2)
            new_esc_cnt  = cycles

        with _get_connection() as conn:
            conn.execute("""
                UPDATE violations
                SET violation_timestamp = ?,
                    due_date            = ?,
                    penalty_amount      = ?,
                    escalation_count    = ?,
                    status              = CASE
                        WHEN ? > 0 THEN 'escalated'
                        ELSE 'notified'
                    END
                WHERE challan_id = ?
            """, (
                new_ts.strftime("%Y-%m-%d %H:%M:%S"),
                new_due.strftime("%Y-%m-%d"),
                new_penalty,
                new_esc_cnt,
                new_esc_cnt,
                challan_id,
            ))
            conn.commit()

        logger.info(
            "simulate_days_passed | %s | new_due=%s | "
            "penalty=Rs.%.2f (esc=%d cycles)",
            challan_id, new_due, new_penalty, new_esc_cnt,
        )

        # Regenerate PDF with updated amount + fresh QR code
        pdf_path = self.generate_challan(challan_id)

        # Automatically trigger escalation notification via DeliveryAgent
        # if penalty was actually escalated (lazy import avoids circular deps)
        if new_esc_cnt > 0:
            try:
                from delivery_agent import DeliveryAgent
                _da = DeliveryAgent()
                _da.send_escalation_notification(challan_id)
            except Exception as _da_exc:
                logger.warning(
                    "simulate_days_passed: could not send escalation email: %s",
                    _da_exc,
                )

        return pdf_path

    # ─────────────────────────────────────────────────────────────────────

    def get_current_penalty(self, violation: sqlite3.Row) -> float:
        """
        Calculate the current penalty (with escalation) for a violation row.
        Delegates to the module-level pure function.
        """
        return get_current_penalty(violation)

    # ─────────────────────────────────────────────────────────────────────

    def mark_as_paid(
        self,
        challan_id:     str,
        transaction_id: Optional[str] = None,
    ) -> bool:
        """Mark a challan as paid. Returns True if updated, False if not found."""
        suffix = f" | txn={transaction_id}" if transaction_id else ""
        with _get_connection() as conn:
            cur = conn.execute("""
                UPDATE violations
                SET status = 'paid',
                    notes  = notes || ?
                WHERE challan_id = ? AND status != 'paid'
            """, (suffix, challan_id))
            conn.commit()

        if cur.rowcount == 0:
            logger.warning("mark_as_paid: '%s' not found or already paid",
                           challan_id)
            return False

        logger.info("PAID | %s%s", challan_id, suffix)
        return True

    # ── Query helpers ─────────────────────────────────────────────────────

    def get_violation_by_challan(
        self, challan_id: str
    ) -> Optional[sqlite3.Row]:
        with _get_connection() as conn:
            return conn.execute(
                "SELECT * FROM violations WHERE challan_id = ?", (challan_id,)
            ).fetchone()

    def list_pending(self) -> list:
        """Return all non-paid violations as a list of dicts."""
        with _get_connection() as conn:
            rows = conn.execute("""
                SELECT challan_id, plate_number, owner_name,
                       penalty_amount, base_penalty_amount,
                       due_date, status, violation_timestamp,
                       escalation_count
                FROM   violations
                WHERE  status IN ('pending', 'notified', 'escalated')
                ORDER  BY violation_timestamp DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        """Aggregate stats across all violations."""
        with _get_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                           AS total,
                    SUM(CASE WHEN status='paid'      THEN 1 ELSE 0 END) AS paid,
                    SUM(CASE WHEN status='escalated' THEN 1 ELSE 0 END) AS escalated,
                    SUM(CASE WHEN status IN ('pending','notified')
                             THEN 1 ELSE 0 END)                        AS pending,
                    COALESCE(SUM(penalty_amount),  0)                  AS total_amount,
                    COALESCE(SUM(CASE WHEN status='paid'
                                     THEN penalty_amount ELSE 0 END),0) AS collected
                FROM violations
            """).fetchone()
        return dict(row)


# ══════════════════════════════════════════════════════════════════════════════
#  Integration shim — convenience wrapper for run_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

def process_pipeline_violation(
    plate_number:     Optional[str],
    evidence_video:   Optional[str] = None,
    evidence_plate:   Optional[str] = None,
    location:         str           = "",
    confidence:       float         = 0.0,
    auto_pdf:         bool          = True,
) -> Optional[str]:
    """
    One-call convenience wrapper for run_pipeline.py.

    Creates the violation record and optionally generates the PDF challan.
    Returns challan_id on success, None on failure.

    Example usage in run_pipeline.py after Layer5 fires a VIOLATION:

        from penalty_manager import process_pipeline_violation
        challan_id = process_pipeline_violation(
            plate_number   = enhancer_result.plate_text,
            evidence_plate = enhancer_result.saved_paths[0],
            location       = "Outer Ring Road, Bengaluru",
            confidence     = event.confidence,
        )
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
        logger.exception("process_pipeline_violation failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CLI demo — python penalty_manager.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  VidTrace Penalty & Challan Management System — Demo v2")
    print("=" * 65)

    pm = PenaltyManager()

    # ── Test 1: Known plate (owner in DB) ─────────────────────────────────
    print("\n[TEST 1] Known vehicle — KA05KK5546")
    cid1 = pm.create_violation(
        plate_number              = "KA05KK5546",
        evidence_plate_image_path = "evidence/plate.jpg",
        location                  = "Outer Ring Road, Bengaluru",
        confidence                = 0.72,
    )
    pdf1 = pm.generate_challan(cid1)
    print(f"  challan_id : {cid1}")
    print(f"  pdf        : {pdf1}")

    # Show current penalty (no escalation yet)
    v1 = pm.get_violation_by_challan(cid1)
    print(f"  current penalty : Rs.{pm.get_current_penalty(v1):.2f}")

    # ── Test 2: Simulate 10 days passed (escalation demo) ─────────────────
    print("\n[TEST 2] Simulate 10 days passed for challan 1")
    pdf1_esc = pm.simulate_days_passed(cid1, 10)
    v1_esc   = pm.get_violation_by_challan(cid1)
    print(f"  escalated penalty : Rs.{v1_esc['penalty_amount']:.2f}")
    print(f"  escalation cycles : {v1_esc['escalation_count']}")
    print(f"  regenerated pdf   : {pdf1_esc}")

    # ── Test 3: Unknown plate ─────────────────────────────────────────────
    print("\n[TEST 3] Unknown vehicle — MH12AB1234")
    cid2 = pm.create_violation(
        plate_number = "MH12AB1234",
        location     = "Whitefield, Bengaluru",
        confidence   = 0.58,
    )
    pdf2 = pm.generate_challan(cid2)
    print(f"  challan_id : {cid2}")
    print(f"  pdf        : {pdf2}")

    # ── Test 4: Pedestrian (no plate) ─────────────────────────────────────
    print("\n[TEST 4] Pedestrian — no plate")
    cid3 = pm.create_violation(
        plate_number = None,
        location     = "MG Road, Bengaluru",
        confidence   = 0.63,
    )
    pdf3 = pm.generate_challan(cid3)
    print(f"  challan_id : {cid3}")
    print(f"  pdf        : {pdf3}")

    # ── Escalation check (real date-based) ────────────────────────────────
    print("\n[ESCALATION] Running check_and_escalate() …")
    n = pm.check_and_escalate()
    print(f"  newly escalated : {n}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[SUMMARY]")
    s = pm.summary()
    print(f"  Total violations : {s['total']}")
    print(f"  Pending          : {s['pending']}")
    print(f"  Escalated        : {s['escalated']}")
    print(f"  Paid             : {s['paid']}")
    print(f"  Total amount     : Rs. {s['total_amount']:.2f}")
    print(f"  Collected        : Rs. {s['collected']:.2f}")

    print("\n[DONE] Check the 'challans/' folder for generated PDFs.")
    print("=" * 65)