"""
FaceID/notifier.py
==================
Sends violation notice email with challan PDF attached.
Works with any SMTP server — configure below.
"""

import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── SMTP Configuration ────────────────────────────────────────────────────────
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587
SMTP_USER       = "poojithabalamurali@gmail.com"
SMTP_PASSWORD   = "lhfs ildb ienw ujdl"
SENDER_NAME     = "VidTrace BBMP Enforcement"
# ─────────────────────────────────────────────────────────────────────────────


def send_violation_email(
    recipient_email: str,
    recipient_name:  str,
    challan_id:      str,
    penalty_amount:  float,
    location:        str,
    pdf_path:        Optional[str] = None,
) -> bool:
    if not recipient_email:
        logger.warning("[Notifier] No email — skipping.")
        return False

    subject = f"BBMP Violation Notice — Challan {challan_id}"
    body = f"""
Dear {recipient_name},

This is an automated notice from the Bruhat Bengaluru Mahanagara Palike (BBMP)
Solid Waste Management and Environment Division.

You have been identified in connection with an illegal dumping violation
detected by the VidTrace Municipal Enforcement System.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Challan ID     : {challan_id}
  Penalty Amount : Rs. {penalty_amount:.2f}
  Location       : {location}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please find your challan attached.
Payment must be made within 7 days to avoid escalation.

Pay via UPI : {challan_id}
Portal      : https://bbmp.gov.in/payments
Grievances  : grievance.swm@bbmp.gov.in

Regards,
{SENDER_NAME}
BBMP — Bengaluru
    """.strip()

    msg = MIMEMultipart()
    msg["From"]    = f"{SENDER_NAME} <{SMTP_USER}>"
    msg["To"]      = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={Path(pdf_path).name}",
        )
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, recipient_email, msg.as_string())
        print(f"[Notifier] ✅ Email sent to {recipient_name} <{recipient_email}>")
        return True
    except Exception as exc:
        print(f"[Notifier] ❌ Email failed: {exc}")
        return False