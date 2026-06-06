"""
DeliveryAgent — VidTrace Notification & Escalation Delivery System
===================================================================
Sends email notifications for new violations and escalation reminders.
Works with the SAME penalties.db used by PenaltyManager.

How to run
----------
Option A — Same process as run_pipeline.py (recommended):
    The DeliveryAgent is imported and started inside run_pipeline.py.
    It runs its scheduler in a background thread automatically.
    No extra terminal needed.

Option B — Standalone background process:
    python delivery_agent.py
    Leave this running in a separate terminal while the pipeline runs.
    It polls the DB every 60 seconds for new unsent notifications.

Email setup (Gmail)
--------------------
1. Go to your Google Account → Security → 2-Step Verification → App passwords
2. Create an App Password for "Mail"
3. Fill in SMTP_USER and SMTP_PASSWORD below with your Gmail + app password
4. Set SENDER_EMAIL and SENDER_NAME

Install:
    pip install apscheduler

Usage:
    from delivery_agent import DeliveryAgent
    agent = DeliveryAgent()
    agent.start()                                         # starts scheduler
    agent.send_violation_notification("BBMP-VH-...")      # immediate send
    agent.send_escalation_notification("BBMP-VH-...")     # escalation alert
    agent.mark_as_paid("BBMP-VH-...", txn_id="TXN123")   # mark paid
    agent.stop()                                          # shutdown scheduler
"""

from __future__ import annotations

import logging
import smtplib
import sqlite3
import time
from datetime import datetime, date, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

# ── APScheduler ───────────────────────────────────────────────────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

# ── PenaltyManager shared DB helpers ─────────────────────────────────────────
# We import only DB_PATH and _get_connection so we share the same DB.
# PenaltyManager is imported separately when needed to avoid circular issues.
from penalty_manager import (
    DB_PATH,
    _get_connection,
    PenaltyManager,
    AUTHORITY_NAME,
    AUTHORITY_EMAIL,
    AUTHORITY_PHONE,
    UPI_ID,
    PAYMENT_PORTAL,
)

# ══════════════════════════════════════════════════════════════════════════════
#  ── CONFIGURE THESE BEFORE RUNNING ──
# ══════════════════════════════════════════════════════════════════════════════

# Your Gmail address (the one sending emails)
SMTP_USER       = "rachfpatil@gmail.com"

# Gmail App Password (NOT your normal password)
# Get it from: Google Account → Security → 2-Step Verification → App passwords
SMTP_PASSWORD   = "qgwgydxwpqjlypos"

# Display name shown in the From field
SENDER_NAME     = "BBMP VidTrace Enforcement"
SENDER_EMAIL    = SMTP_USER        # usually same as SMTP_USER

# Gmail SMTP settings (don't change these for Gmail)
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587

# How often to check for unsent notifications (seconds) — used in standalone mode
POLL_INTERVAL   = 60

# Daily escalation check time — "HH:MM" 24-hour format
DAILY_CHECK_TIME = "09:00"

# ══════════════════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DeliveryAgent] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeliveryAgent")


# ══════════════════════════════════════════════════════════════════════════════
#  DB migration — adds notification_status column if it doesn't exist
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_notification_column() -> None:
    """
    Adds `notification_status` column to violations table if it's missing.
    Safe to call multiple times — uses ALTER TABLE ... IF NOT EXISTS pattern.

    Possible values:
        NULL      — not yet attempted
        'sent'    — email delivered successfully
        'failed'  — last attempt failed (will retry)
        'no_email'— no email address on record, cannot send
    """
    with _get_connection() as conn:
        # Check if column already exists
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(violations)"
        ).fetchall()]
        if "notification_status" not in cols:
            conn.execute(
                "ALTER TABLE violations ADD COLUMN notification_status TEXT"
            )
            conn.commit()
            logger.info("Added 'notification_status' column to violations table")


# ══════════════════════════════════════════════════════════════════════════════
#  Email builder helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_violation_email(v: dict, pdf_path: Optional[str]) -> MIMEMultipart:
    """
    Build the initial violation challan email.

    Contains:
    - Challan ID
    - Violation timestamp and location
    - Amount due and due date
    - Payment instructions
    - PDF challan attached
    """
    msg = MIMEMultipart("mixed")
    msg["Subject"] = (
        f"[BBMP] Illegal Dumping Challan Issued — "
        f"Challan ID: {v['challan_id']}"
    )
    msg["From"]    = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    msg["To"]      = v["email"]

    # ── HTML body ─────────────────────────────────────────────────────────
    plate_info = (
        f"Vehicle Number Plate: <b>{v['plate_number']}</b>"
        if v.get("plate_number")
        else "Offender Type: <b>Pedestrian (no plate detected)</b>"
    )
    owner_info = v.get("owner_name") or "Unknown"

    html = f"""
    <html><body style="font-family: Arial, sans-serif; color: #333;">

    <div style="background:#1B2A6B;padding:16px;color:white;border-radius:4px 4px 0 0;">
      <h2 style="margin:0;">&#9888; Illegal Dumping Violation Challan</h2>
      <p style="margin:4px 0 0 0;font-size:13px;">{AUTHORITY_NAME}</p>
    </div>

    <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 4px 4px;">
      <p>Dear <b>{owner_info}</b>,</p>

      <p>This is to inform you that a violation has been recorded against
      you for <b>illegal dumping</b> in a public area. A challan has been
      issued as follows:</p>

      <table style="border-collapse:collapse;width:100%;margin:16px 0;">
        <tr style="background:#f5f5f5;">
          <td style="padding:8px 12px;font-weight:bold;width:40%;
                     border:1px solid #ddd;">Challan ID</td>
          <td style="padding:8px 12px;border:1px solid #ddd;">
            <b>{v['challan_id']}</b></td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:bold;
                     border:1px solid #ddd;">Violation Date &amp; Time</td>
          <td style="padding:8px 12px;border:1px solid #ddd;">
            {v['violation_timestamp']}</td>
        </tr>
        <tr style="background:#f5f5f5;">
          <td style="padding:8px 12px;font-weight:bold;
                     border:1px solid #ddd;">Location</td>
          <td style="padding:8px 12px;border:1px solid #ddd;">
            {v.get('location') or 'Not recorded'}</td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:bold;
                     border:1px solid #ddd;">{plate_info.split(':')[0]}</td>
          <td style="padding:8px 12px;border:1px solid #ddd;">
            <b>{v.get('plate_number') or 'N/A (Pedestrian)'}</b></td>
        </tr>
        <tr style="background:#f5f5f5;">
          <td style="padding:8px 12px;font-weight:bold;
                     border:1px solid #ddd;">Penalty Amount</td>
          <td style="padding:8px 12px;border:1px solid #ddd;
                     color:#E8500A;font-size:16px;">
            <b>Rs. {v['penalty_amount']:.2f}</b></td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:bold;
                     border:1px solid #ddd;">Payment Due Date</td>
          <td style="padding:8px 12px;border:1px solid #ddd;
                     color:#CC0000;">
            <b>{v['due_date']}</b></td>
        </tr>
      </table>

      <div style="background:#FFF8E1;border-left:4px solid #E8500A;
                  padding:12px;margin:16px 0;">
        <b>How to Pay:</b><br/>
        Pay via UPI to: <b>{UPI_ID}</b><br/>
        Use Challan ID <b>{v['challan_id']}</b> as the payment reference.<br/>
        Online portal: <a href="{PAYMENT_PORTAL}">{PAYMENT_PORTAL}</a><br/>
        Or visit your nearest BBMP ward office.
      </div>

      <div style="background:#FFF3F3;border-left:4px solid #CC0000;
                  padding:12px;margin:16px 0;font-size:13px;">
        <b>Warning:</b> If not paid by <b>{v['due_date']}</b>, an additional
        10% penalty will be added every 2 days. Continued non-payment may
        result in legal action.
      </div>

      <p>The detailed challan PDF is attached to this email.</p>

      <p>For grievances, contact:<br/>
      Email: <a href="mailto:{AUTHORITY_EMAIL}">{AUTHORITY_EMAIL}</a><br/>
      Phone: {AUTHORITY_PHONE}</p>

      <p style="color:#888;font-size:11px;margin-top:24px;">
        This is an auto-generated notification from the VidTrace Municipal
        Enforcement System. Do not reply to this email.
      </p>
    </div>

    </body></html>
    """

    msg.attach(MIMEText(html, "html"))

    # ── Attach PDF ────────────────────────────────────────────────────────
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        filename = Path(pdf_path).name
        part.add_header(
            "Content-Disposition", f'attachment; filename="{filename}"'
        )
        msg.attach(part)
        logger.info("Attached PDF: %s", pdf_path)
    else:
        logger.warning("PDF not found for attachment: %s", pdf_path)

    return msg


def _build_escalation_email(v: dict, pdf_path: Optional[str]) -> MIMEMultipart:
    """
    Build the escalation reminder email sent when penalty increases.
    """
    msg = MIMEMultipart("mixed")
    msg["Subject"] = (
        f"[BBMP] URGENT — Penalty Escalated | "
        f"Challan {v['challan_id']} | Now Rs. {v['penalty_amount']:.2f}"
    )
    msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    msg["To"]   = v["email"]

    owner_info = v.get("owner_name") or "Offender"

    html = f"""
    <html><body style="font-family: Arial, sans-serif; color: #333;">

    <div style="background:#CC0000;padding:16px;color:white;
                border-radius:4px 4px 0 0;">
      <h2 style="margin:0;">&#9888; URGENT: Penalty Escalated</h2>
      <p style="margin:4px 0 0 0;font-size:13px;">{AUTHORITY_NAME}</p>
    </div>

    <div style="border:1px solid #ddd;border-top:none;padding:20px;
                border-radius:0 0 4px 4px;">
      <p>Dear <b>{owner_info}</b>,</p>

      <p style="color:#CC0000;font-size:15px;">
        <b>Your penalty has been escalated due to non-payment.</b>
      </p>

      <table style="border-collapse:collapse;width:100%;margin:16px 0;">
        <tr style="background:#f5f5f5;">
          <td style="padding:8px 12px;font-weight:bold;width:40%;
                     border:1px solid #ddd;">Challan ID</td>
          <td style="padding:8px 12px;border:1px solid #ddd;">
            <b>{v['challan_id']}</b></td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:bold;
                     border:1px solid #ddd;">Escalation Count</td>
          <td style="padding:8px 12px;border:1px solid #ddd;">
            {v['escalation_count']} cycle(s)</td>
        </tr>
        <tr style="background:#FFF3F3;">
          <td style="padding:8px 12px;font-weight:bold;
                     border:1px solid #ddd;">New Amount Due</td>
          <td style="padding:8px 12px;border:1px solid #ddd;
                     color:#CC0000;font-size:18px;">
            <b>Rs. {v['penalty_amount']:.2f}</b></td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:bold;
                     border:1px solid #ddd;">Due Date</td>
          <td style="padding:8px 12px;border:1px solid #ddd;">
            <b style="color:#CC0000;">{v['due_date']}</b></td>
        </tr>
      </table>

      <div style="background:#FFF8E1;border-left:4px solid #E8500A;
                  padding:12px;margin:16px 0;">
        <b>Pay Immediately:</b><br/>
        UPI ID: <b>{UPI_ID}</b><br/>
        Reference: <b>{v['challan_id']}</b><br/>
        Portal: <a href="{PAYMENT_PORTAL}">{PAYMENT_PORTAL}</a>
      </div>

      <p style="color:#CC0000;">
        <b>Every 2 days of non-payment adds another 10% to your penalty.
        Please pay now to stop further escalation.</b>
      </p>

      <p style="color:#888;font-size:11px;margin-top:24px;">
        Auto-generated by VidTrace Municipal Enforcement System.
      </p>
    </div>

    </body></html>
    """

    msg.attach(MIMEText(html, "html"))

    # Attach updated PDF with new amount
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{Path(pdf_path).name}"',
        )
        msg.attach(part)

    return msg


# ══════════════════════════════════════════════════════════════════════════════
#  SMTP sender — handles no-internet gracefully
# ══════════════════════════════════════════════════════════════════════════════

def _send_email(msg: MIMEMultipart, to_address: str) -> bool:
    """
    Send an email via Gmail SMTP. Returns True on success, False on failure.
    Gracefully handles no-internet / wrong credentials by logging the error.
    """
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_address, msg.as_string())
        logger.info("Email sent → %s | subject: %s", to_address, msg["Subject"])
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "SMTP auth failed — check SMTP_USER/SMTP_PASSWORD in delivery_agent.py"
        )
    except smtplib.SMTPException as exc:
        logger.error("SMTP error: %s", exc)
    except OSError as exc:
        # Covers no internet, DNS failure, timeout, etc.
        logger.warning(
            "Network error sending email (will retry later): %s", exc
        )
    except Exception as exc:
        logger.error("Unexpected email error: %s", exc)

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  DeliveryAgent
# ══════════════════════════════════════════════════════════════════════════════

class DeliveryAgent:
    """
    Handles all outbound notifications for the VidTrace penalty system.

    Responsibilities
    ----------------
    1. send_violation_notification(challan_id)
       → Called immediately after a new challan is created.
       → Sends the challan email with PDF attached.

    2. send_escalation_notification(challan_id)
       → Called when simulate_days_passed() or check_and_escalate() escalates
         a penalty.
       → Sends an urgent escalation reminder email.

    3. check_escalations_and_reminders()
       → Scheduled daily job (APScheduler).
       → Calls PenaltyManager.check_and_escalate(), then sends escalation
         emails for any newly escalated violations.
       → Also retries any 'failed' notifications from previous attempts.

    4. mark_as_paid(challan_id, transaction_id, proof_path)
       → Marks challan paid in DB, stops future escalations.

    5. start() / stop()
       → Starts/stops the APScheduler background thread.
    """

    def __init__(self):
        _ensure_notification_column()
        self._pm        = PenaltyManager()
        self._scheduler = None
        logger.info("DeliveryAgent initialised")

    # ── Scheduler control ─────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the APScheduler background scheduler.

        Schedules:
          - Daily escalation + reminder check at DAILY_CHECK_TIME
          - Retry of failed notifications every 30 minutes
        """
        if not APSCHEDULER_AVAILABLE:
            logger.warning(
                "APScheduler not installed — scheduler disabled. "
                "Run: pip install apscheduler"
            )
            return

        if self._scheduler and self._scheduler.running:
            logger.warning("Scheduler already running")
            return

        self._scheduler = BackgroundScheduler(
            job_defaults={"misfire_grace_time": 300}
        )

        # Daily escalation check at configured time
        h, m = DAILY_CHECK_TIME.split(":")
        self._scheduler.add_job(
            self.check_escalations_and_reminders,
            trigger="cron",
            hour=int(h),
            minute=int(m),
            id="daily_escalation_check",
        )

        # Retry failed notifications every 30 minutes
        self._scheduler.add_job(
            self._retry_failed_notifications,
            trigger="interval",
            minutes=30,
            id="retry_failed_notifications",
        )

        self._scheduler.start()
        logger.info(
            "Scheduler started — daily check at %s, retry every 30 min",
            DAILY_CHECK_TIME,
        )

    def stop(self) -> None:
        """Shut down the scheduler cleanly."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

    # ── Core notification methods ──────────────────────────────────────────

    def send_violation_notification(self, challan_id: str) -> bool:
        """
        Send the initial violation challan email with PDF attached.

        Called immediately after PenaltyManager.generate_challan() succeeds.

        Parameters
        ----------
        challan_id : str — e.g. "BBMP-VH-KA05KK5546-ABC12345"

        Returns
        -------
        bool — True if email sent, False if failed or no email on record.
        """
        v = self._get_violation(challan_id)
        if v is None:
            return False

        email = v.get("email")
        if not email:
            logger.info(
                "No email on record for %s — skipping notification", challan_id
            )
            self._set_notification_status(challan_id, "no_email")
            return False

        pdf_path = v.get("pdf_challan_path")
        msg      = _build_violation_email(v, pdf_path)
        success  = _send_email(msg, email)

        self._set_notification_status(
            challan_id, "sent" if success else "failed"
        )
        return success

    def send_escalation_notification(self, challan_id: str) -> bool:
        """
        Send an escalation reminder email when penalty has increased.

        Called automatically by simulate_days_passed() and
        check_escalations_and_reminders().

        Parameters
        ----------
        challan_id : str

        Returns
        -------
        bool — True if sent, False if failed or no email.
        """
        v = self._get_violation(challan_id)
        if v is None:
            return False

        email = v.get("email")
        if not email:
            logger.info(
                "No email for escalation notice — %s", challan_id
            )
            self._set_notification_status(challan_id, "no_email")
            return False

        # Regenerate PDF so it reflects the current (escalated) amount
        pdf_path = self._pm.generate_challan(challan_id)

        # Re-fetch row so pdf_challan_path is up to date
        v        = self._get_violation(challan_id)
        msg      = _build_escalation_email(v, pdf_path)
        success  = _send_email(msg, email)

        self._set_notification_status(
            challan_id, "sent" if success else "failed"
        )

        if success:
            logger.info(
                "Escalation email sent | %s | Rs.%.2f → %s",
                challan_id, v["penalty_amount"], email,
            )
        return success

    # ── Daily scheduled job ───────────────────────────────────────────────

    def check_escalations_and_reminders(self) -> None:
        """
        Daily job (run by APScheduler at DAILY_CHECK_TIME):

        1. Run PenaltyManager.check_and_escalate() to update DB penalties.
        2. For every newly escalated violation, send escalation email.
        3. Retry any previous 'failed' notification attempts.

        Also callable manually for testing:
            agent.check_escalations_and_reminders()
        """
        logger.info("Daily escalation check started …")

        try:
            # Step 1: apply escalation to overdue violations
            count = self._pm.check_and_escalate()
            logger.info("check_and_escalate: %d violation(s) escalated", count)

            # Step 2: send escalation emails for all escalated violations
            # that haven't had a successful notification yet today
            with _get_connection() as conn:
                rows = conn.execute("""
                    SELECT challan_id, email
                    FROM   violations
                    WHERE  status = 'escalated'
                    AND    email IS NOT NULL
                    AND    email != ''
                    AND    (notification_status IS NULL
                            OR notification_status = 'failed')
                """).fetchall()

            for row in rows:
                self.send_escalation_notification(row["challan_id"])
                time.sleep(1)   # brief pause to avoid SMTP rate-limits

        except Exception as exc:
            logger.error("check_escalations_and_reminders failed: %s", exc)

        logger.info("Daily escalation check complete")

    # ── Payment tracking ──────────────────────────────────────────────────

    def mark_as_paid(
        self,
        challan_id:     str,
        transaction_id: Optional[str] = None,
        proof_path:     Optional[str] = None,
    ) -> bool:
        """
        Mark a challan as paid. Stops all future escalations for this challan.

        Parameters
        ----------
        challan_id     : str  — BBMP challan ID
        transaction_id : str  — UPI reference / bank transaction ID (optional)
        proof_path     : str  — path to payment proof screenshot (optional, logged)

        Returns
        -------
        bool — True on success, False if not found or already paid.
        """
        success = self._pm.mark_as_paid(challan_id, transaction_id)

        if success:
            # Log proof path if provided
            if proof_path:
                with _get_connection() as conn:
                    conn.execute("""
                        UPDATE violations
                        SET notes = notes || ?
                        WHERE challan_id = ?
                    """, (f" | proof={proof_path}", challan_id))
                    conn.commit()

            logger.info(
                "PAID | challan=%s | txn=%s | proof=%s",
                challan_id,
                transaction_id or "—",
                proof_path or "—",
            )

            # Send a payment confirmation email
            self._send_payment_confirmation(challan_id)

        return success

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_violation(self, challan_id: str) -> Optional[dict]:
        """Fetch violation row as dict. Returns None if not found."""
        row = self._pm.get_violation_by_challan(challan_id)
        if row is None:
            logger.error("Violation not found: %s", challan_id)
            return None
        return dict(row)

    def _set_notification_status(
        self, challan_id: str, status: str
    ) -> None:
        """Write notification_status to the violations table."""
        with _get_connection() as conn:
            conn.execute("""
                UPDATE violations
                SET notification_status = ?
                WHERE challan_id = ?
            """, (status, challan_id))
            conn.commit()

    def _retry_failed_notifications(self) -> None:
        """
        Retry sending emails that previously failed (e.g. no internet).
        Runs every 30 minutes via APScheduler.
        """
        with _get_connection() as conn:
            rows = conn.execute("""
                SELECT challan_id, status
                FROM   violations
                WHERE  notification_status = 'failed'
                AND    email IS NOT NULL
                AND    email != ''
                AND    violations.status != 'paid'
            """).fetchall()

        if not rows:
            return

        logger.info("Retrying %d failed notification(s) …", len(rows))
        for row in rows:
            if row["status"] == "escalated":
                self.send_escalation_notification(row["challan_id"])
            else:
                self.send_violation_notification(row["challan_id"])
            time.sleep(1)

    def _send_payment_confirmation(self, challan_id: str) -> None:
        """Send a simple payment confirmation email (best-effort)."""
        v = self._get_violation(challan_id)
        if not v or not v.get("email"):
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = (
            f"[BBMP] Payment Received — Challan {challan_id}"
        )
        msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg["To"]   = v["email"]

        html = f"""
        <html><body style="font-family:Arial,sans-serif;color:#333;">
        <div style="background:#1A7A4A;padding:16px;color:white;
                    border-radius:4px 4px 0 0;">
          <h2 style="margin:0;">&#10004; Payment Confirmed</h2>
        </div>
        <div style="border:1px solid #ddd;border-top:none;padding:20px;">
          <p>Dear <b>{v.get('owner_name') or 'Offender'}</b>,</p>
          <p>Your payment for Challan <b>{challan_id}</b> of
             <b>Rs. {v['penalty_amount']:.2f}</b> has been recorded.
             No further action is required.</p>
          <p style="color:#888;font-size:11px;">
            Auto-generated by VidTrace Municipal Enforcement System.
          </p>
        </div>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))
        _send_email(msg, v["email"])   # best-effort, no status update needed

    def notify_new_challan(self, challan_id: str) -> None:
        """
        Convenience method called from run_pipeline.py after challan is issued.
        Sends violation notification immediately (non-blocking — logs errors).
        """
        try:
            self.send_violation_notification(challan_id)
        except Exception as exc:
            logger.error(
                "notify_new_challan failed for %s: %s", challan_id, exc
            )

    # ── Standalone polling loop (Option B) ───────────────────────────────

    def run_forever(self) -> None:
        """
        Run as a standalone process: polls DB every POLL_INTERVAL seconds
        for unsent notifications and sends them.

        Use this when running delivery_agent.py in a separate terminal.
        """
        self.start()
        logger.info(
            "DeliveryAgent running standalone — polling every %ds. "
            "Press Ctrl+C to stop.",
            POLL_INTERVAL,
        )
        try:
            while True:
                self._send_pending_notifications()
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("DeliveryAgent stopped by user")
        finally:
            self.stop()

    def _send_pending_notifications(self) -> None:
        """
        Find violations that have no notification sent yet and send them.
        Called in polling loop (standalone mode).
        """
        with _get_connection() as conn:
            rows = conn.execute("""
                SELECT challan_id, status
                FROM   violations
                WHERE  (notification_status IS NULL
                        OR notification_status = 'failed')
                AND    email IS NOT NULL
                AND    email != ''
                AND    status != 'paid'
            """).fetchall()

        for row in rows:
            if row["status"] == "escalated":
                self.send_escalation_notification(row["challan_id"])
            else:
                self.send_violation_notification(row["challan_id"])
            time.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone entry point — python delivery_agent.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  VidTrace DeliveryAgent — Standalone Mode")
    print("=" * 60)
    print(f"  SMTP user   : {SMTP_USER}")
    print(f"  Daily check : {DAILY_CHECK_TIME}")
    print(f"  Poll every  : {POLL_INTERVAL}s")
    print(f"  DB          : {DB_PATH}")
    print("=" * 60)

    if SMTP_USER == "your_gmail@gmail.com":
        print("\n[ERROR] You must configure SMTP_USER and SMTP_PASSWORD")
        print("        Edit delivery_agent.py — top CONFIG section.")
        exit(1)

    agent = DeliveryAgent()
    agent.run_forever()
