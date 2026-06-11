"""
Reusable email sender via Gmail SMTP.

Borrows patterns from send_megansoft_emails.py:
  - Gmail App Password via env
  - HTML body support
  - DRY_RUN mode
  - Delay between sends
  - File attachment support
"""

from __future__ import annotations

import logging
import os
import smtplib
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Sequence

try:
    # When loaded as part of the weather_agent package (ADK from parent dir)
    from ..config import (
        DRY_RUN,
        GMAIL_APP_PASSWORD,
        GMAIL_SENDER,
        GMAIL_SENDER_NAME,
        SEND_DELAY_SECONDS,
        SMTP_HOST,
        SMTP_PORT,
    )
except ImportError:
    # When run standalone from inside the project directory
    from config import (  # type: ignore[import-untyped]
        DRY_RUN,
        GMAIL_APP_PASSWORD,
        GMAIL_SENDER,
        GMAIL_SENDER_NAME,
        SEND_DELAY_SECONDS,
        SMTP_HOST,
        SMTP_PORT,
    )

logger = logging.getLogger(__name__)


class EmailError(Exception):
    """Raised when an email cannot be sent."""


def _resolve_password() -> str:
    """Locate the Gmail App Password from env or fallback file."""
    pw = GMAIL_APP_PASSWORD
    if pw:
        return pw
    raise EmailError(
        "GMAIL_APP_PASSWORD not set in .env. "
        "Get it from https://myaccount.google.com/apppasswords"
    )


def _attach_file(msg: MIMEMultipart, path: Path) -> None:
    """Attach a file to a MIME message."""
    with open(path, "rb") as fh:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(fh.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{path.name}"')
    msg.attach(part)


def build_html_body(
    city: str,
    temperature: float | None,
    rain_prob: float,
    rain_amount: float,
    rain_likely: bool,
    extra_html: str = "",
) -> str:
    """Build an HTML weather-alert body (mirrors send_megansoft table style)."""
    alert_icon = "🌧️" if rain_likely else "☀️"
    return f"""\
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.45; color: #1f2937;">
    <h2>{alert_icon} Weather Alert for {city}</h2>
    <table border="1" cellpadding="8" cellspacing="0"
           style="border-collapse: collapse; width: 100%;">
        <thead style="background-color: #f3f4f6;">
            <tr>
                <th align="left">Metric</th>
                <th align="left">Value</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Temperature</td><td>{temperature or "N/A"} °F</td></tr>
            <tr><td>Rain Probability</td><td>{rain_prob}%</td></tr>
            <tr><td>Rain Amount</td><td>{rain_amount:.2f} in</td></tr>
            <tr><td>Rain Likely</td><td>{'Yes' if rain_likely else 'No'}</td></tr>
        </tbody>
    </table>
    {extra_html}
    <p>Sent by <strong>weather-agent</strong> · Google ADK</p>
</body>
</html>"""


def send_email(
    to_email: str,
    subject: str,
    html_body: str,
    attachments: Sequence[str | Path] | None = None,
    *,
    dry_run: bool | None = None,
) -> dict:
    """Send an HTML email with optional file attachments.

    Args:
        to_email: Recipient address.
        subject: Email subject line.
        html_body: Full HTML body string.
        attachments: Optional list of file paths to attach.
        dry_run: Override the global DRY_RUN setting.

    Returns:
        dict with keys ``status``, ``recipient``, ``subject``.
    """
    effective_dry_run = DRY_RUN if dry_run is None else dry_run
    subject = subject or "Weather Alert"

    logger.info(
        "[%s] To: %s | Subject: %s | Attachments: %s",
        "DRY-RUN" if effective_dry_run else "SEND",
        to_email,
        subject,
        len(attachments or []),
    )
    if effective_dry_run:
        return {
            "status": "dry_run",
            "recipient": to_email,
            "subject": subject,
        }

    password = _resolve_password()

    msg = MIMEMultipart()
    msg["From"] = f"{GMAIL_SENDER_NAME} <{GMAIL_SENDER}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    if attachments:
        for p in attachments:
            path = Path(p)
            if path.exists():
                _attach_file(msg, path)
            else:
                logger.warning("Attachment not found: %s", path)

    try:
        smtp = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        smtp.ehlo()
        smtp.starttls()
        smtp.login(GMAIL_SENDER, password)
        smtp.send_message(msg)
        smtp.quit()
    except Exception as exc:
        raise EmailError(f"SMTP send failed: {exc}") from exc

    if SEND_DELAY_SECONDS:
        time.sleep(SEND_DELAY_SECONDS)

    return {
        "status": "sent",
        "recipient": to_email,
        "subject": subject,
    }


def send_weather_alert(
    to_email: str,
    city: str,
    temperature: float | None,
    rain_prob: float,
    rain_amount: float,
    rain_likely: bool,
    subject: str | None = None,
    *,
    dry_run: bool | None = None,
) -> dict:
    """Convenience: build a weather-alert email from weather data and send it.

    Returns the same dict as :func:`send_email`.
    """
    if not subject:
        icon = "Rain Alert" if rain_likely else "Weather Update"
        subject = f"[weather-agent] {icon} — {city}"
    body = build_html_body(city, temperature, rain_prob, rain_amount, rain_likely)
    return send_email(to_email, subject, body, dry_run=dry_run)
