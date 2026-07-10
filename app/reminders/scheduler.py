"""Background scheduler that sends WhatsApp messages for due reminders."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from sqlmodel import Session, select

from app.config import settings
from app.db.engine import engine
from app.db.models import ReminderItem, ReminderItemStatus
from app.services.whatsapp import Dialog360Client

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 30
STUCK_SENDING_TIMEOUT = timedelta(minutes=5)
MAX_ATTEMPTS = 3

# NOTE: This scheduler assumes a single uvicorn worker. If multiple workers
# are used, both will query and send the same reminders. The eventual fix is
# SELECT ... FOR UPDATE SKIP LOCKED (Postgres only).


def _naive_utc_now() -> datetime:
    """Current UTC time as naive datetime (for DB comparisons with SQLite)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _recover_stuck_sending(session: Session) -> int:
    """Reset SENDING rows that have been stuck longer than the timeout back to PENDING."""
    cutoff = _naive_utc_now() - STUCK_SENDING_TIMEOUT
    stuck = session.exec(
        select(ReminderItem).where(
            ReminderItem.status == ReminderItemStatus.SENDING,
            ReminderItem.updated_at < cutoff,
        )
    ).all()

    for row in stuck:
        row.status = ReminderItemStatus.PENDING
        row.attempts += 1
        row.updated_at = _naive_utc_now()
        session.add(row)

    if stuck:
        session.commit()
        logger.warning("Recovered %d stuck SENDING reminder(s)", len(stuck))

    return len(stuck)


async def _send_due_reminders(wa: Dialog360Client) -> int:
    """Query due reminders, send them, and update status. Returns count sent."""
    now = _naive_utc_now()
    sent_count = 0

    with Session(engine) as session:
        # Recover stuck SENDING rows first
        await _recover_stuck_sending(session)

        # Query due PENDING reminders
        due = session.exec(
            select(ReminderItem).where(
                ReminderItem.status == ReminderItemStatus.PENDING,
                ReminderItem.due_at <= now,
            )
        ).all()

        if not due:
            return 0

        for row in due:
            try:
                # Mark SENDING before send (prevents double-fire on crash)
                row.status = ReminderItemStatus.SENDING
                row.updated_at = _naive_utc_now()
                session.add(row)
                session.commit()

                # Send the fire message (not the confirmation message)
                fire_message = f"⏰ Reminder: {row.what}"
                await wa.send_text(to=row.chat_id, body=fire_message)

                # Mark SENT
                row.status = ReminderItemStatus.SENT
                row.sent_at = _naive_utc_now()
                row.updated_at = _naive_utc_now()
                session.add(row)
                session.commit()
                sent_count += 1
                logger.info("Sent reminder %s to %s", row.reminder_id, row.chat_id)

            except Exception:
                logger.exception(
                    "Failed to send reminder %s to %s",
                    row.reminder_id,
                    row.chat_id,
                )
                # Increment attempts; mark FAILED if over cap, else back to PENDING
                row.attempts += 1
                row.updated_at = _naive_utc_now()
                if row.attempts >= MAX_ATTEMPTS:
                    row.status = ReminderItemStatus.FAILED
                    logger.error(
                        "Reminder %s marked FAILED after %d attempts",
                        row.reminder_id,
                        row.attempts,
                    )
                else:
                    row.status = ReminderItemStatus.PENDING
                session.add(row)
                session.commit()

    return sent_count


async def _reminder_loop() -> None:
    """Background loop that polls for due reminders and sends them."""
    wa = Dialog360Client(settings)
    logger.info("Reminder scheduler started (poll interval=%ds)", POLL_INTERVAL_SECONDS)

    while True:
        try:
            sent = await _send_due_reminders(wa)
            if sent:
                logger.info("Reminder scheduler: sent %d reminder(s)", sent)
        except Exception:
            logger.exception("Error in reminder scheduler cycle")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


_scheduler_task: asyncio.Task | None = None


def start_scheduler() -> None:
    """Start the reminder scheduler as a background asyncio task."""
    global _scheduler_task
    if _scheduler_task is not None:
        return
    _scheduler_task = asyncio.create_task(_reminder_loop())


async def stop_scheduler() -> None:
    """Cancel the reminder scheduler task."""
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None
