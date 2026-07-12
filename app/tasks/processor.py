from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.config import Settings
from app.db.engine import engine
from app.db.models import (
    ItemRecord,
    ItemReviewStatus,
    Task,
    TaskStatus,
)
from app.services.whatsapp import Dialog360Client
from app.tasks.session import DueDateResolution, ReviewSession, ReviewSessionStore
from app.tasks.task_agent import TaskReviewAgent

logger = logging.getLogger(__name__)

_CLARIFICATION_LIMIT_MSG = "לא הצלחתי להפוך את הפריט הזה למשימה ברורה, אז השארתי אותו לפעם הבאה."
_NO_ITEMS_MSG = "אין פריטים לעיבוד 🎉"
_ALREADY_IN_SESSION_MSG = "אנחנו כבר בסשן עיון. תוכל לענות, או לסיים ב-\"{end_keyword}\"."
_END_SUMMARY = "עצרתי את העיון. יצרתי {created} משימות. נשארו {remaining} פריטים בסשן הזה לפעם הבאה."
_COMPLETE_SUMMARY = "סיימנו! יצרתי {created} משימות. נשארו {remaining} פריטים לפעם הבאה."
_INITIAL_Q_TEMPLATE = 'לגבי "{subject}" — מה המשימה שתרצה ליצור, ומתי לבצע אותה?'
_FALLBACK_CLARIFICATION = "לא הבנתי. תוכל לנסח מחדש את המשימה?"


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def _is_true(value: Any) -> bool:
    return _safe_str(value).lower() == "true"


def _parse_due_at(raw: str) -> datetime | None:
    raw = _safe_str(raw)
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def _is_valid_resolution(value: str) -> bool:
    return value in (
        DueDateResolution.PROVIDED.value,
        DueDateResolution.INTENTIONALLY_ABSENT.value,
        DueDateResolution.UNKNOWN.value,
    )


class TaskReviewService:
    def __init__(
        self,
        agent: TaskReviewAgent,
        session_store: ReviewSessionStore,
        settings: Settings,
    ) -> None:
        self.agent = agent
        self.sessions = session_store
        self.settings = settings

    async def start_review(
        self, tenant_id: str, chat_id: str, wa: Dialog360Client
    ) -> None:
        existing = self.sessions.get(tenant_id, chat_id)
        if existing:
            msg = _ALREADY_IN_SESSION_MSG.format(end_keyword=self.settings.review_end_keyword)
            await wa.send_text(to=chat_id, body=msg)
            return

        item_ids = self._load_pending_item_ids(tenant_id, chat_id)
        if not item_ids:
            await wa.send_text(to=chat_id, body=_NO_ITEMS_MSG)
            return

        session = self.sessions.start(tenant_id, chat_id, item_ids)
        await self._present_current_item(session, wa)

    async def handle_reply(
        self,
        tenant_id: str,
        chat_id: str,
        user_msg: str,
        wa: Dialog360Client,
        current_time: datetime,
    ) -> None:
        session = self.sessions.get(tenant_id, chat_id)
        if session is None:
            return

        session.touch()

        trimmed = user_msg.strip()
        if trimmed == self.settings.review_end_keyword:
            await self._end_review(tenant_id, chat_id, wa)
            return

        item = self._reload_and_validate_item(session)
        if item is None:
            await self._advance_or_end(session, wa)
            return

        try:
            pred = await self.agent.aforward(
                raw_subject=item.subject,
                current_candidate_subject=session.candidate_subject,
                current_candidate_due_at=session.candidate_due_at,
                current_due_date_resolution=session.due_date_resolution.value,
                user_reply=user_msg,
                current_time=current_time.isoformat(),
            )
        except Exception:
            logger.exception("LLM call failed during review for item %s", session.current_item_id)
            session.clarification_count += 1
            await self._handle_clarification_limit(session, wa)
            return

        updated_subject = _safe_str(pred.updated_subject)
        updated_due_at_raw = _safe_str(pred.updated_due_at)
        resolution_raw = _safe_str(pred.due_date_resolution)
        needs_clarification = _is_true(pred.needs_clarification)
        clarification_question = _safe_str(pred.clarification_question)

        if not _is_valid_resolution(resolution_raw):
            logger.warning("Invalid due_date_resolution from LLM: %r", resolution_raw)
            resolution_raw = DueDateResolution.UNKNOWN.value

        parsed_due_at = _parse_due_at(updated_due_at_raw)
        if updated_due_at_raw and parsed_due_at is None:
            logger.warning("Invalid due_at from LLM: %r", updated_due_at_raw)
            needs_clarification = True
            if not clarification_question:
                clarification_question = _FALLBACK_CLARIFICATION

        session.candidate_subject = updated_subject
        session.candidate_due_at = (
            parsed_due_at.isoformat() if parsed_due_at else ""
        )
        session.due_date_resolution = DueDateResolution(resolution_raw)

        if needs_clarification:
            session.clarification_count += 1
            if not clarification_question:
                clarification_question = _FALLBACK_CLARIFICATION
            await self._handle_clarification(session, clarification_question, wa)
            return

        if not updated_subject:
            session.clarification_count += 1
            await self._handle_clarification(session, _FALLBACK_CLARIFICATION, wa)
            return

        due_date_ok = session.due_date_resolution in (
            DueDateResolution.PROVIDED,
            DueDateResolution.INTENTIONALLY_ABSENT,
        )
        if not due_date_ok:
            session.clarification_count += 1
            await self._handle_clarification(
                session, "מתי תרצה לבצע את זה?", wa
            )
            return

        due_at_utc = parsed_due_at
        created = self._create_task_transactional(
            tenant_id=tenant_id,
            chat_id=chat_id,
            subject=updated_subject,
            due_at=due_at_utc,
            source_item_id=item.id,
        )
        if created:
            session.created_count += 1

        await self._advance_or_end(session, wa)

    async def end_review(
        self, tenant_id: str, chat_id: str, wa: Dialog360Client
    ) -> None:
        await self._end_review(tenant_id, chat_id, wa)

    async def _end_review(
        self, tenant_id: str, chat_id: str, wa: Dialog360Client
    ) -> None:
        session = self.sessions.end(tenant_id, chat_id)
        if session is None:
            return
        msg = _END_SUMMARY.format(
            created=session.created_count,
            remaining=session.remaining_count,
        )
        await wa.send_text(to=chat_id, body=msg)

    async def _present_current_item(
        self, session: ReviewSession, wa: Dialog360Client
    ) -> None:
        item = self._reload_and_validate_item(session)
        if item is None:
            await self._advance_or_end(session, wa)
            return
        msg = _INITIAL_Q_TEMPLATE.format(subject=item.subject)
        await wa.send_text(to=session.chat_id, body=msg)

    async def _advance_or_end(
        self, session: ReviewSession, wa: Dialog360Client
    ) -> None:
        session.advance()
        if session.current_item_id is None:
            msg = _COMPLETE_SUMMARY.format(
                created=session.created_count,
                remaining=session.remaining_count,
            )
            self.sessions.end(session.tenant_id, session.chat_id)
            await wa.send_text(to=session.chat_id, body=msg)
        else:
            await self._present_current_item(session, wa)

    async def _handle_clarification(
        self, session: ReviewSession, question: str, wa: Dialog360Client
    ) -> None:
        await self._handle_clarification_limit(session, wa, question=question)

    async def _handle_clarification_limit(
        self,
        session: ReviewSession,
        wa: Dialog360Client,
        question: str | None = None,
    ) -> None:
        if question is None:
            question = _CLARIFICATION_LIMIT_MSG

        if session.clarification_count < self.settings.review_clarification_limit:
            await wa.send_text(to=session.chat_id, body=question)
            return

        await wa.send_text(to=session.chat_id, body=_CLARIFICATION_LIMIT_MSG)
        await self._advance_or_end(session, wa)

    def _load_pending_item_ids(
        self, tenant_id: str, chat_id: str
    ) -> list[str]:
        with Session(engine) as session:
            undated = session.exec(
                select(ItemRecord)
                .where(
                    ItemRecord.tenant_id == tenant_id,
                    ItemRecord.chat_id == chat_id,
                    ItemRecord.review_status == ItemReviewStatus.PENDING,
                    ItemRecord.due_at.is_(None),
                )
                .order_by(ItemRecord.created_at, ItemRecord.id)
            ).all()
            dated = session.exec(
                select(ItemRecord)
                .where(
                    ItemRecord.tenant_id == tenant_id,
                    ItemRecord.chat_id == chat_id,
                    ItemRecord.review_status == ItemReviewStatus.PENDING,
                    ItemRecord.due_at.is_not(None),
                )
                .order_by(ItemRecord.due_at, ItemRecord.created_at, ItemRecord.id)
            ).all()
            return [r.id for r in undated] + [r.id for r in dated]

    def _reload_and_validate_item(
        self, session: ReviewSession
    ) -> ItemRecord | None:
        item_id = session.current_item_id
        if item_id is None:
            return None
        with Session(engine) as db:
            item = db.get(ItemRecord, item_id)
            if item is None:
                return None
            if item.tenant_id != session.tenant_id:
                return None
            if item.chat_id != session.chat_id:
                return None
            if item.review_status != ItemReviewStatus.PENDING:
                return None
            return item

    def _create_task_transactional(
        self,
        tenant_id: str,
        chat_id: str,
        subject: str,
        due_at: datetime | None,
        source_item_id: str,
    ) -> bool:
        with Session(engine) as db:
            try:
                item = db.get(ItemRecord, source_item_id)
                if item is None or item.review_status != ItemReviewStatus.PENDING:
                    return False

                task = Task(
                    tenant_id=tenant_id,
                    chat_id=chat_id,
                    subject=subject,
                    due_at=due_at,
                    source_item_id=source_item_id,
                    status=TaskStatus.ACTIVE,
                )
                db.add(task)

                item.review_status = ItemReviewStatus.CONVERTED
                item.converted_at = datetime.now(timezone.utc)
                db.add(item)

                db.commit()
                return True

            except IntegrityError:
                db.rollback()
                existing_item = db.get(ItemRecord, source_item_id)
                if existing_item and existing_item.review_status != ItemReviewStatus.CONVERTED:
                    existing_item.review_status = ItemReviewStatus.CONVERTED
                    existing_item.converted_at = datetime.now(timezone.utc)
                    db.add(existing_item)
                    db.commit()
                return False
