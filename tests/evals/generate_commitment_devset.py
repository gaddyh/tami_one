"""Deterministic generator for commitment extraction evaluation examples.

Produces 72 controlled-probe examples across 6 categories, split into
trainset.json (48), devset.json (12), and testset.json (12).

Each example tests exactly one behavior whenever possible. Contrastive
pairs (one tiny change flips the expected output) are used to catch
real regressions.

Run: python tests/evals/generate_commitment_devset.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CHAT_ID = "972546610653@c.us"
CHAT_NAME = "Gaddy"


# ─── Helpers ───────────────────────────────────────────────────────────


def commitment(
    *,
    id: str | None = None,
    committed_party: str | None,
    required_action: str,
    deadline: str | None = None,
    context: str,
    status: str = "waiting",
    notification: str = "none",
) -> dict[str, Any]:
    return {
        "id": id,
        "chat_id": CHAT_ID,
        "chat_name": CHAT_NAME,
        "committed_party": committed_party,
        "required_action": required_action,
        "deadline": deadline,
        "context": context,
        "status": status,
        "notification": notification,
    }


def example(
    *,
    messages: str,
    expected_commitments: list[dict[str, Any]],
    existing_commitments: list[dict[str, Any]] | None = None,
    chat_id: str = CHAT_ID,
    chat_name: str = CHAT_NAME,
    category: str,
    scenario: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "scenario": scenario,
        "chat_id": chat_id,
        "chat_name": chat_name,
        "existing_commitments_json": json.dumps(
            existing_commitments or [],
            ensure_ascii=False,
        ),
        "messages": messages,
        "expected_commitments": expected_commitments,
    }


def _existing(
    *,
    id: str,
    committed_party: str | None,
    required_action: str,
    deadline: str | None = None,
    context: str,
    status: str = "waiting",
) -> dict[str, Any]:
    return commitment(
        id=id,
        committed_party=committed_party,
        required_action=required_action,
        deadline=deadline,
        context=context,
        status=status,
    )


# ─── A. Act vs Ignore (12) ─────────────────────────────────────────────


def act_vs_ignore_examples() -> list[dict[str, Any]]:
    return [
        example(
            category="act_vs_ignore",
            scenario="social_chat_ignore",
            messages=(
                "Gaddy: hey how are you?\n"
                "Gaddy: long time no see\n"
                "Gaddy: let's catch up soon"
            ),
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="soft_social_future_ignore",
            messages="Gaddy: let's grab coffee sometime",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="explicit_self_commitment",
            messages=(
                "Gaddy: I will send the documents by tomorrow\n"
                "Gaddy: just need to finish the last page"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I will send the documents by tomorrow",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="request_plus_acceptance",
            messages=(
                "Alice: can you send me the report by Friday?\n"
                "Bob: sure, I'll have it ready by Friday"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Send the report",
                    deadline="Friday",
                    context="Alice asked Bob to send the report by Friday, Bob agreed",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="waiting_external_party",
            messages=(
                "Gaddy: I'm waiting for the bank to process the loan\n"
                "Gaddy: nothing I can do until they respond"
            ),
            expected_commitments=[
                commitment(
                    committed_party="the bank",
                    required_action="Process the loan",
                    deadline=None,
                    context="I'm waiting for the bank to process the loan",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="vague_action_unclear",
            messages="Gaddy: we should probably do something about the project at some point",
            expected_commitments=[
                commitment(
                    committed_party=None,
                    required_action="Do something about the project",
                    deadline=None,
                    context="we should probably do something about the project at some point",
                    status="unclear",
                )
            ],
        ),
        # Contrastive pair: generic callback without topic → ignore
        example(
            category="act_vs_ignore",
            scenario="generic_callback_ignore",
            messages="Gaddy: let me think about it and get back to you",
            expected_commitments=[],
        ),
        # Contrastive pair: generic callback WITH topic + deadline → commitment
        example(
            category="act_vs_ignore",
            scenario="generic_callback_with_topic_positive",
            messages=(
                "Dana: can you approve the quote?\n"
                "Gaddy: let me think about the quote and get back to you tomorrow"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Get back to Dana about the quote",
                    deadline="tomorrow",
                    context="let me think about the quote and get back to you tomorrow",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="thanks_only_ignore",
            messages=(
                "Gaddy: good morning!\n"
                "Gaddy: thanks for the help yesterday\n"
                "Gaddy: have a great day"
            ),
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="motivational_ignore",
            messages="Gaddy: you've got this, keep up the great work!",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="question_only_ignore",
            messages="Gaddy: what time is the meeting tomorrow?",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="explicit_promise_different_action",
            messages="Gaddy: I'll call the supplier today",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline="today",
                    context="I'll call the supplier today",
                )
            ],
        ),
    ]


# ─── B. Party Extraction (12) ──────────────────────────────────────────


def party_flip_examples() -> list[dict[str, Any]]:
    return [
        # Contrastive pair: Alice asks Bob → Bob commits
        example(
            category="args_party",
            scenario="alice_asks_bob_bob_accepts",
            messages=(
                "Alice: Bob, can you send me the report by Friday?\n"
                "Bob: sure, I'll have it ready by Friday"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Send the report",
                    deadline="Friday",
                    context="Alice asked Bob to send the report by Friday, Bob agreed",
                )
            ],
        ),
        # Contrastive pair: Bob asks Alice → Alice commits
        example(
            category="args_party",
            scenario="bob_asks_alice_alice_accepts",
            messages=(
                "Bob: Alice, can you send me the report by Friday?\n"
                "Alice: sure, I'll have it ready by Friday"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Alice",
                    required_action="Send the report",
                    deadline="Friday",
                    context="Bob asked Alice to send the report by Friday, Alice agreed",
                )
            ],
        ),
        example(
            category="args_party",
            scenario="external_party_waiting",
            messages=(
                "Gaddy: I'm waiting for the bank to process the loan\n"
                "Gaddy: nothing I can do until they respond"
            ),
            expected_commitments=[
                commitment(
                    committed_party="the bank",
                    required_action="Process the loan",
                    deadline=None,
                    context="I'm waiting for the bank to process the loan",
                )
            ],
        ),
        example(
            category="args_party",
            scenario="unknown_party_unclear",
            messages="Gaddy: someone needs to call the supplier about the order",
            expected_commitments=[
                commitment(
                    committed_party=None,
                    required_action="Call the supplier",
                    deadline=None,
                    context="someone needs to call the supplier about the order",
                    status="unclear",
                )
            ],
        ),
        example(
            category="args_party",
            scenario="self_commitment_speaker",
            messages="Gaddy: I will send the documents by tomorrow",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I will send the documents by tomorrow",
                )
            ],
        ),
        example(
            category="args_party",
            scenario="third_party_named",
            messages="Gaddy: Bob will send the documents by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Send the documents",
                    deadline="Friday",
                    context="Bob will send the documents by Friday",
                )
            ],
        ),
        example(
            category="args_party",
            scenario="group_we_need_unclear",
            messages="Gaddy: we need to send the documents to the client",
            expected_commitments=[
                commitment(
                    committed_party=None,
                    required_action="Send the documents to the client",
                    deadline=None,
                    context="we need to send the documents to the client",
                    status="unclear",
                )
            ],
        ),
        example(
            category="args_party",
            scenario="alice_asks_bob_bob_refuses",
            messages=(
                "Alice: Bob, can you send me the report by Friday?\n"
                "Bob: sorry, I won't be able to do it this week"
            ),
            expected_commitments=[],
        ),
        example(
            category="args_party",
            scenario="multi_party_only_acceptor_committed",
            messages=(
                "Alice: we need someone to prepare the report by Friday\n"
                "Bob: I can't, I'm swamped\n"
                "Charlie: I'll do it, I'll have it ready by Friday"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Charlie",
                    required_action="Prepare the report",
                    deadline="Friday",
                    context="Alice asked for someone to prepare the report, Charlie volunteered",
                )
            ],
        ),
        example(
            category="args_party",
            scenario="request_to_group_one_volunteers",
            messages=(
                "Gaddy: someone needs to book the meeting room for Tuesday\n"
                "Dana: I'll book it"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Dana",
                    required_action="Book the meeting room",
                    deadline="Tuesday",
                    context="Gaddy asked someone to book the meeting room, Dana volunteered",
                )
            ],
        ),
        # Party handoff: existing commitment for Alice, new message says Bob will handle it
        example(
            category="args_party",
            scenario="party_handoff_update",
            existing_commitments=[
                _existing(
                    id="abc-100",
                    committed_party="Alice",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="Alice will send the documents by tomorrow",
                )
            ],
            messages="Gaddy: actually Bob will handle it instead of Alice",
            expected_commitments=[
                commitment(
                    id="abc-100",
                    committed_party="Bob",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="actually Bob will handle it instead of Alice",
                )
            ],
        ),
        example(
            category="args_party",
            scenario="speaker_commits_for_other_named",
            messages="Gaddy: Alice will send the documents, I'll make sure of it",
            expected_commitments=[
                commitment(
                    committed_party="Alice",
                    required_action="Send the documents",
                    deadline=None,
                    context="Alice will send the documents, I'll make sure of it",
                )
            ],
        ),
    ]


# ─── C. Deadline Extraction (12) ───────────────────────────────────────


def deadline_examples() -> list[dict[str, Any]]:
    cases: list[tuple[str, str, str | None]] = [
        ("by Friday", "Friday"),
        ("by tomorrow", "tomorrow"),
        ("next Monday", "next Monday"),
        ("by 17:00", "by 17:00"),
        ("by July 15", "July 15"),
        ("soon", None),
        ("when I can", None),
        ("by end of week", "end of week"),
        ("next week", "next week"),
        ("today", "today"),
        ("by 5pm", "by 5pm"),
    ]

    examples: list[dict[str, Any]] = []

    for phrase, expected_deadline in cases:
        scenario_name = f"deadline_{phrase.replace(' ', '_').replace(':', '')}"
        examples.append(
            example(
                category="args_deadline",
                scenario=scenario_name,
                messages=f"Gaddy: I will send the documents {phrase}",
                expected_commitments=[
                    commitment(
                        committed_party="Gaddy",
                        required_action="Send the documents",
                        deadline=expected_deadline,
                        context=f"I will send the documents {phrase}",
                    )
                ],
            )
        )

    # No deadline at all
    examples.append(
        example(
            category="args_deadline",
            scenario="deadline_none",
            messages="Gaddy: I will send the documents",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline=None,
                    context="I will send the documents",
                )
            ],
        )
    )

    return examples


# ─── D. Required Action Extraction (12) ────────────────────────────────


def required_action_examples() -> list[dict[str, Any]]:
    return [
        example(
            category="args_required_action",
            scenario="send_docs_forward",
            messages="Gaddy: I'll forward the documents tomorrow",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I'll forward the documents tomorrow",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="send_docs_send_over",
            messages="Gaddy: I'll send over the docs tomorrow",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I'll send over the docs tomorrow",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="send_docs_paperwork",
            messages="Gaddy: I'll get the paperwork to you tomorrow",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I'll get the paperwork to you tomorrow",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="call_supplier_ring",
            messages="Gaddy: I need to ring the supplier about the order",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline=None,
                    context="I need to ring the supplier about the order",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="call_supplier_phone",
            messages="Gaddy: I'll give the supplier a call about the order today",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline="today",
                    context="I'll give the supplier a call about the order today",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="pay_invoice_take_care",
            messages="Dana: I'll take care of the invoice payment today",
            expected_commitments=[
                commitment(
                    committed_party="Dana",
                    required_action="Pay the invoice",
                    deadline="today",
                    context="I'll take care of the invoice payment today",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="pay_invoice_settle",
            messages="Dana: I'll settle the invoice by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Dana",
                    required_action="Pay the invoice",
                    deadline="Friday",
                    context="I'll settle the invoice by Friday",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="prepare_report",
            messages="Gaddy: I'll have the report ready by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Prepare the report",
                    deadline="Friday",
                    context="I'll have the report ready by Friday",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="prepare_report_draft",
            messages="Gaddy: I'll draft the report by end of week",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Prepare the report",
                    deadline="end of week",
                    context="I'll draft the report by end of week",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="book_meeting_room_reserve",
            messages="Gaddy: I'll reserve the meeting room for Tuesday",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Book the meeting room",
                    deadline="Tuesday",
                    context="I'll reserve the meeting room for Tuesday",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="upload_signed_form",
            messages="Gaddy: I'll upload the signed form by Wednesday",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Upload the signed form",
                    deadline="Wednesday",
                    context="I'll upload the signed form by Wednesday",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="process_loan",
            messages="Gaddy: the bank needs to process the loan application",
            expected_commitments=[
                commitment(
                    committed_party="the bank",
                    required_action="Process the loan",
                    deadline=None,
                    context="the bank needs to process the loan application",
                )
            ],
        ),
    ]


# ─── E. New vs Update (12) ─────────────────────────────────────────────


def update_vs_new_examples() -> list[dict[str, Any]]:
    existing_send_docs = _existing(
        id="abc-123",
        committed_party="Gaddy",
        required_action="Send the documents",
        deadline="tomorrow",
        context="I will send the documents by tomorrow",
    )

    existing_call_supplier = _existing(
        id="abc-456",
        committed_party="Gaddy",
        required_action="Call the supplier",
        deadline=None,
        context="Need to call the supplier about the order",
    )

    existing_alice_docs = _existing(
        id="abc-200",
        committed_party="Alice",
        required_action="Send the documents",
        deadline="Friday",
        context="Alice will send the documents by Friday",
    )

    existing_prepare_report = _existing(
        id="abc-300",
        committed_party="Bob",
        required_action="Prepare the report",
        deadline="Friday",
        context="Bob will prepare the report by Friday",
    )

    return [
        # Update existing deadline
        example(
            category="lifecycle_update_vs_new",
            scenario="update_existing_deadline",
            existing_commitments=[existing_send_docs],
            messages="Gaddy: I won't manage tomorrow, I'll send the documents next Monday",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="next Monday",
                    context="I won't manage tomorrow, I'll send the documents next Monday",
                )
            ],
        ),
        # New unrelated commitment (different action)
        example(
            category="lifecycle_update_vs_new",
            scenario="new_unrelated_commitment",
            existing_commitments=[existing_send_docs],
            messages="Gaddy: I also need to call the supplier today",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline="today",
                    context="I also need to call the supplier today",
                )
            ],
        ),
        # Same existing, no duplicate — update deadline only
        example(
            category="lifecycle_update_vs_new",
            scenario="same_existing_no_duplicate",
            existing_commitments=[existing_call_supplier],
            messages="Gaddy: I'll call the supplier later today",
            expected_commitments=[
                commitment(
                    id="abc-456",
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline="today",
                    context="I'll call the supplier later today",
                )
            ],
        ),
        # Party handoff: Alice → Bob
        example(
            category="lifecycle_update_vs_new",
            scenario="party_handoff_update",
            existing_commitments=[existing_alice_docs],
            messages="Gaddy: actually Bob will handle it instead of Alice",
            expected_commitments=[
                commitment(
                    id="abc-200",
                    committed_party="Bob",
                    required_action="Send the documents",
                    deadline="Friday",
                    context="actually Bob will handle it instead of Alice",
                )
            ],
        ),
        # New when existing exists but different action
        example(
            category="lifecycle_update_vs_new",
            scenario="new_when_existing_different_action",
            existing_commitments=[existing_send_docs],
            messages="Gaddy: I need to prepare the report by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Prepare the report",
                    deadline="Friday",
                    context="I need to prepare the report by Friday",
                )
            ],
        ),
        # Update context only (no deadline change)
        example(
            category="lifecycle_update_vs_new",
            scenario="update_existing_context_only",
            existing_commitments=[existing_send_docs],
            messages="Gaddy: I'll send the documents as soon as I get the signature",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I'll send the documents as soon as I get the signature",
                )
            ],
        ),
        # Existing unchanged — reaffirming, still waiting
        example(
            category="lifecycle_update_vs_new",
            scenario="existing_unchanged_no_update",
            existing_commitments=[existing_send_docs],
            messages="Gaddy: still planning to send the documents tomorrow",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="still planning to send the documents tomorrow",
                )
            ],
        ),
        # Multiple existing, update only one
        example(
            category="lifecycle_update_vs_new",
            scenario="multiple_existing_update_one",
            existing_commitments=[existing_send_docs, existing_call_supplier],
            messages="Gaddy: I'll send the documents by Friday instead",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="Friday",
                    context="I'll send the documents by Friday instead",
                )
            ],
        ),
        # New commitment with same action as existing but different party → new
        example(
            category="lifecycle_update_vs_new",
            scenario="same_action_different_party_new",
            existing_commitments=[existing_send_docs],
            messages="Gaddy: Dana will also send her documents by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Dana",
                    required_action="Send the documents",
                    deadline="Friday",
                    context="Dana will also send her documents by Friday",
                )
            ],
        ),
        # Update action on existing (change from call to email)
        example(
            category="lifecycle_update_vs_new",
            scenario="update_action_on_existing",
            existing_commitments=[existing_call_supplier],
            messages="Gaddy: actually I'll email the supplier instead of calling",
            expected_commitments=[
                commitment(
                    id="abc-456",
                    committed_party="Gaddy",
                    required_action="Email the supplier",
                    deadline=None,
                    context="actually I'll email the supplier instead of calling",
                )
            ],
        ),
        # New commitment when no existing at all
        example(
            category="lifecycle_update_vs_new",
            scenario="new_when_no_existing",
            existing_commitments=[],
            messages="Gaddy: I'll send the documents tomorrow",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I'll send the documents tomorrow",
                )
            ],
        ),
        # Update deadline on existing with different party
        example(
            category="lifecycle_update_vs_new",
            scenario="update_deadline_different_party_existing",
            existing_commitments=[existing_prepare_report],
            messages="Bob: I need more time, can I prepare the report by next Monday?",
            expected_commitments=[
                commitment(
                    id="abc-300",
                    committed_party="Bob",
                    required_action="Prepare the report",
                    deadline="next Monday",
                    context="I need more time, can I prepare the report by next Monday?",
                )
            ],
        ),
    ]


# ─── F. Completion and Dismissed (12) ──────────────────────────────────


def completion_examples() -> list[dict[str, Any]]:
    existing_docs = _existing(
        id="abc-123",
        committed_party="Gaddy",
        required_action="Send the documents",
        deadline="tomorrow",
        context="I will send the documents by tomorrow",
    )

    existing_supplier = _existing(
        id="abc-456",
        committed_party="Gaddy",
        required_action="Call the supplier",
        deadline=None,
        context="Need to call the supplier about the order",
    )

    existing_report = _existing(
        id="abc-300",
        committed_party="Bob",
        required_action="Prepare the report",
        deadline="Friday",
        context="Bob will prepare the report by Friday",
    )

    existing_invoice = _existing(
        id="abc-400",
        committed_party="Dana",
        required_action="Pay the invoice",
        deadline="Monday",
        context="Dana will pay the invoice by Monday",
    )

    return [
        # Contrastive pair: existing + "sent" → done
        example(
            category="lifecycle_completion",
            scenario="existing_mark_done",
            existing_commitments=[existing_docs],
            messages=(
                "Gaddy: I sent the documents yesterday\n"
                "Gaddy: they should have arrived"
            ),
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I sent the documents yesterday",
                    status="done",
                )
            ],
        ),
        # Contrastive pair: no existing + "sent" → ignore
        example(
            category="lifecycle_completion",
            scenario="past_tense_without_existing_ignore",
            messages="Gaddy: I sent the documents yesterday",
            expected_commitments=[],
        ),
        # Started ≠ done → no update
        example(
            category="lifecycle_completion",
            scenario="started_not_done",
            existing_commitments=[existing_docs],
            messages="Gaddy: I started working on the documents",
            expected_commitments=[],
        ),
        # Almost done ≠ done → no update
        example(
            category="lifecycle_completion",
            scenario="almost_done_not_done",
            existing_commitments=[existing_docs],
            messages="Gaddy: almost done with the documents",
            expected_commitments=[],
        ),
        # Dismiss existing
        example(
            category="lifecycle_completion",
            scenario="dismiss_existing",
            existing_commitments=[existing_supplier],
            messages=(
                "Gaddy: forget about calling the supplier\n"
                "Gaddy: I found another source"
            ),
            expected_commitments=[
                commitment(
                    id="abc-456",
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline=None,
                    context="forget about calling the supplier, I found another source",
                    status="dismissed",
                )
            ],
        ),
        # Passive voice completion
        example(
            category="lifecycle_completion",
            scenario="done_passive_voice",
            existing_commitments=[existing_docs],
            messages="Gaddy: the documents were sent this morning",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="the documents were sent this morning",
                    status="done",
                )
            ],
        ),
        # Dismiss with vague language
        example(
            category="lifecycle_completion",
            scenario="dismiss_vague",
            existing_commitments=[existing_docs],
            messages="Gaddy: never mind about the documents, we don't need them anymore",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="never mind about the documents, we don't need them anymore",
                    status="dismissed",
                )
            ],
        ),
        # Future promise on existing → still waiting (not done)
        example(
            category="lifecycle_completion",
            scenario="future_promise_not_done",
            existing_commitments=[existing_docs],
            messages="Gaddy: I'll definitely send the documents tomorrow as promised",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I'll definitely send the documents tomorrow as promised",
                )
            ],
        ),
        # Done by different party member
        example(
            category="lifecycle_completion",
            scenario="done_by_different_party",
            existing_commitments=[existing_report],
            messages="Alice: Bob finished the report and submitted it",
            expected_commitments=[
                commitment(
                    id="abc-300",
                    committed_party="Bob",
                    required_action="Prepare the report",
                    deadline="Friday",
                    context="Bob finished the report and submitted it",
                    status="done",
                )
            ],
        ),
        # "Done" with matching context
        example(
            category="lifecycle_completion",
            scenario="done_short_confirmation",
            existing_commitments=[existing_invoice],
            messages="Dana: done, the invoice is paid",
            expected_commitments=[
                commitment(
                    id="abc-400",
                    committed_party="Dana",
                    required_action="Pay the invoice",
                    deadline="Monday",
                    context="done, the invoice is paid",
                    status="done",
                )
            ],
        ),
        # Dismiss then new same action → new commitment (dismissed not passed as existing)
        example(
            category="lifecycle_completion",
            scenario="dismiss_then_new_same_action",
            existing_commitments=[],
            messages=(
                "Gaddy: forget about the old supplier order\n"
                "Gaddy: I need to call a new supplier about a different order"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Call a new supplier",
                    deadline=None,
                    context="I need to call a new supplier about a different order",
                )
            ],
        ),
        # "Will do" is not done — it's a promise
        example(
            category="lifecycle_completion",
            scenario="will_do_not_done",
            existing_commitments=[existing_supplier],
            messages="Gaddy: I'll do it, I'll call the supplier today",
            expected_commitments=[
                commitment(
                    id="abc-456",
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline="today",
                    context="I'll do it, I'll call the supplier today",
                )
            ],
        ),
    ]


# ─── Split Logic ───────────────────────────────────────────────────────


def _split_category(examples: list[dict[str, Any]]) -> tuple[list, list, list]:
    """Split a category's examples into train/dev/test.

    For 12 examples: 8 train, 2 dev, 2 test.
    """
    n = len(examples)
    train_end = int(n * 0.67)
    dev_end = train_end + int(n * 0.17)
    train = examples[:train_end]
    dev = examples[train_end:dev_end]
    test = examples[dev_end:]
    return train, dev, test


def build_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    examples.extend(act_vs_ignore_examples())
    examples.extend(party_flip_examples())
    examples.extend(deadline_examples())
    examples.extend(required_action_examples())
    examples.extend(update_vs_new_examples())
    examples.extend(completion_examples())
    return examples


def main() -> None:
    all_examples = build_examples()

    # Build per-category lists for balanced splitting
    categories: dict[str, list[dict[str, Any]]] = {}
    for ex in all_examples:
        cat = ex["category"]
        categories.setdefault(cat, []).append(ex)

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for cat_name in sorted(categories):
        cat_examples = categories[cat_name]
        tr, dv, te = _split_category(cat_examples)
        train.extend(tr)
        dev.extend(dv)
        test.extend(te)

    output_dir = Path(__file__).parent

    for name, data in [
        ("trainset.json", train),
        ("devset.json", dev),
        ("testset.json", test),
    ]:
        path = output_dir / name
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Summary
    print(f"Total: {len(all_examples)} examples")
    print(f"  train: {len(train)}")
    print(f"  dev:   {len(dev)}")
    print(f"  test:  {len(test)}")
    print()
    for cat_name in sorted(categories):
        tr, dv, te = _split_category(categories[cat_name])
        print(f"  {cat_name}: {len(tr)} train, {len(dv)} dev, {len(te)} test")


if __name__ == "__main__":
    main()
