"""Deterministic generator for commitment extraction evaluation examples.

Produces ~90 controlled-probe examples across 6 categories, each tagged with
difficulty (easy/medium/hard). Split into trainset/devset/testset with hard
examples weighted toward dev/test.

Each example tests exactly one behavior whenever possible. Contrastive
pairs (one tiny change flips the expected output) are used to catch
real regressions.

Policy decisions for ambiguous cases are documented inline.

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
    difficulty: str = "medium",
    policy_note: str | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "category": category,
        "scenario": scenario,
        "difficulty": difficulty,
        "chat_id": chat_id,
        "chat_name": chat_name,
        "existing_commitments_json": json.dumps(
            existing_commitments or [],
            ensure_ascii=False,
        ),
        "messages": messages,
        "expected_commitments": expected_commitments,
    }
    if policy_note:
        d["policy_note"] = policy_note
    return d


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


# ─── A. Act vs Ignore (18) ─────────────────────────────────────────────


def act_vs_ignore_examples() -> list[dict[str, Any]]:
    return [
        # --- easy ---
        example(
            category="act_vs_ignore",
            scenario="social_chat_ignore",
            difficulty="easy",
            messages=(
                "Gaddy: hey how are you?\n"
                "Gaddy: long time no see\n"
                "Gaddy: let's catch up soon"
            ),
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="explicit_self_commitment",
            difficulty="easy",
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
            scenario="thanks_only_ignore",
            difficulty="easy",
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
            difficulty="easy",
            messages="Gaddy: you've got this, keep up the great work!",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="explicit_promise_different_action",
            difficulty="easy",
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
        # --- medium ---
        example(
            category="act_vs_ignore",
            scenario="soft_social_future_ignore",
            difficulty="medium",
            messages="Gaddy: let's grab coffee sometime",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="request_plus_acceptance",
            difficulty="medium",
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
            difficulty="medium",
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
            scenario="question_only_ignore",
            difficulty="medium",
            messages="Gaddy: what time is the meeting tomorrow?",
            expected_commitments=[],
        ),
        # --- hard ---
        # Contrastive pair: generic callback without topic → ignore
        example(
            category="act_vs_ignore",
            scenario="generic_callback_ignore",
            difficulty="hard",
            policy_note="Generic callback without a specific topic or deadline is not a commitment.",
            messages="Gaddy: let me think about it and get back to you",
            expected_commitments=[],
        ),
        # Contrastive pair: generic callback WITH topic + deadline → commitment
        example(
            category="act_vs_ignore",
            scenario="generic_callback_with_topic_positive",
            difficulty="hard",
            policy_note="Contrastive pair with generic_callback_ignore: adding topic + deadline makes it a commitment.",
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
            scenario="vague_action_unclear",
            difficulty="hard",
            policy_note="Vague action with no clear owner or deadline → unclear status, not ignored.",
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
        # Request without acceptance → ignore (no one agreed to do it)
        example(
            category="act_vs_ignore",
            scenario="request_without_acceptance_ignore",
            difficulty="hard",
            policy_note="A request with no acceptance or volunteer is not yet a commitment.",
            messages="Alice: can someone send me the report by Friday?",
            expected_commitments=[],
        ),
        # Conditional promise, untriggered → ignore
        example(
            category="act_vs_ignore",
            scenario="conditional_promise_untriggered_ignore",
            difficulty="hard",
            policy_note="Conditional promise where condition has not been met → ignore, not a commitment yet.",
            messages="Gaddy: if they send the quote, I'll approve it",
            expected_commitments=[],
        ),
        # "Should" without "will" → ignore (no commitment to act)
        example(
            category="act_vs_ignore",
            scenario="should_without_will_ignore",
            difficulty="hard",
            policy_note="'We should do X' is an opinion, not a commitment to act.",
            messages="Gaddy: we should really call the supplier about the delay",
            expected_commitments=[],
        ),
        # --- additional hard ignore examples ---
        # Rhetorical question → ignore
        example(
            category="act_vs_ignore",
            scenario="rhetorical_question_ignore",
            difficulty="hard",
            policy_note="Rhetorical questions are not commitments, even if they mention an action.",
            messages="Gaddy: who has time to prepare a full report by Friday?",
            expected_commitments=[],
        ),
        # Hedged commitment with no clear intent → unclear
        example(
            category="act_vs_ignore",
            scenario="hedged_commitment_unclear",
            difficulty="hard",
            policy_note="Hedged language ('might', 'maybe') with no clear commitment → unclear, not ignore.",
            messages="Gaddy: I might be able to send the documents tomorrow, but I'm not sure",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I might be able to send the documents tomorrow, but I'm not sure",
                    status="unclear",
                )
            ],
        ),
        # Past tense completion without existing commitment → ignore
        example(
            category="act_vs_ignore",
            scenario="past_tense_no_existing_ignore",
            difficulty="hard",
            policy_note="Past-tense statements about completed actions without existing commitments → ignore.",
            messages="Gaddy: I already called the supplier yesterday and sorted it out",
            expected_commitments=[],
        ),
    ]


# ─── B. Party Extraction (15) ──────────────────────────────────────────


def party_flip_examples() -> list[dict[str, Any]]:
    return [
        # --- easy ---
        # Contrastive pair: Alice asks Bob → Bob commits
        example(
            category="args_party",
            scenario="alice_asks_bob_bob_accepts",
            difficulty="easy",
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
            difficulty="easy",
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
            scenario="self_commitment_speaker",
            difficulty="easy",
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
            difficulty="easy",
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
        # --- medium ---
        example(
            category="args_party",
            scenario="external_party_waiting",
            difficulty="medium",
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
            scenario="alice_asks_bob_bob_refuses",
            difficulty="medium",
            messages=(
                "Alice: Bob, can you send me the report by Friday?\n"
                "Bob: sorry, I won't be able to do it this week"
            ),
            expected_commitments=[],
        ),
        example(
            category="args_party",
            scenario="multi_party_only_acceptor_committed",
            difficulty="medium",
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
            difficulty="medium",
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
        # --- hard ---
        example(
            category="args_party",
            scenario="unknown_party_unclear",
            difficulty="hard",
            policy_note="'Someone needs to do X' has no committed party → unclear.",
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
            scenario="group_we_need_unclear",
            difficulty="hard",
            policy_note="'We need to do X' has no individual owner → unclear.",
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
        # Party handoff: existing commitment for Alice, new message says Bob will handle it
        example(
            category="args_party",
            scenario="party_handoff_update",
            difficulty="hard",
            policy_note="Party handoff: update existing commitment with new party.",
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
            difficulty="hard",
            policy_note="Speaker asserts another person will do it → commitment with that person as party.",
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
        # --- new hard examples ---
        # Implicit party from context — "the contractor" is the party
        example(
            category="args_party",
            scenario="party_implied_by_role",
            difficulty="hard",
            policy_note="Party implied by role title, not a named person.",
            messages="Gaddy: the contractor needs to finish the renovation by next week",
            expected_commitments=[
                commitment(
                    committed_party="the contractor",
                    required_action="Finish the renovation",
                    deadline="next week",
                    context="the contractor needs to finish the renovation by next week",
                )
            ],
        ),
        # Speaker volunteers someone else without that person confirming
        example(
            category="args_party",
            scenario="speaker_volunteers_other_without_confirmation",
            difficulty="hard",
            policy_note="Speaker names someone else as responsible without their confirmation → commitment with named person as party.",
            messages="Gaddy: Dana will handle the invoice payment by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Dana",
                    required_action="Handle the invoice payment",
                    deadline="Friday",
                    context="Dana will handle the invoice payment by Friday",
                )
            ],
        ),
        # Implicit party from conversational context
        example(
            category="args_party",
            scenario="implicit_party_from_context",
            difficulty="hard",
            policy_note="Party must be inferred from conversational context, not explicitly named in the commitment sentence.",
            messages=(
                "Alice: I talked to the lawyer yesterday\n"
                "Alice: he said he'll file the motion by Wednesday"
            ),
            expected_commitments=[
                commitment(
                    committed_party="the lawyer",
                    required_action="File the motion",
                    deadline="Wednesday",
                    context="he said he'll file the motion by Wednesday",
                )
            ],
        ),
    ]


# ─── C. Deadline Extraction (15) ───────────────────────────────────────


def deadline_examples() -> list[dict[str, Any]]:
    # (phrase, expected_deadline, difficulty)
    cases: list[tuple[str, str | None, str]] = [
        ("by Friday", "Friday", "easy"),
        ("by tomorrow", "tomorrow", "easy"),
        ("today", "today", "easy"),
        ("next Monday", "next Monday", "easy"),
        ("by 5pm", "by 5pm", "easy"),
        ("by 17:00", "by 17:00", "medium"),
        ("by July 15", "July 15", "medium"),
        ("by end of week", "end of week", "medium"),
        ("next week", "next week", "medium"),
        ("soon", None, "hard"),
        ("when I can", None, "hard"),
    ]

    examples: list[dict[str, Any]] = []

    for phrase, expected_deadline, diff in cases:
        scenario_name = f"deadline_{phrase.replace(' ', '_').replace(':', '')}"
        policy = None
        if expected_deadline is None:
            policy = f"'{phrase}' is not a concrete deadline → null."
        examples.append(
            example(
                category="args_deadline",
                scenario=scenario_name,
                difficulty=diff,
                policy_note=policy,
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
            difficulty="easy",
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

    # --- new hard examples ---
    # Relative deadline with context
    examples.append(
        example(
            category="args_deadline",
            scenario="deadline_relative_context",
            difficulty="hard",
            policy_note="Relative deadline 'by end of the quarter' is a valid explicit deadline.",
            messages="Gaddy: I'll send the documents by end of the quarter",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="end of the quarter",
                    context="I'll send the documents by end of the quarter",
                )
            ],
        )
    )

    # Implied deadline from event
    examples.append(
        example(
            category="args_deadline",
            scenario="deadline_implied_from_event",
            difficulty="hard",
            policy_note="Deadline implied by an event ('before the meeting') is a valid explicit deadline.",
            messages="Gaddy: I'll send the documents before the meeting",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="before the meeting",
                    context="I'll send the documents before the meeting",
                )
            ],
        )
    )

    # Conflicting deadlines in multi-message — use the most specific/recent one
    examples.append(
        example(
            category="args_deadline",
            scenario="deadline_conflicting_multi_message",
            difficulty="hard",
            policy_note="When two deadlines are mentioned, use the most recent/specific one.",
            messages=(
                "Gaddy: I'll send the documents by Friday\n"
                "Gaddy: actually, let me send them by tomorrow instead"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="actually, let me send them by tomorrow instead",
                )
            ],
        )
    )

    return examples


# ─── D. Required Action Extraction (15) ────────────────────────────────


def required_action_examples() -> list[dict[str, Any]]:
    return [
        # --- easy ---
        example(
            category="args_required_action",
            scenario="send_docs_forward",
            difficulty="easy",
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
        example(
            category="args_required_action",
            scenario="call_supplier_ring",
            difficulty="easy",
            messages="Gaddy: I need to call the supplier about the order",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline=None,
                    context="I need to call the supplier about the order",
                )
            ],
        ),
        example(
            category="args_required_action",
            scenario="prepare_report",
            difficulty="easy",
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
            scenario="book_meeting_room_reserve",
            difficulty="easy",
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
            difficulty="easy",
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
        # --- medium ---
        example(
            category="args_required_action",
            scenario="send_docs_send_over",
            difficulty="medium",
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
            difficulty="medium",
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
            scenario="call_supplier_phone",
            difficulty="medium",
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
            difficulty="medium",
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
            difficulty="medium",
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
            scenario="prepare_report_draft",
            difficulty="medium",
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
        # --- hard ---
        example(
            category="args_required_action",
            scenario="process_loan",
            difficulty="hard",
            policy_note="External party as committed_party with action inferred from context.",
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
        # --- new hard examples ---
        # Very indirect action
        example(
            category="args_required_action",
            scenario="very_indirect_action",
            difficulty="hard",
            policy_note="Vague indirect action ('make sure it's taken care of') → extract the underlying action.",
            messages="Gaddy: I'll make sure the invoice is taken care of by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Take care of the invoice",
                    deadline="Friday",
                    context="I'll make sure the invoice is taken care of by Friday",
                )
            ],
        ),
        # Multi-step action
        example(
            category="args_required_action",
            scenario="multi_step_action",
            difficulty="hard",
            policy_note="Multi-step action described in one message → single commitment with combined action.",
            messages="Gaddy: I'll review the contract, sign it, and send it back by tomorrow",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Review, sign, and send back the contract",
                    deadline="tomorrow",
                    context="I'll review the contract, sign it, and send it back by tomorrow",
                )
            ],
        ),
        # Action split across messages
        example(
            category="args_required_action",
            scenario="action_split_across_messages",
            difficulty="hard",
            policy_note="Action details split across multiple messages → single commitment combining the information.",
            messages=(
                "Gaddy: I need to handle the paperwork\n"
                "Gaddy: specifically the vendor registration forms\n"
                "Gaddy: I'll do it by Wednesday"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Handle the vendor registration forms",
                    deadline="Wednesday",
                    context="I need to handle the paperwork, specifically the vendor registration forms",
                )
            ],
        ),
    ]


# ─── E. New vs Update (15) ─────────────────────────────────────────────


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
        # --- easy ---
        # Update existing deadline
        example(
            category="lifecycle_update_vs_new",
            scenario="update_existing_deadline",
            difficulty="easy",
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
            difficulty="easy",
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
        # New commitment when no existing at all
        example(
            category="lifecycle_update_vs_new",
            scenario="new_when_no_existing",
            difficulty="easy",
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
        # --- medium ---
        # Same existing, no duplicate — update deadline only
        example(
            category="lifecycle_update_vs_new",
            scenario="same_existing_no_duplicate",
            difficulty="medium",
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
        # New when existing exists but different action
        example(
            category="lifecycle_update_vs_new",
            scenario="new_when_existing_different_action",
            difficulty="medium",
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
            difficulty="medium",
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
            difficulty="medium",
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
        # --- hard ---
        # Party handoff: Alice → Bob
        example(
            category="lifecycle_update_vs_new",
            scenario="party_handoff_update",
            difficulty="hard",
            policy_note="Party handoff: update existing commitment with new party.",
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
        # Multiple existing, update only one
        example(
            category="lifecycle_update_vs_new",
            scenario="multiple_existing_update_one",
            difficulty="hard",
            policy_note="With multiple existing commitments, only update the one that changed.",
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
            difficulty="hard",
            policy_note="Same action but different party → new commitment, not an update.",
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
            difficulty="hard",
            policy_note="Action change on existing commitment → update with new required_action.",
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
        # Update deadline on existing with different party
        example(
            category="lifecycle_update_vs_new",
            scenario="update_deadline_different_party_existing",
            difficulty="hard",
            policy_note="Deadline update by the committed party themselves.",
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
        # --- new hard examples ---
        # Same action, same party, different context — is it an update or no-op?
        example(
            category="lifecycle_update_vs_new",
            scenario="same_action_same_party_context_change",
            difficulty="hard",
            policy_note="Same action+party but new context info → update context on existing.",
            existing_commitments=[existing_send_docs],
            messages="Gaddy: I'll send the documents once the legal team approves them",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I'll send the documents once the legal team approves them",
                )
            ],
        ),
        # Multiple fields changed at once
        example(
            category="lifecycle_update_vs_new",
            scenario="multiple_fields_changed",
            difficulty="hard",
            policy_note="Multiple fields changed simultaneously: deadline + context.",
            existing_commitments=[existing_call_supplier],
            messages="Gaddy: I'll call the supplier about the new order today instead of the old one",
            expected_commitments=[
                commitment(
                    id="abc-456",
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline="today",
                    context="I'll call the supplier about the new order today instead of the old one",
                )
            ],
        ),
        # New commitment that looks like an update (no existing match)
        example(
            category="lifecycle_update_vs_new",
            scenario="new_that_looks_like_update",
            difficulty="hard",
            policy_note="New commitment with similar action to existing but different enough → new, not update.",
            existing_commitments=[existing_send_docs],
            messages="Gaddy: I also need to send the contract to the client by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Send the contract to the client",
                    deadline="Friday",
                    context="I also need to send the contract to the client by Friday",
                )
            ],
        ),
    ]


# ─── F. Completion and Dismissed (15) ──────────────────────────────────


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
        # --- easy ---
        # Contrastive pair: existing + "sent" → done
        example(
            category="lifecycle_completion",
            scenario="existing_mark_done",
            difficulty="easy",
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
            difficulty="easy",
            messages="Gaddy: I sent the documents yesterday",
            expected_commitments=[],
        ),
        # Dismiss existing
        example(
            category="lifecycle_completion",
            scenario="dismiss_existing",
            difficulty="easy",
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
        # "Done" with matching context
        example(
            category="lifecycle_completion",
            scenario="done_short_confirmation",
            difficulty="easy",
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
        # --- medium ---
        # Passive voice completion
        example(
            category="lifecycle_completion",
            scenario="done_passive_voice",
            difficulty="medium",
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
            difficulty="medium",
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
            difficulty="medium",
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
            difficulty="medium",
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
        # --- hard ---
        # Started ≠ done → no update
        example(
            category="lifecycle_completion",
            scenario="started_not_done",
            difficulty="hard",
            policy_note="Progress updates do not complete existing commitments.",
            existing_commitments=[existing_docs],
            messages="Gaddy: I started working on the documents",
            expected_commitments=[],
        ),
        # Almost done ≠ done → no update
        example(
            category="lifecycle_completion",
            scenario="almost_done_not_done",
            difficulty="hard",
            policy_note="Being almost done is not done — no update.",
            existing_commitments=[existing_docs],
            messages="Gaddy: almost done with the documents",
            expected_commitments=[],
        ),
        # Dismiss then new same action → new commitment
        example(
            category="lifecycle_completion",
            scenario="dismiss_then_new_same_action",
            difficulty="hard",
            policy_note="Dismiss old + new commitment with same action but different target → new only.",
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
            difficulty="hard",
            policy_note="'Will do' is a future promise, not a completion.",
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
        # --- new hard examples ---
        # Completion reported by third party not in chat
        example(
            category="lifecycle_completion",
            scenario="third_party_completion_report",
            difficulty="hard",
            policy_note="Third-party reports completion of someone else's commitment → done.",
            existing_commitments=[existing_report],
            messages="Charlie: Alice told me Bob submitted the report yesterday",
            expected_commitments=[
                commitment(
                    id="abc-300",
                    committed_party="Bob",
                    required_action="Prepare the report",
                    deadline="Friday",
                    context="Alice told me Bob submitted the report yesterday",
                    status="done",
                )
            ],
        ),
        # Dismiss with condition — keep waiting, not dismissed
        example(
            category="lifecycle_completion",
            scenario="dismiss_with_condition",
            difficulty="hard",
            policy_note="Conditional dismiss ('unless they respond today') → keep waiting, update deadline.",
            existing_commitments=[existing_supplier],
            messages="Gaddy: forget about calling the supplier unless they respond today",
            expected_commitments=[
                commitment(
                    id="abc-456",
                    committed_party="Gaddy",
                    required_action="Call the supplier",
                    deadline="today",
                    context="forget about calling the supplier unless they respond today",
                )
            ],
        ),
        # Partial completion — not done
        example(
            category="lifecycle_completion",
            scenario="partial_completion_not_done",
            difficulty="hard",
            policy_note="Partial completion ('sent 3 out of 5 documents') → not done, still waiting.",
            existing_commitments=[existing_docs],
            messages="Gaddy: I sent 3 out of the 5 documents, will send the rest tomorrow",
            expected_commitments=[
                commitment(
                    id="abc-123",
                    committed_party="Gaddy",
                    required_action="Send the documents",
                    deadline="tomorrow",
                    context="I sent 3 out of the 5 documents, will send the rest tomorrow",
                )
            ],
        ),
    ]


# ─── Split Logic ───────────────────────────────────────────────────────


_SPLIT_RATIOS = {
    "easy": (0.70, 0.15, 0.15),
    "medium": (0.50, 0.25, 0.25),
    "hard": (0.35, 0.32, 0.33),
}

_ALL_CATEGORIES = [
    "act_vs_ignore",
    "args_party",
    "args_deadline",
    "args_required_action",
    "lifecycle_update_vs_new",
    "lifecycle_completion",
]


def _split_bucket(examples: list[dict[str, Any]], difficulty: str) -> tuple[list, list, list]:
    """Split a single (category, difficulty) bucket by target ratios."""
    r_train, r_dev, _ = _SPLIT_RATIOS[difficulty]
    n = len(examples)
    train_end = max(1, int(n * r_train)) if n > 1 else 1
    dev_end = train_end + max(1, int(n * r_dev)) if n > 2 else train_end
    if n <= 1:
        return examples, [], []
    if n == 2:
        return examples[:1], examples[1:], []
    train = examples[:train_end]
    dev = examples[train_end:dev_end]
    test = examples[dev_end:]
    return train, dev, test


def _assert_has_hard_per_category(examples: list[dict[str, Any]], split_name: str) -> None:
    """Assert every category has at least 1 hard example in this split."""
    hard_cats = {ex["category"] for ex in examples if ex.get("difficulty") == "hard"}
    missing = set(_ALL_CATEGORIES) - hard_cats
    if missing:
        raise AssertionError(
            f"{split_name} missing hard examples for categories: {sorted(missing)}"
        )


def _assert_split_has_all_categories(examples: list[dict[str, Any]], split_name: str) -> None:
    """Assert all 6 categories are present in this split."""
    cats = {ex["category"] for ex in examples}
    missing = set(_ALL_CATEGORIES) - cats
    if missing:
        raise AssertionError(
            f"{split_name} missing categories: {sorted(missing)}"
        )


def build_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    examples.extend(act_vs_ignore_examples())
    examples.extend(party_flip_examples())
    examples.extend(deadline_examples())
    examples.extend(required_action_examples())
    examples.extend(update_vs_new_examples())
    examples.extend(completion_examples())
    return examples


# ─── Challenge Split: Act vs Ignore ────────────────────────────────────


def challenge_act_ignore_examples() -> list[dict[str, Any]]:
    """Dedicated challenge split focused on act vs ignore boundary.

    ~36 examples, roughly 50% act / 50% ignore, mostly hard contrastive pairs.
    """
    return [
        # --- IGNORE cases (hard) ---
        example(
            category="act_vs_ignore",
            scenario="challenge_request_no_acceptance",
            difficulty="hard",
            policy_note="Request with no acceptance or volunteer → ignore.",
            messages="Alice: can someone prepare the slides for Monday?",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_conditional_untriggered",
            difficulty="hard",
            policy_note="Conditional promise, condition not met → ignore.",
            messages="Gaddy: if the client approves the budget, I'll start the project",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_generic_callback",
            difficulty="hard",
            policy_note="Generic callback without topic or deadline → ignore.",
            messages="Bob: I'll get back to you on that",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_should_not_will",
            difficulty="hard",
            policy_note="'Should' is opinion, not commitment → ignore.",
            messages="Gaddy: we should really update the website at some point",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_rhetorical_question",
            difficulty="hard",
            policy_note="Rhetorical question mentioning action → ignore.",
            messages="Gaddy: who has time to write a full proposal this week?",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_past_tense_no_existing",
            difficulty="hard",
            policy_note="Past-tense completion without existing commitment → ignore.",
            messages="Gaddy: I already sent the email to the team yesterday",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_social_vague",
            difficulty="hard",
            policy_note="Vague social intent without specifics → ignore.",
            messages="Gaddy: we should all hang out sometime soon",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_opinion_not_commitment",
            difficulty="hard",
            policy_note="Expressing opinion about what needs doing → ignore.",
            messages="Gaddy: the report really needs to be more detailed",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_question_about_past",
            difficulty="hard",
            policy_note="Question about past events → ignore.",
            messages="Alice: did anyone follow up with the vendor last week?",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_suggestion_not_promise",
            difficulty="hard",
            policy_note="Suggesting an approach is not committing to it → ignore.",
            messages="Bob: maybe we could try reaching out to the new supplier",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_hope_not_commitment",
            difficulty="hard",
            policy_note="Expressing hope/wish is not a commitment → ignore.",
            messages="Gaddy: I hope we can finish the migration by next month",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_report_status_no_action",
            difficulty="hard",
            policy_note="Reporting current status without committing to action → ignore.",
            messages="Gaddy: the server is down again, looking into it",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_appreciation_not_commitment",
            difficulty="hard",
            policy_note="Thanking someone for future help is not a commitment → ignore.",
            messages="Alice: thanks in advance for handling the client call",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_reminder_not_commitment",
            difficulty="hard",
            policy_note="Reminding someone else is not a commitment by the speaker → ignore.",
            messages="Gaddy: don't forget Bob needs the data by Friday",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_worry_not_commitment",
            difficulty="hard",
            policy_note="Expressing concern is not a commitment → ignore.",
            messages="Gaddy: I'm worried we won't meet the deadline for the launch",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_information_sharing",
            difficulty="hard",
            policy_note="Sharing information without action item → ignore.",
            messages="Gaddy: the client said they'll review the proposal and get back to us next week",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_almost_done_no_existing",
            difficulty="hard",
            policy_note="Progress report without existing commitment → ignore.",
            messages="Gaddy: I'm almost done with the presentation",
            expected_commitments=[],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_started_no_existing",
            difficulty="hard",
            policy_note="Started working without existing commitment → ignore.",
            messages="Gaddy: I started looking into the bug",
            expected_commitments=[],
        ),

        # --- ACT cases (hard contrastive pairs) ---
        example(
            category="act_vs_ignore",
            scenario="challenge_request_with_acceptance",
            difficulty="hard",
            policy_note="Request + explicit acceptance → commitment.",
            messages=(
                "Alice: can you send me the slides by Monday?\n"
                "Bob: sure, I'll send them by Monday"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Send the slides",
                    deadline="Monday",
                    context="Alice asked Bob to send the slides by Monday, Bob agreed",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_generic_callback_with_topic",
            difficulty="hard",
            policy_note="Contrastive pair: generic callback but WITH topic + deadline → commitment.",
            messages=(
                "Alice: what about the budget?\n"
                "Bob: let me review the budget and get back to you tomorrow"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Get back to Alice about the budget",
                    deadline="tomorrow",
                    context="let me review the budget and get back to you tomorrow",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_explicit_self_promise",
            difficulty="hard",
            policy_note="Explicit 'I will' with specific action and deadline → commitment.",
            messages="Gaddy: I will definitely call the client before noon",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Call the client",
                    deadline="before noon",
                    context="I will definitely call the client before noon",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_volunteer",
            difficulty="hard",
            policy_note="Volunteering to do something specific → commitment.",
            messages=(
                "Alice: we need someone to handle the vendor registration\n"
                "Bob: I'll take care of the vendor registration by Wednesday"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Handle the vendor registration",
                    deadline="Wednesday",
                    context="I'll take care of the vendor registration by Wednesday",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_conditional_triggered",
            difficulty="hard",
            policy_note="Conditional promise where condition IS satisfied → commitment.",
            messages=(
                "Alice: the client just approved the budget!\n"
                "Gaddy: great, I'll start the project tomorrow then"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Start the project",
                    deadline="tomorrow",
                    context="the client approved the budget, I'll start the project tomorrow",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_will_with_specific_action",
            difficulty="hard",
            policy_note="'I will' with concrete action but no deadline → commitment without deadline.",
            messages="Gaddy: I'll prepare the quarterly report",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Prepare the quarterly report",
                    deadline=None,
                    context="I'll prepare the quarterly report",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_third_party_commits",
            difficulty="hard",
            policy_note="Speaker asserts someone else will do something → commitment with that party.",
            messages="Gaddy: Dana will send the contract to the client by Friday",
            expected_commitments=[
                commitment(
                    committed_party="Dana",
                    required_action="Send the contract to the client",
                    deadline="Friday",
                    context="Dana will send the contract to the client by Friday",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_hedged_unclear",
            difficulty="hard",
            policy_note="Hedged commitment → unclear status, not ignore.",
            messages="Gaddy: I might be able to review the code tomorrow, but I'm not sure",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Review the code",
                    deadline="tomorrow",
                    context="I might be able to review the code tomorrow, but I'm not sure",
                    status="unclear",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_multi_message_commitment",
            difficulty="hard",
            policy_note="Commitment built across multiple messages → commitment.",
            messages=(
                "Gaddy: I need to deal with the invoice\n"
                "Gaddy: I'll pay it by end of week"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Pay the invoice",
                    deadline="end of week",
                    context="I'll pay it by end of week",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_implicit_acceptance",
            difficulty="hard",
            policy_note="Request followed by action-oriented response → commitment.",
            messages=(
                "Alice: can you update the spreadsheet?\n"
                "Bob: on it"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Update the spreadsheet",
                    deadline=None,
                    context="Alice asked Bob to update the spreadsheet, Bob said 'on it'",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_waiting_external",
            difficulty="hard",
            policy_note="Waiting for external party to act → commitment with external party.",
            messages=(
                "Gaddy: I submitted the application last week\n"
                "Gaddy: now just waiting for the city to approve the permit"
            ),
            expected_commitments=[
                commitment(
                    committed_party="the city",
                    required_action="Approve the permit",
                    deadline=None,
                    context="waiting for the city to approve the permit",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_let_me_with_action",
            difficulty="hard",
            policy_note="'Let me X' with specific action → commitment.",
            messages="Gaddy: let me check with the accountant and I'll update you today",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Check with the accountant and update",
                    deadline="today",
                    context="let me check with the accountant and I'll update you today",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_responsibility_claim",
            difficulty="hard",
            policy_note="Claiming responsibility for a task → commitment.",
            messages="Gaddy: I'll handle the client meeting on Thursday",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Handle the client meeting",
                    deadline="Thursday",
                    context="I'll handle the client meeting on Thursday",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_need_to_with_intent",
            difficulty="hard",
            policy_note="'I need to X' with clear intent to act → commitment.",
            messages="Gaddy: I need to call the bank about the fees today",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Call the bank about the fees",
                    deadline="today",
                    context="I need to call the bank about the fees today",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_confirm_and_commit",
            difficulty="hard",
            policy_note="Confirming a task and committing to deadline → commitment.",
            messages=(
                "Alice: did you confirm the venue?\n"
                "Bob: not yet, I'll call them this afternoon"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Call the venue to confirm",
                    deadline="this afternoon",
                    context="not yet, I'll call them this afternoon",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_plan_with_action",
            difficulty="hard",
            policy_note="Stating a plan with specific action and timing → commitment.",
            messages="Gaddy: I'm going to submit the proposal by Friday end of day",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Submit the proposal",
                    deadline="Friday end of day",
                    context="I'm going to submit the proposal by Friday end of day",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_offer_help",
            difficulty="hard",
            policy_note="Offering to do something specific → commitment.",
            messages="Gaddy: I can review the contract and send comments by tomorrow",
            expected_commitments=[
                commitment(
                    committed_party="Gaddy",
                    required_action="Review the contract and send comments",
                    deadline="tomorrow",
                    context="I can review the contract and send comments by tomorrow",
                )
            ],
        ),
        example(
            category="act_vs_ignore",
            scenario="challenge_commit_after_question",
            difficulty="hard",
            policy_note="Question followed by commitment to act → commitment.",
            messages=(
                "Alice: has anyone followed up with the vendor?\n"
                "Bob: not yet, I'll email them today"
            ),
            expected_commitments=[
                commitment(
                    committed_party="Bob",
                    required_action="Email the vendor",
                    deadline="today",
                    context="not yet, I'll email them today",
                )
            ],
        ),
    ]


def main() -> None:
    all_examples = build_examples()

    # Group by (category, difficulty) buckets
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ex in all_examples:
        key = (ex["category"], ex["difficulty"])
        buckets.setdefault(key, []).append(ex)

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for (cat, diff), bucket_examples in sorted(buckets.items()):
        tr, dv, te = _split_bucket(bucket_examples, diff)
        train.extend(tr)
        dev.extend(dv)
        test.extend(te)

    # Coverage assertions
    _assert_split_has_all_categories(train, "train")
    _assert_split_has_all_categories(dev, "dev")
    _assert_split_has_all_categories(test, "test")
    _assert_has_hard_per_category(dev, "dev")
    _assert_has_hard_per_category(test, "test")

    output_dir = Path(__file__).parent

    for name, data in [
        ("trainset.json", train),
        ("devset.json", dev),
        ("testset.json", test),
        ("challenge_act_ignore.json", challenge_act_ignore_examples()),
    ]:
        path = output_dir / name
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Summary
    challenge = challenge_act_ignore_examples()
    challenge_ignore = sum(1 for e in challenge if not e["expected_commitments"])
    challenge_act = len(challenge) - challenge_ignore
    print(f"Total: {len(all_examples)} examples")
    print(f"  train: {len(train)}")
    print(f"  dev:   {len(dev)}")
    print(f"  test:  {len(test)}")
    print(f"  challenge: {len(challenge)} ({challenge_act} act / {challenge_ignore} ignore)")
    print()

    # Per-category × difficulty summary
    for cat_name in _ALL_CATEGORIES:
        parts = []
        for diff in ("easy", "medium", "hard"):
            bucket = buckets.get((cat_name, diff), [])
            tr = [e for e in bucket if e in train]
            dv = [e for e in bucket if e in dev]
            te = [e for e in bucket if e in test]
            parts.append(f"{diff}: {len(tr)}t/{len(dv)}d/{len(te)}t")
        print(f"  {cat_name}: {' | '.join(parts)}")


if __name__ == "__main__":
    main()
