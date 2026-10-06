"""Session-boundary flush behaviour, asserted through the provider interface.

The four ways a session can end each reach the provider differently:

  1. ``on_session_end`` alone — CLI exit / TUI+desktop finalize / gateway shutdown call
     ``MemoryManager.on_session_end()`` -> ``provider.on_session_end()``
  2. ``on_session_switch`` — ``/new``, ``/reset``, compression (rotates ids mid-process)
  3. ``shutdown`` — process teardown (``MemoryManager.shutdown_all()``)
  4. end then switch — CLI ``/new`` runs both hooks as one serialized task

With ``retain_every_n_turns > 1`` a turn buffered below the cadence must not vanish.
``sync_turn`` only enqueues at the modulo boundary, so something has to flush the tail
at a real session boundary. The recording ``FakeClient`` makes each retain observable,
so these tests state exactly which paths ship the tail — and which do not.

Requested by @carlosrve in vectorize-io/hindsight#5218 (via #4662).
"""

import json

from conftest import FakeClient

CADENCE = 8  # > the 2 turns each test buffers, so nothing ships without a flush


def _buffered(provider):
    """A provider with two turns buffered below the cadence — the lost-tail case."""
    instance, fake = provider({"retain_every_n_turns": CADENCE})
    instance.sync_turn("first question", "first answer")
    instance.sync_turn("second question", "second answer")
    return instance, fake


def _message_dicts(fake: FakeClient, index: int = 0) -> list[dict]:
    """The retained conversation, unwrapping per-turn arrays.

    Upstream main still nests (one array per turn — see #5262); the flat form is what
    the server's turn-aware chunker wants. Both shapes are accepted so the flush
    assertions do not depend on that separate chunking fix.
    """
    payload = json.loads(fake.retains[index]["items"][0]["content"])
    if payload and isinstance(payload[0], list):
        payload = [m for turn in payload for m in turn]
    return payload


def _turn_texts(fake: FakeClient, index: int = 0) -> list[str]:
    return [m["content"] for m in _message_dicts(fake, index)]


EXPECTED_TAIL = [
    "User: first question",
    "Assistant: first answer",
    "User: second question",
    "Assistant: second answer",
]


def test_session_end_alone_flushes_below_cadence(provider):
    """(1) ``on_session_end`` with no switch and no shutdown must ship the buffered tail.

    This is the reported gap: the base hook is a no-op, so without an override the turns
    are dropped on CLI exit / desktop finalize / gateway shutdown. ``shutdown()`` only
    drains jobs already on the writer queue, so it cannot stand in for the missing flush.
    """
    instance, fake = _buffered(provider)
    assert fake.retains == []  # 2 < 8: nothing shipped yet
    assert len(instance._session_turns) == 2  # both turns still buffered

    instance.on_session_end([])
    assert instance._session_turns == []  # buffer handed to the writer, not just left

    instance.shutdown()  # drains the writer; does NOT itself flush the buffer
    assert len(fake.retains) == 1
    assert _turn_texts(fake) == EXPECTED_TAIL


def test_session_switch_flushes_below_cadence(provider):
    """(2) The existing flush-on-switch path keeps working: the tail ships under the OLD
    document id before the rotation, and the new session starts from an empty buffer."""
    instance, fake = _buffered(provider)

    instance.on_session_switch("session-2")
    instance.shutdown()

    assert len(fake.retains) == 1
    assert fake.retains[0]["document_id"] == "session-1"
    assert "session:session-1" in fake.retains[0]["items"][0]["tags"]
    assert _turn_texts(fake) == EXPECTED_TAIL


def test_shutdown_alone_does_not_flush_below_cadence(provider):
    """(3) Recorded, not fixed: ``shutdown()`` alone never flushes the buffer — it only
    drains the writer queue, so a below-cadence tail is lost here.

    This is the second layer tracked in vectorize-io/hindsight#5218 (a durable outbox),
    which the ``on_session_end`` override does NOT cover. Asserting the loss keeps that
    boundary explicit instead of letting a green suite imply it is solved.
    """
    instance, fake = _buffered(provider)
    instance.shutdown()
    assert fake.retains == []


def test_session_end_then_switch_retains_the_tail_once(provider):
    """(4) CLI ``/new`` runs end then switch as one serialized task. The end flush must
    clear the buffer so the switch sees nothing left to send; otherwise the same turns
    ship twice (append-mode duplicate, hermes-agent#41911)."""
    instance, fake = _buffered(provider)

    instance.on_session_end([])
    instance.on_session_switch("session-2")
    instance.shutdown()

    assert len(fake.retains) == 1, "the buffered tail must ship exactly once"
    assert _turn_texts(fake) == EXPECTED_TAIL
