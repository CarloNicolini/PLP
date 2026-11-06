"""Retry layer in ``plp._llm`` — patches transient failures on a real client."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

import plp
from plp import PLPChatClient, PLPState, asample, sample
from plp._llm import (
    RetryConfig,
    ainvoke_llm,
    ainvoke_llm_with_attempts,
    get_retry_config,
    invoke_llm,
    invoke_llm_with_attempts,
    set_retry_config,
)


@pytest.fixture(autouse=True)
def _fast_retries() -> None:
    original = get_retry_config()
    set_retry_config(RetryConfig(max_attempts=4, max_wait_s=0.01, multiplier=0.001))
    try:
        yield
    finally:
        set_retry_config(original)


def test_invoke_retries_then_succeeds(llm: PLPChatClient) -> None:
    real = llm.invoke
    state = {"n": 0}

    def flaky(messages):  # type: ignore[no-untyped-def]
        state["n"] += 1
        if state["n"] <= 2:
            raise TimeoutError("transient")
        return real(messages)

    with mock.patch.object(llm, "invoke", side_effect=flaky):
        output, attempts = invoke_llm_with_attempts(llm, "Reply with: hi")
    assert "hi" in output.lower()
    assert attempts == 3


def test_invoke_exhausts_attempts(llm: PLPChatClient) -> None:
    with mock.patch.object(llm, "invoke", side_effect=TimeoutError("fail")):
        with pytest.raises(TimeoutError):
            invoke_llm_with_attempts(llm, "ping")


def test_invoke_no_retry_on_value_error(llm: PLPChatClient) -> None:
    calls = {"n": 0}

    def bad(messages):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise ValueError("not retryable")

    with mock.patch.object(llm, "invoke", side_effect=bad):
        with pytest.raises(ValueError):
            invoke_llm_with_attempts(llm, "ping")
    assert calls["n"] == 1


def test_retry_disabled_single_shot(llm: PLPChatClient) -> None:
    prior = get_retry_config()
    set_retry_config(RetryConfig(enabled=False))
    try:
        calls = {"n": 0}
        real = llm.invoke

        def once_timeout(messages):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("x")
            return real(messages)

        with mock.patch.object(llm, "invoke", side_effect=once_timeout):
            with pytest.raises(TimeoutError):
                invoke_llm_with_attempts(llm, "ping")
        assert calls["n"] == 1
    finally:
        set_retry_config(prior)


def test_invoke_llm_string_return(llm: PLPChatClient) -> None:
    out = invoke_llm(llm, "Reply with the single word: pong")
    assert "pong" in out.lower()


def test_async_retry_and_string_helpers(llm: PLPChatClient) -> None:
    real = llm.ainvoke
    state = {"n": 0}

    async def flaky(messages):  # type: ignore[no-untyped-def]
        state["n"] += 1
        if state["n"] <= 1:
            raise TimeoutError("transient")
        return await real(messages)

    with mock.patch.object(llm, "ainvoke", side_effect=flaky):
        out, attempts = asyncio.run(ainvoke_llm_with_attempts(llm, "Say: ok"))
    assert "ok" in out.lower()
    assert attempts == 2

    plain = asyncio.run(ainvoke_llm(llm, "Say: yes"))
    assert "yes" in plain.lower()


def test_sample_records_llm_attempts(llm: PLPChatClient) -> None:
    real = llm.invoke
    state = {"n": 0}

    def flaky(messages):  # type: ignore[no-untyped-def]
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("x")
        return real(messages)

    @sample(llm, name="answer")
    def ask(s: PLPState, q: str) -> str:
        return f"{q} One digit only."

    with mock.patch.object(llm, "invoke", side_effect=flaky):
        st, out = ask(PLPState(), "2+2=")
    assert any(c.isdigit() for c in out)
    events = [e for e in st.trace if e.primitive == "answer"]
    assert events[-1].meta.get("llm_attempts") == 2


def test_asample_records_llm_attempts(llm: PLPChatClient) -> None:
    real = llm.ainvoke
    state = {"n": 0}

    async def flaky(messages):  # type: ignore[no-untyped-def]
        state["n"] += 1
        if state["n"] <= 2:
            raise TimeoutError("x")
        return await real(messages)

    @asample(llm, name="answer")
    def ask(s: PLPState, q: str) -> str:
        return f"{q} One digit only."

    async def run() -> None:
        with mock.patch.object(llm, "ainvoke", side_effect=flaky):
            return await ask(PLPState(), "2+2=")

    st, out = asyncio.run(run())
    assert any(c.isdigit() for c in out)
    events = [e for e in st.trace if e.primitive == "answer"]
    assert events[-1].meta.get("llm_attempts") == 3
