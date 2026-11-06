"""Async primitives, plate, and handler context — live LLM."""

from __future__ import annotations

import asyncio

import plp
from plp import PLPChatClient, PLPState


def test_ainvoke_llm_live(llm: PLPChatClient) -> None:
    async def run() -> None:
        out = await plp.ainvoke_llm(
            llm,
            "Reply with the single word hello.",
            system_prompt="You follow instructions exactly.",
        )

        assert "hello" in out.lower()

    asyncio.run(run())


def test_asample_live(llm: PLPChatClient) -> None:
    @plp.asample(llm, name="async_site")
    def site(state: PLPState, q: str) -> str:
        return f"{q} Reply with just the digit 4."

    async def run() -> None:
        state, out = await site(PLPState(), "2+2=")
        assert "4" in out
        assert state.trace[-1].primitive == "async_site"

    asyncio.run(run())


def test_plate_exposes_indices_and_scale() -> None:
    with plp.plate("candidates", 5, subsample_size=2) as ctx:
        assert list(ctx) == ctx.indices
        assert len(ctx) == 2
        assert ctx.scale == 2.5


def test_plate_map_live(llm: PLPChatClient) -> None:
    @plp.sample(llm, name="sync_site")
    def site(state: PLPState, question: str) -> str:
        return f"{question} Reply with the single word done."

    states, outputs = plp.plate("batch", 2).map(site, "Why use plate?")
    assert len(states) == 2
    assert len(outputs) == 2
    assert all("done" in output.lower() for output in outputs)


def test_plate_amap_live(llm: PLPChatClient) -> None:
    @plp.asample(llm, name="async_site")
    def site(state: PLPState, question: str) -> str:
        return f"{question} Reply with the single word ok."

    async def run() -> None:
        states, outputs = await plp.plate("batch", 2).amap(
            site,
            "Why use plate?",
            max_concurrency=2,
        )
        assert len(states) == 2
        assert all("ok" in output.lower() for output in outputs)

    asyncio.run(run())


def test_handler_contexts_task_local(llm: PLPChatClient) -> None:
    @plp.asample(llm, name="shared_site")
    def site(state: PLPState) -> str:
        return "ignored"

    async def run_with(value: str) -> str:
        with plp.do({"shared_site": value}):
            await asyncio.sleep(0)
            _, out = await site(PLPState())
            return out

    async def run() -> None:
        results = await asyncio.gather(run_with("A"), run_with("B"))
        assert results == ["A", "B"]

    asyncio.run(run())
