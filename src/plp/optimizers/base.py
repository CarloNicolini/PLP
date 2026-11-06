"""Generic amortized guide interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from plp.client import PLPChatClient

from .objectives import ObjectiveProfile

SpecT = TypeVar("SpecT", bound=BaseModel)
SummaryT = TypeVar("SummaryT")


class SpecFamily(Protocol, Generic[SpecT, SummaryT]):
    """Hot-pluggable schema family for amortized guides."""

    response_format: type[SpecT]

    def system_prompt(self) -> str:
        """Return the family-level system prompt."""

    def render_user_prompt(
        self,
        history: list[tuple[SpecT, SummaryT]],
        question: str,
    ) -> str:
        """Render the user prompt for a proposal call."""

    def objective_profile(self) -> ObjectiveProfile:
        """Describe objective/signal assumptions for this family."""


@dataclass
class AmortizedGuide(Generic[SpecT, SummaryT]):
    """Generic amortized guide over a schema/spec family.

    The guide proposes a structured spec ``s`` from
    ``q_psi(s | x, history)``. Concretely, this is implemented via a
    structured LLM call whose response format is supplied by the plugged-in
    spec family.
    """

    client: PLPChatClient
    family: SpecFamily[SpecT, SummaryT]

    def objective_profile(self) -> ObjectiveProfile:
        """Expose objective assumptions declared by the spec family."""
        return self.family.objective_profile()

    def propose(
        self,
        history: list[tuple[SpecT, SummaryT]],
        question: str,
    ) -> SpecT:
        """Propose a structured spec conditioned on question and history."""
        user = self.family.render_user_prompt(history, question)
        return self.client.invoke_parsed(
            [
                {"role": "system", "content": self.family.system_prompt()},
                {"role": "user", "content": user},
            ],
            response_format=self.family.response_format,
        )


# Backward-compatible names from the first refactor.
AmortizedInferer = AmortizedGuide
ProposalFamily = SpecFamily
