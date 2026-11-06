"""Generic schema types for PLP guide-produced program specifications."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field


class SampleSite(BaseModel):
    """Directed proposal node: an LLM call whose output is a sample site.

    ``prompt_template`` may contain ``{question}`` and ``{<parent_name>}``
    placeholders. Parents must be declared in ``parents`` to induce directed
    edges in the compiled PLP graph.
    """

    name: str
    prompt_template: str
    parents: list[str] = Field(default_factory=list)


class DeterministicFactor(BaseModel):
    """Deterministic scoring factor: sandboxed Python log-weight over parents."""

    kind: Literal["python"] = "python"
    name: str
    parents: list[str]
    code: str


class JudgeFactor(BaseModel):
    """LLM-as-judge scoring factor.

    ``prompt_template`` may use ``{question}`` and ``{<parent_name>}``. The
    judge LLM is expected to reply with one scalar log-weight.
    """

    kind: Literal["judge"] = "judge"
    name: str
    parents: list[str]
    prompt_template: str


FactorSpec = Union[DeterministicFactor, JudgeFactor]


class ProgramSpec(BaseModel):
    """Generic chain-graph program spec produced by an amortized guide."""

    sites: list[SampleSite]
    factors: list[FactorSpec] = Field(default_factory=list)
    output_site: str
