"""``parents`` metadata and ``ChainGraph.from_decorated``."""

from __future__ import annotations

import pytest

from plp import PLPChatClient, PLPState, asample, factor, sample
from plp.inference.graph import ChainGraph


def test_decorator_parent_metadata(llm: PLPChatClient) -> None:
    @sample(llm, name="x")
    def site_x(state: PLPState) -> str:
        return "prompt"

    @sample(llm, name="y", parents=["x"])
    def site_y(state: PLPState) -> str:
        return "prompt"

    @asample(llm, name="z", parents=["y"])
    def site_z(state: PLPState) -> str:
        return "prompt"

    @factor(name="phi", parents=["a", "b"])
    def phi(state: PLPState, a: str, b: str) -> float:
        return 0.0

    @factor(name="auto")
    def auto(state: PLPState, foo: str, bar: str) -> float:
        return 0.0

    assert site_x.__plp_parents__ == ()
    assert site_y.__plp_parents__ == ("x",)
    assert site_z.__plp_parents__ == ("y",)
    assert phi.__plp_parents__ == ("a", "b")
    assert auto.__plp_parents__ == ("foo", "bar")


def test_chain_graph_from_decorated(llm: PLPChatClient) -> None:
    @asample(llm, name="x")
    def site_x(state: PLPState) -> str:
        return "px"

    @asample(llm, name="y", parents=["x"])
    def site_y(state: PLPState) -> str:
        return "py"

    @factor(name="phi", parents=["x", "y"])
    def phi(state: PLPState, x: str, y: str) -> float:
        return 1.0

    g = ChainGraph.from_decorated(site_x, site_y, phi, output_site="y")
    assert {s.name for s in g.sites} == {"x", "y"}
    assert g.sites_by_name["y"].parents == ("x",)
    assert len(g.factors) == 1
    assert g.output_site == "y"


def test_cyclic_parents_allowed_in_graph(llm: PLPChatClient) -> None:
    @asample(llm, name="x", parents=["y"])
    def site_x(state: PLPState) -> str:
        return "x"

    @asample(llm, name="y", parents=["x"])
    def site_y(state: PLPState) -> str:
        return "y"

    g = ChainGraph.from_decorated(site_x, site_y)
    assert g.topological_order() is None
    assert g.has_cycle_in_parents()


def test_non_decorated_raises() -> None:
    def plain(state: PLPState) -> str:
        return "nope"

    with pytest.raises(ValueError, match="not a PLP-decorated"):
        ChainGraph.from_decorated(plain)
