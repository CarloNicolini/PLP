"""Pure structural tests for ChainGraph — no LLM calls."""

from __future__ import annotations

from plp.inference.graph import ChainGraph, FactorSite, SampleSite


def _site(name: str, *parents: str) -> SampleSite:
    return SampleSite(name=name, fn=lambda state: name, parents=tuple(parents))


def _factor(name: str, *parents: str) -> FactorSite:
    return FactorSite(name=name, fn=lambda state, *args: 0.0, parents=tuple(parents))


class TestChainGraphConstruction:
    def test_empty_graph(self) -> None:
        graph = ChainGraph()
        assert graph.topological_order() == []
        assert graph.moralised_adjacency() == {}
        assert not graph.has_cycle_in_parents()
        assert not graph.has_cycle_in_moralised_graph()

    def test_single_site_no_factors(self) -> None:
        graph = ChainGraph(sites=[_site("x")])
        assert graph.topological_order() == ["x"]
        assert not graph.has_cycle_in_parents()
        assert not graph.has_cycle_in_moralised_graph()

    def test_list_accepts_name_mismatch_with_dict_key_gone(self) -> None:
        """No validation: keying is by ``Site.name``, not list index."""
        g = ChainGraph(sites=[SampleSite(name="y", fn=lambda s: "y")])
        assert g.site_names == ["y"]
        assert g.sites_by_name["y"].name == "y"


class TestTopologicalOrderCaseA:
    """Case A: directed Bayes net (acyclic parents)."""

    def test_chain_three_nodes(self) -> None:
        graph = ChainGraph(
            sites=[_site("x"), _site("y", "x"), _site("z", "y")],
        )
        order = graph.topological_order()
        assert order is not None
        assert order.index("x") < order.index("y") < order.index("z")
        assert not graph.has_cycle_in_parents()

    def test_tree(self) -> None:
        graph = ChainGraph(
            sites=[_site("root"), _site("l", "root"), _site("r", "root")],
        )
        order = graph.topological_order()
        assert order is not None
        assert order[0] == "root"

    def test_diamond(self) -> None:
        graph = ChainGraph(
            sites=[
                _site("a"),
                _site("b", "a"),
                _site("c", "a"),
                _site("d", "b", "c"),
            ],
        )
        order = graph.topological_order()
        assert order is not None
        assert order.index("a") < order.index("b") < order.index("d")
        assert order.index("a") < order.index("c") < order.index("d")


class TestCyclicParentsCaseB:
    """Case B: cyclic parents => pi is an MRF, not a Bayes net."""

    def test_two_site_cycle(self) -> None:
        graph = ChainGraph(sites=[_site("x", "y"), _site("y", "x")])
        assert graph.topological_order() is None
        assert graph.has_cycle_in_parents()

    def test_self_loop(self) -> None:
        graph = ChainGraph(sites=[_site("x", "x")])
        assert graph.has_cycle_in_parents()

    def test_three_site_cycle(self) -> None:
        graph = ChainGraph(
            sites=[_site("a", "c"), _site("b", "a"), _site("c", "b")],
        )
        assert graph.has_cycle_in_parents()


class TestMoralisedGraphAndFactorCliques:
    """Case C: factor cliques induce undirected coupling on the posterior."""

    def test_pairwise_factor_couples_two_sites(self) -> None:
        graph = ChainGraph(
            sites=[_site("x"), _site("y")],
            factors=[_factor("phi", "x", "y")],
        )
        adj = graph.moralised_adjacency()
        assert adj["x"] == {"y"}
        assert adj["y"] == {"x"}
        assert not graph.has_cycle_in_moralised_graph()

    def test_triangle_from_three_pairwise_factors(self) -> None:
        graph = ChainGraph(
            sites=[_site("x"), _site("y"), _site("z")],
            factors=[
                _factor("xy", "x", "y"),
                _factor("yz", "y", "z"),
                _factor("zx", "z", "x"),
            ],
        )
        assert graph.has_cycle_in_moralised_graph()

    def test_higher_order_factor_makes_clique(self) -> None:
        graph = ChainGraph(
            sites=[_site("a"), _site("b"), _site("c")],
            factors=[_factor("clique", "a", "b", "c")],
        )
        adj = graph.moralised_adjacency()
        assert adj["a"] == {"b", "c"}
        assert adj["b"] == {"a", "c"}
        assert adj["c"] == {"a", "b"}
        assert graph.has_cycle_in_moralised_graph()

    def test_moralisation_adds_edge_between_coparents(self) -> None:
        graph = ChainGraph(
            sites=[_site("a"), _site("b"), _site("c", "a", "b")],
        )
        adj = graph.moralised_adjacency()
        assert "b" in adj["a"]
        assert "a" in adj["b"]


class TestFactorsOn:
    def test_factors_on_returns_only_attached(self) -> None:
        f_xy = _factor("f_xy", "x", "y")
        f_yz = _factor("f_yz", "y", "z")
        graph = ChainGraph(
            sites=[_site("x"), _site("y"), _site("z")],
            factors=[f_xy, f_yz],
        )
        assert graph.factors_on("x") == [f_xy]
        assert graph.factors_on("y") == [f_xy, f_yz]
        assert graph.factors_on("z") == [f_yz]
