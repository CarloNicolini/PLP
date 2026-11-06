"""Tests for choose_engine — pure topology dispatch, no LLMs."""

from __future__ import annotations

from plp.inference.engines.forward import Forward
from plp.inference.engines.importance import ImportanceSampling
from plp.inference.engines.lbp import LoopyParticleBP
from plp.inference.engines.max_product import MaxProduct
from plp.inference.engines.smc import SMC
from plp.inference.graph import ChainGraph, FactorSite, SampleSite
from plp.inference.engines.select import choose_engine


def _site(name: str, *parents: str) -> SampleSite:
    return SampleSite(name=name, fn=lambda state: name, parents=tuple(parents))


def _factor(name: str, *parents: str) -> FactorSite:
    return FactorSite(name=name, fn=lambda state, *args: 0.0, parents=tuple(parents))


class TestCaseA:
    """Directed acyclic parents, no factors -> Forward."""

    def test_single_site(self) -> None:
        g = ChainGraph(sites=[_site("x")])
        assert choose_engine(g) is Forward

    def test_chain_no_factors(self) -> None:
        g = ChainGraph(
            sites=[_site("a"), _site("b", "a"), _site("c", "b")],
        )
        assert choose_engine(g) is Forward


class TestCaseB:
    """Cyclic parents -> LoopyParticleBP (Gibbs on implicit MRF)."""

    def test_two_site_cycle(self) -> None:
        g = ChainGraph(sites=[_site("x", "y"), _site("y", "x")])
        assert choose_engine(g) is LoopyParticleBP

    def test_cycle_overrides_factor_analysis(self) -> None:
        """Even with no factors, a cyclic parent graph requires LBP."""
        g = ChainGraph(sites=[_site("x", "y"), _site("y", "x")])
        # No factors present.
        assert len(g.factors) == 0
        assert choose_engine(g) is LoopyParticleBP


class TestCaseC:
    """Directed parents plus factor cliques."""

    def test_dag_with_one_unary_factor_is_importance_sampling(self) -> None:
        g = ChainGraph(
            sites=[_site("x")],
            factors=[_factor("phi", "x")],
        )
        assert choose_engine(g) is ImportanceSampling

    def test_chain_with_per_site_factors_is_smc(self) -> None:
        g = ChainGraph(
            sites=[_site("a"), _site("b", "a"), _site("c", "b")],
            factors=[
                _factor("fa", "a"),
                _factor("fb", "b"),
                _factor("fc", "c"),
            ],
        )
        assert choose_engine(g) is SMC

    def test_triangle_of_factors_is_lbp(self) -> None:
        g = ChainGraph(
            sites=[_site("x"), _site("y"), _site("z")],
            factors=[
                _factor("xy", "x", "y"),
                _factor("yz", "y", "z"),
                _factor("zx", "z", "x"),
            ],
        )
        assert choose_engine(g) is LoopyParticleBP

    def test_pairwise_factor_on_two_sites_is_importance_sampling(self) -> None:
        """Two nodes, one pairwise factor: tree, so IS is appropriate."""
        g = ChainGraph(
            sites=[_site("x"), _site("y")],
            factors=[_factor("phi", "x", "y")],
        )
        assert choose_engine(g) is ImportanceSampling


class TestPreferMap:
    def test_overrides_all_other_rules(self) -> None:
        g = ChainGraph(
            sites=[_site("x"), _site("y", "x")],
            factors=[_factor("phi", "y")],
        )
        assert choose_engine(g, prefer_map=True) is MaxProduct
