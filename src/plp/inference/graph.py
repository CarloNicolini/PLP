"""ChainGraph: ordered sample sites, factors, and structural queries (no I/O)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Site:
    """Sample or factor site (decorated callable metadata).

    ``parents`` are names of sample sites read from ``state.vars``.
    ``eq`` matches by ``name`` only so ``Callable`` need not be hashable.
    """

    name: str
    fn: Callable[..., Any]
    parents: tuple[str, ...] = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Site) and self.name == other.name


SampleSite = Site
FactorSite = Site


@dataclass
class ChainGraph:
    """Chain graph: directed sample-parent graph + undirected factor cliques."""

    sites: list[SampleSite] = field(default_factory=list)
    factors: list[FactorSite] = field(default_factory=list)
    output_site: str | None = None

    @property
    def sites_by_name(self) -> dict[str, SampleSite]:
        return {s.name: s for s in self.sites}

    @property
    def site_names(self) -> list[str]:
        return list(dict.fromkeys(s.name for s in self.sites))

    def sample_parent_adjacency(self) -> dict[str, set[str]]:
        """Directed edges: ``succ[parent]`` contains children (sample sites only)."""
        names = self.site_names
        vset = set(names)
        succ = {n: set() for n in names}
        for site in self.sites:
            for parent in site.parents:
                if parent in vset:
                    succ[parent].add(site.name)
        return succ

    def moralised_adjacency(self) -> dict[str, set[str]]:
        """Undirected moralised graph (sample skeleton, moralisation, factor cliques)."""
        adj: dict[str, set[str]] = defaultdict(set)
        for n in self.site_names:
            _ = adj[n]

        def edge(a: str, b: str) -> None:
            if a != b:
                adj[a].add(b)
                adj[b].add(a)

        for site in self.sites:
            for parent in site.parents:
                edge(parent, site.name)
            ps = list(site.parents)
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    edge(ps[i], ps[j])
        for factor in self.factors:
            ps = list(factor.parents)
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    edge(ps[i], ps[j])
        return dict(adj)

    def topological_order(self) -> list[str] | None:
        succ = self.sample_parent_adjacency()
        names = self.site_names
        indeg = {n: 0 for n in names}
        for targets in succ.values():
            for t in targets:
                indeg[t] += 1
        queue = [n for n in names if indeg[n] == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for t in succ[n]:
                indeg[t] -= 1
                if indeg[t] == 0:
                    queue.append(t)
        return order if len(order) == len(names) else None

    def has_cycle_in_parents(self) -> bool:
        return self.topological_order() is None

    def has_cycle_in_moralised_graph(self) -> bool:
        adj = self.moralised_adjacency()
        visited: set[str] = set()

        def dfs(node: str, parent: str | None) -> bool:
            visited.add(node)
            for neighbour in adj.get(node, set()):
                if neighbour == parent:
                    continue
                if neighbour in visited:
                    return True
                if dfs(neighbour, node):
                    return True
            return False

        for node in adj:
            if node not in visited and dfs(node, None):
                return True
        return False

    def factors_on(self, site_name: str) -> list[FactorSite]:
        return [f for f in self.factors if site_name in f.parents]

    @classmethod
    def from_decorated(
        cls,
        *decorated: Callable[..., Any],
        output_site: str | None = None,
    ) -> ChainGraph:
        sites: list[SampleSite] = []
        factors: list[FactorSite] = []
        for fn in decorated:
            kind = getattr(fn, "__plp_kind__", None)
            site_name = getattr(fn, "__plp_site_name__", None)
            parents = getattr(fn, "__plp_parents__", ())
            if kind is None or site_name is None:
                raise ValueError(
                    f"{fn!r} is not a PLP-decorated callable "
                    f"(no __plp_kind__ / __plp_site_name__)."
                )
            if kind in {"sample", "asample"}:
                sites.append(SampleSite(name=site_name, fn=fn, parents=parents))
            elif kind == "factor":
                factors.append(FactorSite(name=site_name, fn=fn, parents=parents))
            else:
                raise ValueError(f"unknown __plp_kind__ {kind!r} on {fn!r}")
        return cls(sites=sites, factors=factors, output_site=output_site)
