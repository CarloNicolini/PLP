# Scientific Context

PLP is best read as probabilistic programming over strings.

Classical probabilistic programming gives names to random variables, observations, plates, and inference algorithms. LLM workflows need the same discipline, but the values are often natural-language strings and the kernels are sample-only model calls.

## Prompting Patterns As Graphs

| Prompting pattern | PLP reading |
| --- | --- |
| Chain of thought | Latent reasoning string followed by an answer site. |
| Self-consistency | Plate over candidate traces plus selection. |
| Best-of-K | Multiple proposals scored by a factor or max-product objective. |
| Debate | Multiple sites coupled by critique and judge factors. |
| Repair loop | Cyclic or sequential dependence with feedback factors. |

This graph view lets PLP choose algorithms rather than treating every scaffold as a one-off trick.

## What Is New Here

The unusual part is not `sample` or `factor`; probabilistic programming already has those. The unusual part is applying them to implicit, string-valued LLM kernels and keeping the trace available for diagnosis.
