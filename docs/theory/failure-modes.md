# Limits And Failure Modes

PLP is not a truth machine. It is a framework for making uncertainty and structure explicit.

## No Off-Policy Density By Default

Most LLM APIs do not expose enough information to evaluate arbitrary trace probabilities. PLP therefore emphasizes sampling and reweighting under the same deployment.

## Loopy Inference Can Be Fragile

`LoopyParticleBP` is useful when the graph is cyclic, but loopy message passing is approximate. It can overcount evidence, converge slowly, or settle into poor fixed points.

## Judges Drift

A factor can be wrong. If the judge is stochastic, poorly calibrated, or vulnerable to style artifacts, high-weight traces may be high-scoring for the wrong reason.

## Composition Is A Scientific Claim

Every parent declaration asserts a conditional dependence structure. If that structure is wrong, the chosen engine and reported confidence can mislead you.

The benefit of PLP is that these issues become inspectable parts of the program instead of hidden behavior inside a prompt chain.
