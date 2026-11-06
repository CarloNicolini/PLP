# Example: From Zero To Multi-Site Inference

A complex PLP workflow should be built in layers. The following pattern scales from a toy scripted run to a hosted multi-agent system.

## Step 1: Propose

Create one or more `sample` sites that produce candidate strings. Store the parts later sites will read in `state.vars`.

## Step 2: Critique

Add another site whose `parents` name the proposal it reads. This turns a flat sample into a directed graph.

## Step 3: Score

Add `factor` functions for checks that are meaningful in your domain: exact symbolic checks, rubric scores, consistency tests, or model judges.

## Step 4: Build The Graph

Use `ChainGraph.from_decorated(...)` and inspect the selected engine. Treat the engine choice as feedback about your declared structure.

## Step 5: Run Particles

Increase particle count only after the proposal can reach good traces. Watch for weight collapse, repeated answers, and judge drift.

## Step 6: Diagnose

When results look wrong, ask which failure family is active:

- support: the model never proposes the right trace;
- selection: the factor rewards the wrong trace;
- dependence: candidates are too correlated;
- composition: the graph omits a dependency.

This is the main benefit of PLP over an ad hoc scaffold: the debugging questions follow from the semantics.
