"""
MAKER Algorithm: Minimal PLP Implementation.

Demonstrates the MAKER algorithm using PLP primitives:
- sample: LLM generation for decomposer and solver
- plate_map: Parallel sampling for SPRT voting (2k-1 initial samples)
- compute_k_for_eps: Calculate vote margin for target reliability

Example: Multiply two large numbers using pure LLM reasoning.
"""

import re
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

from plp import ChatOpenAI, PLPState, sample, plate_map, compute_k_for_eps


# ============================================================
# The Three MAKER Prompts
# ============================================================

DECOMPOSER = """Break this problem into simple sequential steps where each step uses the result from the prior step.
Format: 1. [step description] 2. [step description] ...
IMPORTANT: Each step must produce a number that the next step uses. The final step produces the answer.
Example for 12 × 34:
1. Compute 12 × 4 = 48
2. Compute 12 × 30 = 360  
3. Add 48 + 360 = 408
Note: Step 3 uses results from step 1 and 2, which violates the rule.
Better decomposition where each step only uses the prior result:
1. Compute 12 × 4 to get partial product
2. Add (prior result) to (12 × 30) to get final answer"""

SOLVER = """Solve this step. Show your reasoning, then give your final answer after 'vote:'."""

DISCRIMINATOR = (
    """Which solution is correct? Review both, then vote (1 or 2) after 'vote:'."""
)


# ============================================================
# MAKER with PLP Primitives
# ============================================================


def make_maker(llm: ChatOpenAI):
    """Create MAKER components using PLP sample sites."""

    @sample(llm, name="decompose")
    def decompose(state: PLPState, problem: str) -> str:
        return f"{DECOMPOSER}\n\nProblem: {problem}"

    @sample(llm, name="solve")
    def solve(state: PLPState, step: str, prior: str) -> str:
        ctx = f"Prior result: {prior}\n\n" if prior else ""
        return f"{SOLVER}\n\n{ctx}Step: {step}"

    @sample(llm, name="discriminate")
    def discriminate(state: PLPState, step: str, a: str, b: str) -> str:
        return f"{DISCRIMINATOR}\n\nStep: {step}\n1. {a}\n2. {b}"

    return decompose, solve, discriminate


def extract_vote(response: str) -> str:
    """Extract answer after 'vote:' from response."""
    match = re.search(r"vote:\s*(.+)", response, re.IGNORECASE)
    return match.group(1).strip() if match else response.strip().split("\n")[-1]


def sprt_sample(
    solve_fn,
    state: PLPState,
    step: str,
    prior: str,
    k: int,
    max_samples: int = 50,
) -> str:
    """
    SPRT voting using plate_map for parallel initial sampling.

    1. Use plate_map to generate 2k-1 initial samples in parallel
    2. Count occurrences, check if winner has k-margin
    3. If not, continue sampling until margin reached
    """
    initial = 2 * k - 1
    counts: Counter = Counter()

    # Phase 1: Parallel initial sampling with plate_map
    def sample_one(s: PLPState) -> tuple[PLPState, str]:
        return solve_fn(s, step, prior)

    _, responses = plate_map("sprt_initial", initial, sample_one, state)

    for r in responses:
        counts[extract_vote(r)] += 1

    # Check if we have a winner
    top = counts.most_common(2)
    if len(top) == 1 or (top[0][1] - top[1][1] >= k):
        return top[0][0]

    # Phase 2: Continue sampling until margin reached
    for _ in range(max_samples - initial):
        _, r = solve_fn(state, step, prior)
        counts[extract_vote(r)] += 1

        top = counts.most_common(2)
        if top[0][1] - (top[1][1] if len(top) > 1 else 0) >= k:
            return top[0][0]

    return counts.most_common(1)[0][0]


def parse_steps(response: str) -> list[str]:
    """Parse numbered steps from decomposer response."""
    return [
        m.group(1).strip()
        for m in re.finditer(r"\d+\.\s*(.+?)(?=\d+\.|$)", response, re.DOTALL)
    ]


def maker_solve(
    llm: ChatOpenAI, problem: str, target_reliability: float = 0.95, p: float = 0.7
) -> str:
    """
    Solve a problem using MAKER algorithm.

    Args:
        llm: ChatOpenAI instance
        problem: Natural language problem
        target_reliability: Target overall success probability
        p: Estimated per-step accuracy
    """
    decompose, solve, _ = make_maker(llm)
    state = PLPState()

    # Step 1: Decompose
    state, raw_steps = decompose(state, problem)
    steps = parse_steps(raw_steps)

    if not steps:
        steps = [problem]  # Fallback: solve directly

    print(f"Problem: {problem}")
    print(f"Decomposed into {len(steps)} steps")

    # Calculate k for target reliability (minimum k=8 for robustness)
    per_step_eps = 1 - target_reliability ** (1 / max(len(steps), 1))
    k = max(8, compute_k_for_eps(p, per_step_eps))
    print(f"SPRT k={k}, initial_samples={2 * k - 1}\n")

    # Step 2: Solve each step with SPRT voting
    result = ""
    for i, step in enumerate(steps, 1):
        winner = sprt_sample(solve, state, step, result, k)
        print(f"Step {i}: {step[:40]}... → {winner}")
        result = winner

    print(f"\nFinal: {result}")
    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # Single test: 6-digit × 6-digit multiplication
    problem = "What is 847293 × 159468?"
    expected = 847293 * 159468  # 135099aborators240524

    result = maker_solve(llm, problem)

    # Extract number from result
    numbers = re.findall(r"\d+", result.replace(",", ""))
    computed = int(numbers[-1]) if numbers else 0

    print(
        f"\n{'✓' if computed == expected else '✗'} Result: {computed}, Expected: {expected}"
    )
