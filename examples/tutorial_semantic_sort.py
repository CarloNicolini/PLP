"""
PLP Tutorial: Semantic Sorting

Sort items by a semantic property that only an LLM can evaluate.
Demonstrates the new high-level PLP API.
"""

import plp
from dotenv import load_dotenv

load_dotenv()

# Configure once
plp.configure(
    model="gpt-4.1-mini",
    temperature=0.5,
    system_prompt="You are a helpful assistant that compares items.",
)

# Items to sort by "healthiness"
foods = ["pizza", "kale", "chocolate", "salmon", "chips"]


# Method 1: Direct SPRT voting with LLM judge
def healthier(a: str, b: str) -> str:
    # Using the simplified sample API
    response = plp.sample_fn(f"Which is healthier: {a} or {b}? Reply ONLY 'a' or 'b'.")
    return "a" if "a" in response.lower() else "b" if "b" in response.lower() else "tie"


# Sort using comparison-based sort with SPRT voting
def llm_sort(items: list[str], compare_fn) -> list[str]:
    """Sort items using LLM-based comparison."""
    if len(items) <= 1:
        return items

    # Simple bubble sort with LLM comparisons
    result = list(items)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            print(f"Comparing {result[i]} vs {result[j]}...")
            # Use SPRT to robustly determine ordering
            # We use the new vote() shortcut
            winner = plp.vote(
                [result[i], result[j]],
                judge=compare_fn,
                k=3,  # Wait for 3-vote margin
            )
            print(f"Winner: {winner}")

            # If j is healthier (winner), it should come first
            # (assuming descending order of healthiness)
            if winner == result[j]:
                result[i], result[j] = result[j], result[i]

    return result


# Method 2: Using run_map for single best ranking (demonstrating sampler decorator)
@plp.sampler(name="rank_foods")
def rank_by_health(foods: list[str]) -> str:
    return f"""Rank these foods from healthiest to least healthy:
{", ".join(foods)}

Reply with just the foods in order, separated by commas."""


# Run it!
if __name__ == "__main__":
    print("=== Method 1: SPRT voting sort ===")
    sorted_foods = llm_sort(foods, healthier)
    print(f"Healthiest to least: {sorted_foods}")

    print("\n=== Method 2: Direct ranking with MAP ===")
    # Run multiple times and pick best (simulated MAP)
    # Using the builder pattern
    rankings = (
        plp.infer().with_runs(3).map(lambda: rank_by_health(foods), inputs=[None])
    )
    print(f"Rankings: {rankings}")
