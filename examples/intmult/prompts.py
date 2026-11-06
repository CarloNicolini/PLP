from __future__ import annotations

from jinja2 import Template

DECOMPOSE_PROMPT = """
You are an arithmetic planner. 
Your task is not to directly multiply two integers, but to decompose the multiplication into a sequence of 
simpler arithmetic subtasks that can be solved independently.

## Goal

Given the product of two integers A * B, produce a structured decomposition that:
- Breaks numbers into convenient components (place values, factors, or chunks)
- Reduces the multiplication into simpler multiplications
- Uses only elementary arithmetic in subtasks (single-digit products, additions, shifts)
- Outputs the dependency order between subtasks
- Avoids computing the final result unless explicitly requested

## Rules
1.	Prefer base-10 decomposition (e.g., 347 = 300 + 40 + 7).
2.	Use distributive law to split the multiplication.
3.	Each subtask must be independently solvable.
4.	Label subtasks T1, T2, ...
5.	Final aggregation should be a separate task.
6.	Keep subtasks small enough for reliable LLM execution.

## Example 1
Problem: 347 × 58

Strategy: Decompose both numbers into decimal place values
347 = 300 + 40 + 7
58 = 50 + 8

### Strategy solution:

```json
{
"subtasks":
    {"T1": "300 * 50"},
    {"T2": "300 * 8"},
    {"T3": "40 * 50"},
    {"T4": "40 * 8"},
    {"T5": "7 * 50"},
    {"T6": "7 * 8"}
"aggregation":
    "T1 + T2 + T3 + T4 + T5 + T6"
}
```

Do not compute intermediate values unless explicitly requested.

# Your Task

Given an input multiplication:

{{ problem }}

### Strategy solution:
"""

ANSWER_PROMPT = """
Multiply the two integers exactly.
Expression: {{ problem.expression }}
Return only the base-10 integer product."
Do not include commas, words, equations, or explanations.
"""

ABSTAIN_PROMPT = """
Multiply the two integers exactly.
Expression: {{ problem.expression }}
Do not answer immediately, instead try to gather more ideas to solve the problem.
A good strategy is decimal decomposition or partial products
You may write a short scratchpad.
Reasoning:
    <put scratchpad here>
On the last line write exactly: answer: <integer>
"""

PLAN_PROMPT = """
Present a plan that you would follow to decompose the problem of 
multiplying two integers using their base-10 decomposition, then execute the plan internally.
"""

decompose_template = Template(source=DECOMPOSE_PROMPT)
answer_template = Template(source=ANSWER_PROMPT)
abstain_template = Template(source=ABSTAIN_PROMPT)
