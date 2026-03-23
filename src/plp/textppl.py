"""
Text-first PLP primitives.

This module is PLP's TextGrad-compatible surface: it wraps the textual
gradient/backprop metaphor from *TextGrad: Automatic "Differentiation" via
Text* (Yuksekgonul et al., 2024) in probabilistic-programming semantics. Each
primitive records a TraceEvent, updates log_weight when appropriate, and can be
mixed with observe/factor/plate handlers.

Provided building blocks:

- text_sample: SAMPLE primitive over strings (forward node values)
- text_score: SCORE / factor primitive over traces (textual loss terms)
- text_observe: OBSERVE (hard 0/1 constraint)
- length_penalty: simple score for over-long outputs
- text_transition: TRANSITION / edit step for candidate refinement
- textgrad_step: TextGrad-style gradient step over text
- text_factor / text_overlap / text_kernel_constraint: judge and kernel
  adapters that turn LLM critiques into log-scores or repulsion forces

Use these when you want the user-facing ergonomics of TextGrad but need the
traceability, conditioning, or inference orchestration provided by PLP. Every
primitive operates on a PLPState and appends a TraceEvent so subsequent kernels
can replay, relabel, or condition on the full textual history.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import inspect
import re
from typing import Any, Callable, Sequence

from ._llm import LLMLike, llm_name
from .runtime import run_prompt
from .state import PLPState, TraceEvent


def text_sample(
    state: PLPState,
    llm: LLMLike,
    prompt: str,
    name: str = "text_sample",
    system_prompt: str | None = None,
) -> tuple[PLPState, str]:
    """
    SAMPLE primitive over text.

    Args:
        state: Current PLP state
        llm: Object with an .invoke(prompt, system_prompt) -> str method
        prompt: Prompt string sent to the LLM
        name: Site name for tracing
        system_prompt: Optional system-level instruction

    Returns:
        (new_state, output_text)
    """
    new_state, output = run_prompt(
        llm,
        state,
        name=name,
        prompt=prompt,
        system_prompt=system_prompt,
        meta={"kind": "sample"},
    )
    return new_state, output


def text_score(state: PLPState, log_score: float, name: str = "text_score") -> PLPState:
    """
    SCORE primitive: add a log-score to the trace.

    Args:
        state: Current PLP state
        log_score: Log-score to add (can be positive or negative)
        name: Site name for tracing

    Returns:
        New PLPState with updated log_weight and trace.
    """
    new_state = state.copy()
    new_state.log_weight += log_score

    event = TraceEvent(
        primitive=name,
        prompt="(text_score)",
        output=str(log_score),
        meta={"kind": "score", "log_score": log_score},
    )
    new_state.trace.append(event)
    return new_state


def text_observe(state: PLPState, ok: bool, name: str = "text_observe") -> PLPState:
    """
    OBSERVE primitive: hard 0/1 constraint on the trace.

    If ok is False, the trace is marked invalid by setting log_weight = -inf.
    """
    new_state = state.copy()
    if not ok:
        new_state.log_weight = float("-inf")

    event = TraceEvent(
        primitive=name,
        prompt="(text_observe)",
        output=str(ok),
        meta={"kind": "observe", "passed": ok},
    )
    new_state.trace.append(event)
    return new_state


def length_penalty(
    state: PLPState,
    text: str,
    max_length: int,
    alpha: float = 0.01,
    name: str = "length_penalty",
) -> PLPState:
    """
    Simple SCORE primitive that penalizes overly long outputs.

    log_score = -alpha * max(0, len(text) - max_length)

    Useful as a generic prior against verbosity when a task is simple.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be numeric, got {type(alpha)}")

    excess = max(0, len(text) - max_length)
    log_score = -alpha * excess
    return text_score(state, log_score, name=name)


def text_transition(
    state: PLPState,
    llm: LLMLike,
    current: str,
    instruction: str,
    name: str = "text_transition",
    system_prompt: str | None = None,
) -> tuple[PLPState, str]:
    """
    TRANSITION primitive: propose an edited/refined candidate.

    The default prompt format is:

        {instruction}

        Current candidate:
        {current}

        Revised candidate:

    Args:
        state: Current PLP state
        llm: LLM-like object with invoke()
        current: Current candidate text
        instruction: Natural-language instructions for the edit
        name: Site name for tracing
        system_prompt: Optional system prompt for the LLM
    """
    prompt = (
        f"{instruction.strip()}\n\nCurrent candidate:\n{current}\n\nRevised candidate:"
    )
    new_state, output = run_prompt(
        llm,
        state,
        name=name,
        prompt=prompt,
        system_prompt=system_prompt,
        meta={"kind": "transition"},
    )
    return new_state, output


def textgrad_step(
    state: PLPState,
    llm: LLMLike,
    current: str,
    loss_description: str,
    name: str = "textgrad_step",
    system_prompt: str | None = None,
) -> tuple[PLPState, str]:
    """
    TextGrad-style gradient step over text.

    Mirrors the textual gradient update described in *TextGrad: Automatic
    "Differentiation" via Text* (§2). `current` acts as the node value, the
    `loss_description` is the upstream textual gradient/backprop signal, and
    the LLM plays the role of the optimizer that proposes the next iterate.
    By routing the call through `run_prompt` we log the entire step (prompt,
    textual gradient, edited output) inside PLP's trace so it can mix with
    other primitives or inference kernels.

    This is a generic interface: given a current text and a natural-language
    loss description, ask an LLM to propose a small edit that most improves
    the loss. It behaves like a gradient step in the space of strings.

    Args:
        state: Current PLP state
        llm: LLM-like object with invoke()
        current: Current candidate text
        loss_description: Natural-language description of what should improve
        name: Site name for tracing
        system_prompt: Optional system prompt

    Returns:
        (new_state, edited_text)
    """
    prompt = (
        "You are a 'textual gradient' oracle for text, part of a larger optimization system.\n"
        "You receive a current output as text and a description of a loss function.\n"
        "Return a *small* edited version of the text that is aimed to maximally reduce the loss.\n\n"
        f"Current output:\n{current}\n\n"
        f"Loss description:\n{loss_description}\n\n"
        "Edited output (no commentary, just the revised text):"
    )
    new_state, output = run_prompt(
        llm,
        state,
        name=name,
        prompt=prompt,
        system_prompt=system_prompt,
        meta={"kind": "textgrad"},
    )
    return new_state, output


@dataclass
class LLMJudgeResult:
    """Structured result from a text_factor / judge call."""

    score: float
    feedback: str
    raw: str


def _parse_score_feedback(text: str) -> LLMJudgeResult:
    score_match = re.search(r"Score:\s*([+-]?\d+(?:\.\d+)?)", text)
    score = float(score_match.group(1)) if score_match else 60.0
    score = max(0.0, min(score, 100.0))
    feedback_match = re.search(r"Feedback:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    feedback = (
        feedback_match.group(1).strip() if feedback_match else "Clarify the reasoning."
    )
    feedback = re.sub(r"\s+", " ", feedback)
    return LLMJudgeResult(score=score, feedback=feedback, raw=text.strip())


def text_factor(
    state: PLPState,
    llm: LLMLike,
    candidate: str,
    instructions: str,
    *,
    task: str | None = None,
    name: str = "text_factor",
    score_to_log: Callable[[float], float] | None = None,
    system_prompt: str | None = None,
) -> tuple[PLPState, LLMJudgeResult]:
    """
    Attach a soft factor by asking an LLM judge for a score/feedback pair.

    Args:
        state: Current PLP state.
        llm: LLM client used to evaluate the candidate.
        candidate: Text being judged.
        instructions: Natural-language description of the judging criteria.
        task: Optional task context inserted before the candidate.
        name: Trace prefix; prompt is logged under this name and the resulting
            log-score under f"{name}.score".
        score_to_log: Optional mapping score -> log_weight increment. Defaults
            to a centered linear mapping around 75.
        system_prompt: Optional system-level instruction for the judge.

    Returns:
        (new_state, LLMJudgeResult)
    """
    prompt = (
        f"{instructions.strip()}\n"
        "Return exactly:\n"
        "Score: <integer 0-100>\n"
        "Feedback: <<=40 words summarizing improvements>\n\n"
    )
    if task:
        prompt += f"Task:\n{task.strip()}\n\n"
    prompt += f"Candidate:\n{candidate}\n"

    new_state, raw = run_prompt(
        llm,
        state,
        name=name,
        prompt=prompt,
        system_prompt=system_prompt,
        meta={"kind": "text_factor"},
    )
    result = _parse_score_feedback(raw)

    def _default_mapping(score: float) -> float:
        return (score - 75.0) / 10.0

    mapping = score_to_log or _default_mapping
    log_score = mapping(result.score)
    scored_state = text_score(new_state, log_score, name=f"{name}.score")
    return scored_state, result


@dataclass
class OverlapResult:
    """Structured result from an overlap analysis."""

    overlap: float
    judgement: str
    raw: str


def _parse_overlap_response(text: str) -> OverlapResult:
    score_match = re.search(r"OverlapScore:\s*([01](?:\.\d+)?)", text)
    overlap = float(score_match.group(1)) if score_match else 0.5
    judgement_match = re.search(r"Judgement:\s*(.+)", text, re.IGNORECASE)
    judgement = (
        judgement_match.group(1).strip() if judgement_match else "Neutral comparison."
    )
    judgement = re.sub(r"\s+", " ", judgement)
    return OverlapResult(overlap=overlap, judgement=judgement, raw=text.strip())


def text_overlap(
    state: PLPState,
    llm: LLMLike,
    candidate: str,
    peers: Sequence[str],
    *,
    task: str | None = None,
    name: str = "text_overlap",
    system_prompt: str | None = None,
    instructions: str | None = None,
) -> tuple[PLPState, OverlapResult]:
    """
    Ask an LLM to measure semantic overlap between a candidate and peer texts.
    """
    peers_block = "\n\n".join(peers) if peers else "None."
    base_instructions = (
        "You analyze how similar one persona's answer is to its peers. "
        "Low OverlapScore (near 0) means the candidate contributes distinct details."
    )
    directive = instructions.strip() if instructions else base_instructions
    prompt = (
        f"{directive}\n"
        "Return:\nOverlapScore: <float between 0 and 1>\nJudgement: <<=35 words highlighting uniqueness>\n\n"
    )
    if task:
        prompt += f"Task:\n{task.strip()}\n\n"
    prompt += f"Candidate:\n{candidate}\n\nPeers:\n{peers_block}\n"

    new_state, raw = run_prompt(
        llm,
        state,
        name=name,
        prompt=prompt,
        system_prompt=system_prompt,
        meta={"kind": "text_overlap"},
    )
    result = _parse_overlap_response(raw)
    return new_state, result


def apply_overlap_penalty(
    state: PLPState,
    overlap: float,
    diversity_force: float,
    *,
    name: str = "text_overlap_penalty",
) -> PLPState:
    """Convert an overlap score into a log-weight penalty."""
    log_score = -diversity_force * overlap
    return text_score(state, log_score, name=name)


@dataclass
class KernelAdvice:
    peer: str
    similarity: float
    advice: str


@dataclass
class KernelConstraint:
    summary: str
    advices: list[KernelAdvice]
    raw: str


def _parse_kernel_feedback(text: str) -> KernelConstraint:
    advice_pattern = re.compile(
        r"Peer:\s*(.*?)\s*\|\s*Similarity:\s*([01](?:\.\d+)?)\s*\|\s*Advice:\s*(.+?)(?=\nPeer:|\nSummary:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    advices: list[KernelAdvice] = []
    for match in advice_pattern.finditer(text):
        peer = match.group(1).strip()
        similarity = float(match.group(2).strip())
        advice = re.sub(r"\s+", " ", match.group(3)).strip()
        advices.append(KernelAdvice(peer=peer, similarity=similarity, advice=advice))

    summary_match = re.search(r"Summary:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    summary = (
        summary_match.group(1).strip()
        if summary_match
        else "Differentiate with a unique scenario."
    )
    summary = re.sub(r"\s+", " ", summary)
    return KernelConstraint(summary=summary, advices=advices, raw=text.strip())


def text_kernel_constraint(
    state: PLPState,
    llm: LLMLike,
    *,
    task: str,
    candidate_descriptor: str,
    candidate_answer: str,
    peers: Sequence[str],
    max_peers: int = 2,
    name: str = "text_kernel",
    system_prompt: str | None = None,
    instructions: str | None = None,
) -> tuple[PLPState, KernelConstraint]:
    """
    Translate SVGD kernel repulsion into textual advice from an LLM.

    Mirrors the "repel similar particles" justification in TextGrad's optimizer
    design by asking the judge to describe how a candidate should diverge from
    its closest peers. The resulting KernelConstraint can feed directly into
    SVGD update functions.
    """
    peers_block = "\n\n".join(peers) if peers else "None."
    base = (
        "You are computing kernel-based diversity constraints. "
        "Identify the closest peers and describe how to differentiate."
    )
    directive = instructions.strip() if instructions else base
    prompt = (
        f"{directive}\n"
        f"Report up to {max_peers} peers using:\n"
        "Peer: <descriptor or label> | Similarity: <0-1> | Advice: <<=25 words>\n"
        "Summary: <single sentence on how to diverge>\n\n"
        f"Task:\n{task.strip()}\n\n"
        f"Candidate descriptor:\n{candidate_descriptor}\n\n"
        f"Candidate answer:\n{candidate_answer}\n\n"
        f"Peers:\n{peers_block}\n"
    )
    new_state, raw = run_prompt(
        llm,
        state,
        name=name,
        prompt=prompt,
        system_prompt=system_prompt,
        meta={"kind": "text_kernel"},
    )
    constraint = _parse_kernel_feedback(raw)
    return new_state, constraint


@dataclass
class TextInferenceResult:
    """
    Lightweight container for text-level inference results.

    This is a simple convenience type for future higher-level
    inference interfaces built on top of textppl primitives.
    """

    state: PLPState
    text: str
    log_weight: float


@dataclass
class TextSite:
    """
    TextSite: a single textual random variable with optional checks.

    This is the L-PPL analogue of a random variable with:
    - a generator (prior / proposal) over strings
    - an optional observer (hard constraint / predicate)
    - an optional scorer (soft log-score / likelihood)

    The generator is responsible for calling text_sample or another
    sampling primitive; observe_fn and score_fn can be pure Python
    or can themselves invoke LLM-based judges. You can view a TextSite as the
    PLP analogue of a node in the TextGrad computation graph, but augmented
    with explicit observe/score hooks and traced log-weights.
    """

    name: str
    generator: Callable[[PLPState, Any], tuple[PLPState, str]]
    observe_fn: Callable[[PLPState, str, Any], bool] | None = None
    score_fn: Callable[[PLPState, str, Any], float] | None = None


def make_text_site(
    name: str,
    llm: LLMLike,
    prompt_fn: Callable[[Any], str],
    observe_fn: Callable[[PLPState, str, Any], bool] | None = None,
    score_fn: Callable[[PLPState, str, Any], float] | None = None,
) -> TextSite:
    """
    Convenience constructor for a TextSite from an LLM and a prompt builder.

    Args:
        name: Site name
        llm: LLM used for generation
        prompt_fn: task -> prompt string
        observe_fn: Optional predicate(state, text, task) -> bool
        score_fn: Optional scorer(state, text, task) -> log_score
    """

    def generator(state: PLPState, task: Any) -> tuple[PLPState, str]:
        prompt = prompt_fn(task)
        return text_sample(state, llm, prompt=prompt, name=name)

    return TextSite(
        name=name,
        generator=generator,
        observe_fn=observe_fn,
        score_fn=score_fn,
    )


def text_sampler(
    llm: LLMLike,
    name: str | None = None,
):
    """
    Decorator for defining text-sampling sites with docstring system prompts.

    The wrapped function should have signature:

        fn(state: PLPState, *args, **kwargs) -> str

    and return the *user prompt* string. The function's docstring is
    used as the system prompt for the underlying LLM call.

    Example:
        >>> from plp import text_sampler, PLPState, ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4.1-mini")
        >>>
        >>> @text_sampler(llm, name="solve_math")
        ... def solve(state: PLPState, problem: str) -> str:
        ...     \"\"\"You are a careful math assistant.\"\"\"
        ...     return f"Solve: {problem}. Reply with just the number."
        ...
        >>> state = PLPState()
        >>> new_state, answer = solve(state, "2+2")
    """

    def decorator(fn: Callable[..., str]):
        site_name = name or fn.__name__
        # Clean docstring to use as system prompt (may be empty)
        raw_doc = fn.__doc__ or ""
        system_prompt = inspect.cleandoc(raw_doc) or None

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, str]:
            prompt = fn(state, *args, **kwargs)
            if not isinstance(prompt, str):
                raise TypeError(
                    f"text_sampler-wrapped function {fn.__name__} must return a string prompt, "
                    f"got {type(prompt)}"
                )

            return text_sample(
                state,
                llm,
                prompt=prompt,
                name=site_name,
                system_prompt=system_prompt,
            )

        return wrapped

    return decorator


def _run_site_once(
    site: TextSite,
    task: Any,
    base_state: PLPState,
) -> TextInferenceResult:
    """
    Helper: run a TextSite once from base_state for a given task.

    Applies generator, then optional observe_fn and score_fn.
    """
    state = base_state.copy()
    state_after_gen, text = site.generator(state, task)

    state_cur = state_after_gen

    if site.observe_fn is not None:
        ok = site.observe_fn(state_cur, text, task)
        state_cur = text_observe(state_cur, ok, name=f"{site.name}.observe")

    if site.score_fn is not None:
        log_s = site.score_fn(state_cur, text, task)
        state_cur = text_score(state_cur, log_s, name=f"{site.name}.score")

    return TextInferenceResult(
        state=state_cur, text=text, log_weight=state_cur.log_weight
    )


def infer_rejection(
    site: TextSite,
    task: Any,
    max_attempts: int = 5,
    base_state: PLPState | None = None,
) -> TextInferenceResult:
    """
    Simple rejection-style inference for a single TextSite.

    Repeatedly samples from site's generator and applies observe/score
    until a valid trace (log_weight != -inf) is found or max_attempts
    is reached. Returns the last attempt either way.
    """
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")

    base = base_state or PLPState()
    last_result: TextInferenceResult | None = None

    for _ in range(max_attempts):
        result = _run_site_once(site, task, base)
        last_result = result
        if result.log_weight != float("-inf"):
            return result

    # Return last attempt even if invalid
    assert last_result is not None  # for type checkers
    return last_result


def infer_ensemble_map(
    site: TextSite,
    task: Any,
    num_samples: int = 5,
    base_state: PLPState | None = None,
) -> TextInferenceResult:
    """
    Ensemble-style approximate MAP for a single TextSite.

    Draws num_samples independent candidates from the site's generator,
    applies observe/score, and returns the candidate with highest
    log_weight (ignoring -inf traces when possible).
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    base = base_state or PLPState()
    results: list[TextInferenceResult] = []

    for _ in range(num_samples):
        results.append(_run_site_once(site, task, base))

    # Prefer valid traces; if none, fall back to max by log_weight anyway
    valid = [r for r in results if r.log_weight != float("-inf")]
    candidates = valid if valid else results

    best = max(candidates, key=lambda r: r.log_weight)
    return best


# =============================================================================
# Tools: deterministic transitions that can be invoked by agents
# =============================================================================


@dataclass
class ToolSpec:
    """Specification for a deterministic tool callable from agents."""

    name: str
    fn: Callable[..., tuple[PLPState, Any]]
    description: str = ""


_tools: dict[str, ToolSpec] = {}


def tool(name: str | None = None):
    """
    Decorator to register a deterministic tool.

    The wrapped function should have signature:

        fn(state: PLPState, *args, **kwargs) -> tuple[PLPState, Any]

    The function's docstring first line is used as the description.
    Each call records a deterministic TraceEvent with primitive
    'tool(<name>)'.
    """

    def decorator(fn: Callable[..., tuple[PLPState, Any]]):
        tool_name = name or fn.__name__
        raw_doc = fn.__doc__ or ""
        doc_clean = inspect.cleandoc(raw_doc)
        desc = doc_clean.splitlines()[0] if doc_clean else ""

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, Any]:
            result_state, result = fn(state, *args, **kwargs)
            new_state = result_state.copy()
            new_state.trace.append(
                TraceEvent(
                    primitive=f"tool({tool_name})",
                    prompt=str({"args": args, "kwargs": kwargs}),
                    output=str(result),
                    meta={"deterministic": True, "tool": tool_name},
                )
            )
            return new_state, result

        spec = ToolSpec(name=tool_name, fn=wrapped, description=desc)
        _tools[tool_name] = spec
        return wrapped

    return decorator


def get_tool(name: str) -> ToolSpec | None:
    """Get registered ToolSpec by name, if any."""
    return _tools.get(name)


def list_tools() -> list[str]:
    """List names of all registered tools."""
    return list(_tools.keys())


def _resolve_tools(tools: Sequence[str | ToolSpec]) -> dict[str, ToolSpec]:
    """Normalize a sequence of tool names/specs into a name->spec mapping."""
    mapping: dict[str, ToolSpec] = {}
    for t in tools:
        if isinstance(t, ToolSpec):
            mapping[t.name] = t
        elif isinstance(t, str):
            spec = get_tool(t)
            if spec is None:
                raise ValueError(f"No tool registered with name: {t!r}")
            mapping[spec.name] = spec
        else:
            raise TypeError(f"tools must be ToolSpec or str, got {type(t)}")
    return mapping


def agent_sampler(
    llm: LLMLike,
    tools: Sequence[str | ToolSpec],
    name: str | None = None,
    max_steps: int = 8,
):
    """
    Decorator for defining a basic tool-using agent.

    The wrapped function should have signature:

        fn(state: PLPState, *args, **kwargs) -> str

    and return the initial user prompt. The function's docstring is used
    as the system prompt. The agent interacts with tools via a simple
    text protocol:

        CALL <tool_name>: <arguments>
        FINAL: <answer>

    At each step, the LLM sees either the original prompt (step 0) or the
    latest tool result, and must respond with either a CALL or FINAL line.
    """

    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    tool_specs = _resolve_tools(tools)

    def decorator(fn: Callable[..., str]):
        site_name = name or fn.__name__
        raw_doc = fn.__doc__ or ""
        system_prompt = inspect.cleandoc(raw_doc) or None

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, str]:
            initial_prompt = fn(state, *args, **kwargs)
            if not isinstance(initial_prompt, str):
                raise TypeError(
                    f"agent_sampler-wrapped function {fn.__name__} must return a string prompt, "
                    f"got {type(initial_prompt)}"
                )

            cur_state = state.copy()
            last_tool_result = ""

            tool_meta = {"tools": list(tool_specs.keys()), "agent": site_name}

            for step in range(max_steps):
                if step == 0:
                    prompt = (
                        f"{initial_prompt}\n\n"
                        "You may use tools by replying with:\n"
                        "CALL <tool_name>: <arguments>\n"
                        "When you are done, reply with:\n"
                        "FINAL: <answer>\n"
                    )
                else:
                    prompt = (
                        f"Last tool result:\n{last_tool_result}\n\n"
                        "Now either:\n"
                        " - CALL <tool_name>: <arguments>\n"
                        " - or FINAL: <answer>\n"
                    )

                step_name = f"{site_name}.step{step}"
                cur_state, response = run_prompt(
                    llm,
                    cur_state,
                    name=step_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    meta={"kind": "agent_step", **tool_meta},
                )
                text = response.strip()

                upper = text.upper()
                if upper.startswith("CALL "):
                    after = text[5:].strip()
                    if ":" in after:
                        tool_name_raw, arg_text = after.split(":", 1)
                    else:
                        tool_name_raw, arg_text = after, ""
                    tool_name = tool_name_raw.strip()
                    arg_text = arg_text.strip()

                    spec = tool_specs.get(tool_name)
                    if spec is None:
                        # Unknown tool - treat as final answer
                        break

                    sig = inspect.signature(spec.fn)
                    params = list(sig.parameters.values())[1:]  # skip state
                    if len(params) == 0:
                        cur_state, tool_result = spec.fn(cur_state)
                    else:
                        cur_state, tool_result = spec.fn(cur_state, arg_text)

                    last_tool_result = str(tool_result)
                    continue

                if upper.startswith("FINAL:"):
                    final_answer = text[len("FINAL:") :].strip()
                    return cur_state, final_answer

                # Fallback: treat whole response as final answer
                return cur_state, text

            # If loop exhausted without FINAL, fall back to last tool result or empty
            final_answer = last_tool_result or ""
            final_state = cur_state.copy()
            final_state.trace.append(
                TraceEvent(
                    primitive=site_name,
                    prompt="(agent_sampler exhausted steps)",
                    output=final_answer,
                    meta={"kind": "agent", **tool_meta, "exhausted": True},
                )
            )
            return final_state, final_answer

        return wrapped

    return decorator


# =============================================================================
# Re-export advanced patterns (plate, parameters) for text-first workflows
# =============================================================================



# =============================================================================
# Classic agentic patterns built on textppl
# =============================================================================


def cot_sampler(
    llm: LLMLike,
    name: str | None = None,
):
    """
    Decorator for a simple Chain-of-Thought style sampler.

    Wraps a prompt-building function and appends a generic
    "think step by step" instruction to the user prompt, while
    still using the function's docstring as the system prompt.

    This is syntactic sugar over @text_sampler.
    """

    def decorator(fn: Callable[..., str]):
        site_name = name or fn.__name__

        @wraps(fn)
        def prompt_fn(state: PLPState, *args, **kwargs) -> str:
            base = fn(state, *args, **kwargs)
            hint = (
                "\n\nThink step by step before answering. "
                "Then provide your final answer clearly."
            )
            return f"{base}{hint}"

        return text_sampler(llm, name=site_name)(prompt_fn)

    return decorator


def reflection_sampler(
    llm: LLMLike,
    name: str | None = None,
):
    """
    Decorator for a simple reflection pattern:

    1. Generate an initial draft answer.
    2. Apply a TextGrad-style refinement step to improve it.

    Both steps are logged as separate TraceEvents:
      - <name>.draft
      - <name>.refine
    """

    def decorator(fn: Callable[..., str]):
        site_name = name or fn.__name__
        raw_doc = fn.__doc__ or ""
        system_prompt = inspect.cleandoc(raw_doc) or None

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, str]:
            base_prompt = fn(state, *args, **kwargs)
            if not isinstance(base_prompt, str):
                raise TypeError(
                    f"reflection_sampler-wrapped function {fn.__name__} must "
                    f"return a string prompt, got {type(base_prompt)}"
                )

            # Step 1: draft
            s1, draft = text_sample(
                state,
                llm,
                prompt=base_prompt,
                name=f"{site_name}.draft",
                system_prompt=system_prompt,
            )

            # Step 2: refine via TextGrad
            loss_desc = (
                "Improve the answer for correctness, clarity, and completeness. "
                "Fix any errors and tighten the explanation."
            )
            s2, refined = textgrad_step(
                s1,
                llm,
                current=draft,
                loss_description=loss_desc,
                name=f"{site_name}.refine",
                system_prompt=system_prompt,
            )

            return s2, refined

        return wrapped

    return decorator


def self_consistency_sampler(
    llm: LLMLike,
    n_samples: int = 5,
    name: str | None = None,
):
    """
    Decorator for a simple self-consistency pattern.

    Runs the same prompt multiple times and returns the most frequent
    answer (exact-string majority). Individual draws are recorded as:
      - <name>.sample_0, <name>.sample_1, ...
    and an aggregate event:
      - <name>
    is appended with metadata about counts.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    def decorator(fn: Callable[..., str]):
        site_name = name or fn.__name__
        raw_doc = fn.__doc__ or ""
        system_prompt = inspect.cleandoc(raw_doc) or None

        @wraps(fn)
        def wrapped(state: PLPState, *args, **kwargs) -> tuple[PLPState, str]:
            base_prompt = fn(state, *args, **kwargs)
            if not isinstance(base_prompt, str):
                raise TypeError(
                    f"self_consistency_sampler-wrapped function {fn.__name__} must "
                    f"return a string prompt, got {type(base_prompt)}"
                )

            cur_state = state.copy()
            answers: list[str] = []

            for i in range(n_samples):
                cur_state, out = text_sample(
                    cur_state,
                    llm,
                    prompt=base_prompt,
                    name=f"{site_name}.sample_{i}",
                    system_prompt=system_prompt,
                )
                answers.append(out.strip())

            # Majority vote over exact strings
            counts: dict[str, int] = {}
            for a in answers:
                counts[a] = counts.get(a, 0) + 1
            best_answer, best_count = max(counts.items(), key=lambda kv: kv[1])

            final_state = cur_state.copy()
            final_state.trace.append(
                TraceEvent(
                    primitive=site_name,
                    prompt=base_prompt,
                    output=best_answer,
                    meta={
                        "kind": "self_consistency",
                        "llm": llm_name(llm),
                        "n_samples": n_samples,
                        "counts": counts,
                    },
                )
            )
            return final_state, best_answer

        return wrapped

    return decorator
