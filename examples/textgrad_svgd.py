"""
PLP persona engineering demo.

This script mirrors the explanation in `docs/plp_theoretical_foundations.md`: each persona is a
parameterized prompt, and we treat the set of answers as particles. A TextGrad step refines every
persona prompt and every answer based on textual overlap judgments, so the ensemble gradually
trades off individual quality against redundancy. The CLI focuses on the persona/mix view, nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

import plp
from plp import ChatOpenAI, PLPState
from plp.textppl import textgrad_step

APP = typer.Typer()
CONSOLE = Console()
COLOR_PALETTE = ["cyan", "magenta", "yellow", "green", "bright_blue", "bright_magenta"]


@dataclass
class PersonaProfile:
    descriptor: str


@dataclass
class PersonaState:
    descriptor: str


@dataclass
class KernelAdvice:
    peer: str
    similarity: float
    advice: str


@dataclass
class KernelConstraint:
    summary: str
    advices: list[KernelAdvice]


@dataclass
class PersonaMetrics:
    evaluation_score: float
    evaluation_feedback: str
    overlap_score: float
    overlap_metric: str
    overlap_judgement: str
    kernel_constraint: KernelConstraint


PERSONA_PROMPTS = [
    PersonaProfile(descriptor="You emphasize concise logic."),
    PersonaProfile(descriptor="You compare competing hypotheses."),
    PersonaProfile(descriptor="You weigh trade-offs and safety."),
    PersonaProfile(descriptor="You situate ideas in real-world progress."),
    PersonaProfile(descriptor="You emphasize fairness and clarity."),
]


def build_persona_prompt(
    persona: PersonaProfile, question: str, descriptor: str
) -> str:
    return (
        f"{descriptor} Answer the question below in 3 sentences or fewer.\n"
        f"Question:\n{question}\n"
        "Keep your tone consistent and highlight what makes this persona distinct.\n"
    )


def init_personas(
    state: PLPState, llm: ChatOpenAI, question: str, num_personas: int
) -> tuple[PLPState, list[str], list[PersonaState]]:
    personas: list[str] = []
    states: list[PersonaState] = []
    current_state = state
    for idx in range(num_personas):
        profile = PERSONA_PROMPTS[idx % len(PERSONA_PROMPTS)]
        descriptor = profile.descriptor
        prompt = build_persona_prompt(profile, question, descriptor)
        current_state, response = plp.text_sample(
            state=current_state,
            llm=llm,
            prompt=prompt,
            name=f"persona_init_{idx + 1}",
        )
        personas.append(response.strip())
        states.append(PersonaState(descriptor))
    return current_state, personas, states


def parse_overlap_response(text: str) -> tuple[float, str]:
    score_match = re.search(r"OverlapScore:\s*([01](?:\.\d+)?)", text)
    judgement_match = re.search(r"Judgement:\s*([^\n]+)", text, re.IGNORECASE)

    score = float(score_match.group(1)) if score_match else 0.5
    judgement = judgement_match.group(1).strip() if judgement_match else "Neutral"
    return score, judgement


def parse_score_feedback(text: str) -> tuple[float, str]:
    score_match = re.search(r"Score:\s*(\d{1,3})", text)
    score = float(score_match.group(1)) if score_match else 60.0
    score = max(0.0, min(score, 100.0))
    feedback_match = re.search(r"Feedback:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else "Clarify the reasoning steps."
    feedback = re.sub(r"\s+", " ", feedback)
    return score, feedback


def evaluate_answer(
    state: PLPState,
    llm: ChatOpenAI,
    question: str,
    candidate: str,
    persona_label: str,
) -> tuple[PLPState, str, float, str]:
    prompt = (
        "You are a senior evaluator of reasoning quality. Score the candidate answer to the user question "
        "for correctness, completeness, and clarity on a 0-100 scale. Penalize redundancy and reward concrete detail. "
        "Return exactly:\nScore: <integer 0-100>\nFeedback: <<=35 words describing how to improve>.\n"
        f"Question:\n{question}\n\nCandidate answer:\n{candidate}\n"
    )
    state, output = plp.text_sample(
        state=state,
        llm=llm,
        prompt=prompt,
        name=f"persona_eval_{persona_label}",
    )
    score, feedback = parse_score_feedback(output)
    metric_summary = output.strip().replace("\n", " | ")
    return state, metric_summary, score, feedback


def parse_kernel_feedback(text: str) -> KernelConstraint:
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
    summary = summary_match.group(1).strip() if summary_match else "Differentiate with a unique method or scenario."
    summary = re.sub(r"\s+", " ", summary)
    return KernelConstraint(summary=summary, advices=advices)


def compute_kernel_constraint(
    state: PLPState,
    llm: ChatOpenAI,
    idx: int,
    persona_states: list[PersonaState],
    answers: list[str],
    question: str,
    max_peers: int = 2,
) -> tuple[PLPState, KernelConstraint]:
    if len(persona_states) <= 1:
        return state, KernelConstraint(
            summary="You are the sole persona; lean fully into your descriptor.",
            advices=[],
        )

    descriptor = persona_states[idx].descriptor
    candidate_answer = answers[idx]
    peer_blocks = []
    for peer_idx, peer_state in enumerate(persona_states):
        if peer_idx == idx:
            continue
        peer_text = (
            f"Peer persona[{peer_idx}] descriptor: {peer_state.descriptor}\n"
            f"Peer answer:\n{answers[peer_idx]}\n"
        )
        peer_blocks.append(peer_text)

    peers_formatted = "\n".join(peer_blocks)
    prompt = (
        "You are computing SVGD-style kernel repulsion for persona optimization. "
        "Compare the candidate persona to the peers and identify up to "
        f"{max_peers} closest peers with semantic similarity in [0,1]. "
        "Provide guidance on how to differentiate. Use the format:\n"
        "Peer: <descriptor or label> | Similarity: <float 0-1> | Advice: <<=25 words>.\n"
        "Summary: <single sentence describing the most distinct direction>\n\n"
        f"Task question:\n{question}\n\n"
        f"Candidate descriptor: {descriptor}\n"
        f"Candidate answer:\n{candidate_answer}\n\n"
        f"Peers:\n{peers_formatted}\n"
    )
    state, output = plp.text_sample(
        state=state,
        llm=llm,
        prompt=prompt,
        name=f"kernel_constraint_{idx + 1}",
    )
    constraint = parse_kernel_feedback(output)
    return state, constraint


def pairwise_overlap(
    state: PLPState,
    llm: ChatOpenAI,
    candidate: str,
    others: Iterable[str],
) -> tuple[PLPState, str, float, str]:
    others_block = "\n".join(f"- {o}" for o in others) if others else "None"
    prompt = (
        "You are a text-overlap analyst. Compare this answer to the other persona replies. "
        "Return:\nOverlapScore: float between 0 and 1 (lower means more unique)\n"
        "Judgement: short comment describing what makes this answer stand out—mention an example, tool, or structure the others did not use.\n"
        f"\nCandidate:\n{candidate}\n\nOthers:\n{others_block}\n"
    )
    state, output = plp.text_sample(
        state=state,
        llm=llm,
        prompt=prompt,
        name="pairwise_overlap",
    )
    score, judgement = parse_overlap_response(output)
    metric_summary = output.strip().replace("\n", " | ")
    return state, metric_summary, score, judgement


def make_loss_description(
    answer: str,
    question: str,
    evaluation_score: float,
    evaluation_feedback: str,
    overlap_score: float,
    overlap_metric: str,
    overlap_judgement: str,
    diversity_force: float,
    kernel_constraint: KernelConstraint,
) -> str:
    task_clause = f"Task: {question.strip()}."
    quality_clause = (
        f"Evaluator score {evaluation_score:.0f}/100 for that task. "
        f"Improve by acting on: {evaluation_feedback}."
    )
    overlap_clause = (
        f"Overlap score {overlap_score:.2f} ({overlap_judgement}). {overlap_metric}."
    )
    if kernel_constraint.advices:
        advice_clause = " ".join(
            f"Avoid mirroring {advice.peer} (sim={advice.similarity:.2f}) by {advice.advice}."
            for advice in kernel_constraint.advices
        )
    else:
        advice_clause = "Push into a novel angle; no peers nearby."
    kernel_clause = (
        f"Kernel repulsion (strength {diversity_force:.2f}) summary: {kernel_constraint.summary}. "
        f"{advice_clause}"
    )
    instruction = (
        "Produce a refined answer under 80 words, stay tightly grounded in the task question, maintain persona voice, introduce at least one concrete method or example "
        "missing from peers, and explicitly address the evaluator feedback."
    )
    return f"Loss description: {task_clause} {quality_clause} {overlap_clause} {kernel_clause} {instruction}"


def svgd_iteration(
    state: PLPState,
    answers: list[str],
    persona_states: list[PersonaState],
    llm: ChatOpenAI,
    diversity_force: float,
    question: str,
) -> tuple[PLPState, list[str], list[PersonaMetrics], list[PersonaState]]:
    current_state = state
    new_answers: list[str] = []
    metrics: list[PersonaMetrics] = []
    for idx, current in enumerate(answers):
        persona_label = f"{idx + 1}"
        current_state, _, eval_score, eval_feedback = evaluate_answer(
            current_state, llm, question, current, persona_label
        )
        current_state, kernel_constraint = compute_kernel_constraint(
            current_state, llm, idx, persona_states, answers, question
        )
        others = [a for j, a in enumerate(answers) if j != idx]
        current_state, metric_text, overlap_score, judgement = pairwise_overlap(
            current_state, llm, current, others
        )
        loss_desc = make_loss_description(
            current,
            question,
            eval_score,
            eval_feedback,
            overlap_score,
            metric_text,
            judgement,
            diversity_force,
            kernel_constraint,
        )
        current_state, edited = textgrad_step(
            current_state,
            llm,
            current=current,
            loss_description=loss_desc,
            name=f"textgrad_answer_{idx + 1}",
        )
        new_answers.append(edited.strip())
        metrics.append(
            PersonaMetrics(
                evaluation_score=eval_score,
                evaluation_feedback=eval_feedback,
                overlap_score=overlap_score,
                overlap_metric=metric_text,
                overlap_judgement=judgement,
                kernel_constraint=kernel_constraint,
            )
        )

    for idx, metric in enumerate(metrics):
        descriptor = persona_states[idx].descriptor
        others = (
            "; ".join(
                peer_state.descriptor
                for j, peer_state in enumerate(persona_states)
                if j != idx
            )
            or "none"
        )
        if metric.kernel_constraint.advices:
            kernel_detail = " ".join(
                f"Avoid {advice.peer} by {advice.advice} (sim={advice.similarity:.2f})."
                for advice in metric.kernel_constraint.advices
            )
        else:
            kernel_detail = "Carve out a distinctive scenario; no close peers."
        persona_loss = (
            f"Descriptor #{idx} should rewrite itself more boldly ({metric.overlap_score:.2f} overlap, "
            f"score {metric.evaluation_score:.0f}/100). "
            f"Others now focus on {others}. Push diversity by {diversity_force:.2f}, mention a different concrete scenario, "
            f"call out a tool or method the others haven't referenced, and stay under 40 words. "
            f"Anchor every change on the task question: {question}. "
            f"Kernel repulsion summary: {metric.kernel_constraint.summary}. {kernel_detail}"
        )
        current_state, edited_descriptor = textgrad_step(
            current_state,
            llm,
            current=descriptor,
            loss_description=persona_loss,
            name=f"textgrad_persona_{idx + 1}",
        )
        persona_states[idx].descriptor = edited_descriptor.strip()

    return current_state, new_answers, metrics, persona_states


def aggregate_quality(evaluation_score: float, overlap_score: float) -> float:
    return (evaluation_score / 100.0) * (1 - overlap_score)


def log_descriptors(
    stage: str,
    persona_states: list[PersonaState],
    overlaps: list[float] | None = None,
    scores: list[float] | None = None,
) -> None:
    table = Table(title=stage, show_header=True, header_style="bold magenta")
    table.add_column("Persona", style="bold")
    table.add_column("Descriptor")
    table.add_column("Overlap", justify="right")
    table.add_column("Score", justify="right")

    for idx, persona in enumerate(persona_states):
        overlap_display = f"{overlaps[idx]:.2f}" if overlaps else "n/a"
        score_display = f"{scores[idx]:.0f}" if scores else "n/a"
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        table.add_row(
            f"[{color}]persona[{idx}][/{color}]",
            persona.descriptor,
            overlap_display,
            score_display,
        )

    CONSOLE.print(table)


@APP.command()
def svgd(
    question: str = typer.Option(
        "How would you rephrase a reasoning process to be clearer?", "--question"
    ),
    iterations: int = typer.Option(
        3, "--iterations", "-n", help="Number of TextGrad iterations."
    ),
    personas: int = typer.Option(
        3, "--personas", "-p", help="Number of persona particles to initialize."
    ),
    diversity_force: float = typer.Option(
        0.7,
        "--diversity-force",
        help="Strength of the diversity pressure applied to descriptions.",
    ),
    env_file: str = typer.Option(".env", "--env-file", help="Path to dotenv file."),
) -> None:
    """Run the persona-engineering SVGD-style demo."""
    load_dotenv(env_file)
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1.0)
    state = PLPState()
    state, answers, persona_states = init_personas(state, llm, question, personas)

    CONSOLE.rule(f"Initial personas ({len(answers)})")
    log_descriptors("Initial descriptors", persona_states)

    last_metrics: list[PersonaMetrics] = []
    for iteration in range(1, iterations + 1):
        state, answers, metrics, persona_states = svgd_iteration(
            state, answers, persona_states, llm, diversity_force, question
        )
        last_metrics = metrics
        overlaps = [metric.overlap_score for metric in metrics]
        scores = [metric.evaluation_score for metric in metrics]
        log_descriptors(
            f"Iteration {iteration} descriptors", persona_states, overlaps, scores
        )

    if not last_metrics:
        CONSOLE.print("No iterations were executed.")
        return

    qualities = [
        aggregate_quality(metric.evaluation_score, metric.overlap_score)
        for metric in last_metrics
    ]
    best_idx = max(range(len(qualities)), key=lambda i: qualities[i])
    best_answer = answers[best_idx]
    meta_prompt = (
        "Distill the most reliable persona answer. "
        "Keep the tone concise and mention why this approach is different."
    )
    state, distilled = textgrad_step(
        state,
        llm,
        current=best_answer,
        loss_description=meta_prompt,
        name="textgrad_meta",
    )
    CONSOLE.rule("Final distilled reasoning")
    CONSOLE.print(distilled.strip())


if __name__ == "__main__":
    APP()
