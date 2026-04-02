from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from src.config import Config
from src.metrics.domain import get_domain_metric
from src.metrics.generic import combined_metric


class OptimizationResult:
    def __init__(
        self,
        agent_id: str,
        optimizer_name: str,
        baseline_score: float,
        optimized_score: float,
        program: dspy.Module,
    ):
        self.agent_id = agent_id
        self.optimizer_name = optimizer_name
        self.baseline_score = baseline_score
        self.optimized_score = optimized_score
        self.program = program

    @property
    def improvement(self) -> float:
        if self.baseline_score == 0:
            return self.optimized_score
        return (self.optimized_score - self.baseline_score) / self.baseline_score


def build_metric(agent_id: str, config: Config):
    agent_config = config.get_agent_config(agent_id)
    weights = agent_config.metric_weights

    if not weights:
        return combined_metric

    def metric(example, pred, trace=None):
        total = 0.0
        total_weight = 0.0
        for name, weight in weights.items():
            fn = get_domain_metric(name)
            score = fn(example, pred, trace)
            total += score * weight
            total_weight += weight
        return total / total_weight if total_weight > 0 else 0.0

    return metric


def evaluate_program(
    program: dspy.Module,
    devset: list[dspy.Example],
    metric,
) -> float:
    if not devset:
        return 0.0
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=4,
        display_progress=True,
    )
    result = evaluator(program)
    return float(result)


def run_optimization(
    program: dspy.Module,
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
    agent_id: str,
    config: Config,
    optimizer_name: str | None = None,
    output_dir: Path | None = None,
) -> OptimizationResult:
    metric = build_metric(agent_id, config)
    agent_config = config.get_agent_config(agent_id)
    defaults = config.defaults

    max_bootstrapped = (
        agent_config.max_bootstrapped_demos
        or defaults.max_bootstrapped_demos
    )
    max_labeled = (
        agent_config.max_labeled_demos
        or defaults.max_labeled_demos
    )

    baseline_score = evaluate_program(program, devset, metric)

    optimizers_to_run = (
        [optimizer_name] if optimizer_name
        else agent_config.optimizers
    )

    best_result = OptimizationResult(
        agent_id=agent_id,
        optimizer_name="baseline",
        baseline_score=baseline_score,
        optimized_score=baseline_score,
        program=program,
    )

    for opt_name in optimizers_to_run:
        candidate = program.deepcopy()
        optimized = _run_single_optimizer(
            opt_name, candidate, trainset, metric,
            max_bootstrapped, max_labeled, defaults,
        )

        score = evaluate_program(optimized, devset, metric)

        if score > best_result.optimized_score:
            best_result = OptimizationResult(
                agent_id=agent_id,
                optimizer_name=opt_name,
                baseline_score=baseline_score,
                optimized_score=score,
                program=optimized,
            )

        if best_result.improvement >= defaults.improvement_threshold:
            break

    if output_dir:
        _save_result(best_result, output_dir)

    return best_result


def _run_single_optimizer(
    name: str,
    program: dspy.Module,
    trainset: list[dspy.Example],
    metric,
    max_bootstrapped: int,
    max_labeled: int,
    defaults,
) -> dspy.Module:
    if name == "bootstrap":
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )
        return optimizer.compile(student=program, trainset=trainset)

    elif name == "mipro_light":
        optimizer = MIPROv2(
            metric=metric,
            auto="light",
            num_threads=4,
        )
        return optimizer.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )

    elif name == "mipro_medium":
        optimizer = MIPROv2(
            metric=metric,
            auto="medium",
            num_threads=8,
        )
        return optimizer.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=max_labeled,
        )

    elif name == "gepa":
        from dspy.teleprompt import GEPA
        optimizer = GEPA(
            metric=metric,
        )
        return optimizer.compile(
            program,
            trainset=trainset,
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _save_result(result: OptimizationResult, output_dir: Path):
    agent_dir = output_dir / result.agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    existing = list(agent_dir.glob("v*.json"))
    version = len(existing) + 1
    version_str = f"v{version}"

    result.program.save(str(agent_dir / f"{version_str}.json"))

    report = agent_dir / "report.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"\n## {version_str} — {timestamp}\n\n"
        f"- **Optimizer:** {result.optimizer_name}\n"
        f"- **Baseline score:** {result.baseline_score:.4f}\n"
        f"- **Optimized score:** {result.optimized_score:.4f}\n"
        f"- **Improvement:** {result.improvement:.2%}\n"
    )

    if report.exists():
        content = report.read_text() + entry
    else:
        content = f"# Optimization Report: {result.agent_id}\n" + entry

    report.write_text(content)
