from __future__ import annotations

import dspy


def structural_completeness(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    response = getattr(pred, "response", "")
    if not response:
        return 0.0

    indicators = [
        len(response) > 50,
        "\n" in response,
        any(
            marker in response.lower()
            for marker in ["finding", "severity", "description", "issue", "vulnerability"]
        ),
        any(marker in response for marker in ["##", "- ", "1.", "*"]),
    ]
    return sum(indicators) / len(indicators)


def length_ratio(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    expected = getattr(example, "response", "")
    actual = getattr(pred, "response", "")
    if not expected:
        return 1.0 if actual else 0.0
    ratio = len(actual) / len(expected)
    if ratio > 1.0:
        ratio = 1.0 / ratio
    return min(ratio, 1.0)


def combined_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    weights: dict[str, float] | None = None,
) -> float:
    if weights is None:
        weights = {"structural": 0.5, "length_ratio": 0.5}

    metric_fns = {
        "structural": structural_completeness,
        "length_ratio": length_ratio,
    }

    total = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        if name in metric_fns:
            score = metric_fns[name](example, pred, trace)
            total += score * weight
            total_weight += weight

    return total / total_weight if total_weight > 0 else 0.0
