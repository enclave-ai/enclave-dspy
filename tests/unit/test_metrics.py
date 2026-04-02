import dspy

from src.metrics.generic import combined_metric, structural_completeness


def _make_example_and_pred(response: str, expected: str):
    example = dspy.Example(
        context="some code",
        task="find vulnerabilities",
        response=expected,
    ).with_inputs("context", "task")

    pred = dspy.Prediction(response=response)
    return example, pred


def test_structural_completeness_full_response():
    example, pred = _make_example_and_pred(
        response="## Finding\n\nSeverity: High\n\nDescription: XSS vulnerability found",
        expected="anything",
    )
    score = structural_completeness(example, pred)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_structural_completeness_empty_response():
    example, pred = _make_example_and_pred(response="", expected="anything")
    score = structural_completeness(example, pred)
    assert score == 0.0


def test_combined_metric_returns_float():
    example, pred = _make_example_and_pred(
        response="Found SQL injection in login handler",
        expected="SQL injection in login handler with high severity",
    )
    weights = {"structural": 0.5, "length_ratio": 0.5}
    score = combined_metric(example, pred, trace=None, weights=weights)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_combined_metric_defaults():
    example, pred = _make_example_and_pred(
        response="Found an issue",
        expected="Found a critical issue",
    )
    score = combined_metric(example, pred)
    assert isinstance(score, float)
