from __future__ import annotations

import re

import dspy


SEVERITY_LEVELS = {"critical", "high", "medium", "low", "info"}


def finding_precision_recall(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    expected_findings = _extract_finding_keywords(getattr(example, "response", ""))
    actual_findings = _extract_finding_keywords(getattr(pred, "response", ""))

    if not expected_findings and not actual_findings:
        return 1.0
    if not expected_findings or not actual_findings:
        return 0.0

    matches = expected_findings & actual_findings
    precision = len(matches) / len(actual_findings) if actual_findings else 0.0
    recall = len(matches) / len(expected_findings) if expected_findings else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def severity_accuracy(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    expected_severities = _extract_severities(getattr(example, "response", ""))
    actual_severities = _extract_severities(getattr(pred, "response", ""))

    if not expected_severities:
        return 1.0 if not actual_severities else 0.5

    matches = sum(1 for s in expected_severities if s in actual_severities)
    return matches / len(expected_severities)


def dedup_accuracy(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    expected = getattr(example, "response", "").lower()
    actual = getattr(pred, "response", "").lower()

    expected_is_dup = any(w in expected for w in ["duplicate", "same", "already reported"])
    actual_is_dup = any(w in actual for w in ["duplicate", "same", "already reported"])

    return 1.0 if expected_is_dup == actual_is_dup else 0.0


DOMAIN_METRICS = {
    "finding_precision": finding_precision_recall,
    "finding_recall": finding_precision_recall,
    "severity_accuracy": severity_accuracy,
    "dedup_accuracy": dedup_accuracy,
    "false_merge_rate": dedup_accuracy,
    "coverage": finding_precision_recall,
    "delegation_quality": finding_precision_recall,
    "relevance": finding_precision_recall,
    "correctness": finding_precision_recall,
    "actionability": finding_precision_recall,
    "finding_relevance": finding_precision_recall,
    "false_alarm_rate": dedup_accuracy,
    "severity_calibration": severity_accuracy,
    "false_positive_rate": dedup_accuracy,
    "false_negative_rate": dedup_accuracy,
    "reasoning_quality": finding_precision_recall,
}


def get_domain_metric(metric_name: str):
    return DOMAIN_METRICS.get(metric_name, finding_precision_recall)


def _extract_finding_keywords(text: str) -> set[str]:
    vuln_patterns = [
        r"(?i)(xss|cross.site.scripting)",
        r"(?i)(sql.injection|sqli)",
        r"(?i)(ssrf|server.side.request)",
        r"(?i)(command.injection|rce|remote.code)",
        r"(?i)(path.traversal|directory.traversal)",
        r"(?i)(idor|insecure.direct)",
        r"(?i)(csrf|cross.site.request.forgery)",
        r"(?i)(auth\w*\s+bypass)",
        r"(?i)(information.disclosure|data.leak)",
        r"(?i)(template.injection|ssti)",
    ]
    found = set()
    for pattern in vuln_patterns:
        if re.search(pattern, text):
            found.add(pattern)
    return found


def _extract_severities(text: str) -> list[str]:
    found = []
    for level in SEVERITY_LEVELS:
        if re.search(rf"(?i)\b{level}\b", text):
            found.append(level)
    return found
