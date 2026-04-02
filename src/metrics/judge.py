from __future__ import annotations

import dspy


class JudgeAssessment(dspy.Signature):
    """Assess the quality of an AI agent's response compared to the expected output."""

    task: str = dspy.InputField(desc="The task the agent was asked to perform")
    expected_response: str = dspy.InputField(desc="The expected/ideal response")
    actual_response: str = dspy.InputField(desc="The agent's actual response")
    relevance_score: float = dspy.OutputField(desc="How relevant the response is to the task (0.0-1.0)")
    completeness_score: float = dspy.OutputField(desc="How complete the response is (0.0-1.0)")
    accuracy_score: float = dspy.OutputField(desc="How accurate the response is vs expected (0.0-1.0)")
    feedback: str = dspy.OutputField(desc="Brief textual feedback explaining the scores")


class LLMJudge:
    def __init__(self, judge_model: str = "anthropic/claude-sonnet-4-6"):
        self.judge_lm = dspy.LM(judge_model)
        self.assess = dspy.ChainOfThought(JudgeAssessment)

    def __call__(
        self,
        example: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
    ) -> float:
        task = getattr(example, "task", "")
        expected = getattr(example, "response", "")
        actual = getattr(pred, "response", "")

        if not actual:
            return 0.0

        with dspy.context(lm=self.judge_lm):
            result = self.assess(
                task=task,
                expected_response=expected,
                actual_response=actual,
            )

        scores = [
            float(result.relevance_score),
            float(result.completeness_score),
            float(result.accuracy_score),
        ]
        return sum(scores) / len(scores)
