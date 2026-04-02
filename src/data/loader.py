from __future__ import annotations

import json
from pathlib import Path

import dspy
from pydantic import BaseModel


class DatasetInfo(BaseModel):
    agent_id: str
    train_count: int = 0
    dev_count: int = 0
    test_count: int = 0


def load_dataset(file_path: Path) -> list[dspy.Example]:
    examples = []
    for line in file_path.read_text().strip().splitlines():
        row = json.loads(line)
        inputs = row["inputs"]
        outputs = row["outputs"]
        all_fields = {**inputs, **outputs}
        example = dspy.Example(**all_fields).with_inputs(*inputs.keys())
        examples.append(example)
    return examples


def load_all_splits(agent_dir: Path) -> dict[str, list[dspy.Example]]:
    splits = {}
    for split_name in ("train", "dev", "test"):
        file_path = agent_dir / f"{split_name}.jsonl"
        if file_path.exists():
            splits[split_name] = load_dataset(file_path)
        else:
            splits[split_name] = []
    return splits


def get_dataset_status(datasets_dir: Path, agent_id: str) -> DatasetInfo:
    agent_dir = datasets_dir / agent_id
    if not agent_dir.exists():
        return DatasetInfo(agent_id=agent_id)

    splits = load_all_splits(agent_dir)
    return DatasetInfo(
        agent_id=agent_id,
        train_count=len(splits["train"]),
        dev_count=len(splits["dev"]),
        test_count=len(splits["test"]),
    )
