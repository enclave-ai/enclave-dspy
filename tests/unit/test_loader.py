import json

import dspy

from src.data.loader import get_dataset_status, load_all_splits, load_dataset


def test_load_dataset_from_jsonl(tmp_path):
    data_file = tmp_path / "train.jsonl"
    row1 = json.dumps({
        "inputs": {"context": "code here", "task": "find vulns"},
        "outputs": {"response": "found XSS"},
    })
    row2 = json.dumps({
        "inputs": {"context": "more code", "task": "audit"},
        "outputs": {"response": "no issues"},
    })
    data_file.write_text(row1 + "\n" + row2 + "\n")
    examples = load_dataset(data_file)
    assert len(examples) == 2
    assert isinstance(examples[0], dspy.Example)
    assert examples[0].context == "code here"
    assert examples[0].task == "find vulns"
    assert examples[0].response == "found XSS"


def test_load_dataset_marks_inputs(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        json.dumps({"inputs": {"context": "c", "task": "t"}, "outputs": {"response": "r"}}) + "\n"
    )
    examples = load_dataset(data_file)
    inputs = examples[0].inputs()
    assert "context" in inputs.keys()
    assert "task" in inputs.keys()
    labels = examples[0].labels()
    assert "response" in labels.keys()


def test_load_all_splits(tmp_path):
    agent_dir = tmp_path / "security-analyst"
    agent_dir.mkdir()
    row = json.dumps({"inputs": {"context": "c", "task": "t"}, "outputs": {"response": "r"}})
    (agent_dir / "train.jsonl").write_text(row + "\n" + row + "\n")
    (agent_dir / "dev.jsonl").write_text(row + "\n")
    (agent_dir / "test.jsonl").write_text(row + "\n")

    splits = load_all_splits(agent_dir)
    assert len(splits["train"]) == 2
    assert len(splits["dev"]) == 1
    assert len(splits["test"]) == 1


def test_load_all_splits_missing_file(tmp_path):
    agent_dir = tmp_path / "some-agent"
    agent_dir.mkdir()
    row = json.dumps({"inputs": {"context": "c", "task": "t"}, "outputs": {"response": "r"}})
    (agent_dir / "train.jsonl").write_text(row + "\n")

    splits = load_all_splits(agent_dir)
    assert len(splits["train"]) == 1
    assert len(splits["dev"]) == 0
    assert len(splits["test"]) == 0


def test_get_dataset_status(tmp_path):
    agent_dir = tmp_path / "security-analyst"
    agent_dir.mkdir()
    row = json.dumps({"inputs": {"context": "c", "task": "t"}, "outputs": {"response": "r"}})
    (agent_dir / "train.jsonl").write_text(row + "\n" + row + "\n")
    (agent_dir / "dev.jsonl").write_text(row + "\n")

    info = get_dataset_status(tmp_path, "security-analyst")
    assert info.agent_id == "security-analyst"
    assert info.train_count == 2
    assert info.dev_count == 1
    assert info.test_count == 0
