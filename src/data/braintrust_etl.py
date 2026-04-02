"""Extract training data from Braintrust logs stored in S3.

Handles the full pipeline:
1. Download .jsonl.gz files from S3
2. Parse spans and group by trace (root_span_id)
3. Identify agent boundaries and filter sub-agent spans
4. Extract input/output pairs per agent type
5. Write to datasets/ as train/dev/test JSONL splits
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Agents that appear as sub-agent boundary spans inside other traces.
# When extracting for a parent agent, these spans (and their children) are excluded.
SUB_AGENT_NAMES = {
    "security-analyst",
    "security-analyst:recovery",
    "finding-verifier",
    "finding-dedup-checker",
    "memory-extractor",
    "attack-surface-mapper",
    "attack-surface-mapper:recovery",
    "diff-analyzer",
    "generateObject",
}


def sync_s3_logs(
    bucket: str,
    prefix: str = "logs/",
    local_dir: Path = Path("/tmp/bt-logs"),
    profile: str | None = None,
) -> Path:
    """Download Braintrust log files from S3.

    Returns the local directory containing the downloaded files.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync",
        f"s3://{bucket}/{prefix}",
        str(local_dir),
    ]
    if profile:
        cmd.extend(["--profile", profile])

    logger.info(f"Syncing s3://{bucket}/{prefix} → {local_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed: {result.stderr}")
    logger.info("S3 sync complete")
    return local_dir


def load_all_spans(logs_dir: Path) -> list[dict]:
    """Load all spans from all .jsonl.gz files in the directory tree."""
    spans = []
    for dirpath, _, filenames in sorted(os.walk(logs_dir)):
        for fname in sorted(filenames):
            if not fname.endswith(".jsonl.gz"):
                continue
            fpath = Path(dirpath) / fname
            with gzip.open(fpath, "rt") as f:
                for line in f:
                    try:
                        spans.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    logger.info(f"Loaded {len(spans):,} spans from {logs_dir}")
    return spans


# Need os for os.walk
import os


def build_indices(spans: list[dict]) -> tuple[dict, dict]:
    """Build lookup indices for spans.

    Returns:
        by_root: {root_span_id: [spans]}
        by_sid: {span_id: span}
    """
    by_root: dict[str, list[dict]] = {}
    by_sid: dict[str, dict] = {}
    for s in spans:
        rid = s.get("root_span_id")
        if rid:
            by_root.setdefault(rid, []).append(s)
        by_sid[s["span_id"]] = s
    return by_root, by_sid


def find_root_spans(spans: list[dict], agent_name: str) -> list[dict]:
    """Find all root spans for a given agent name."""
    roots = []
    for s in spans:
        if not s.get("is_root"):
            continue
        attrs = s.get("span_attributes") or {}
        if attrs.get("name") == agent_name:
            roots.append(s)
    logger.info(f"Found {len(roots)} root traces for '{agent_name}'")
    return roots


def get_sub_agent_sids(trace: list[dict]) -> set[str]:
    """Get span_ids of all sub-agent boundary spans in a trace."""
    sids = set()
    for s in trace:
        attrs = s.get("span_attributes") or {}
        name = attrs.get("name", "")
        stype = attrs.get("type", "")
        if name in SUB_AGENT_NAMES and stype not in ("llm", "tool"):
            sids.add(s["span_id"])
    return sids


def get_parent_agent(
    span: dict,
    sub_agent_sids: set[str],
    by_sid: dict[str, dict],
) -> str | None:
    """Walk up the parent chain to find if this span belongs to a sub-agent.

    Returns the sub-agent name if found, None if it belongs to the root agent.
    """
    current = span
    visited: set[str] = set()
    while current:
        parents = current.get("span_parents", [])
        if not parents:
            break
        pid = parents[0]
        if pid in visited:
            break
        visited.add(pid)
        if pid in sub_agent_sids:
            parent = by_sid.get(pid)
            if parent:
                return (parent.get("span_attributes") or {}).get("name", "sub-agent")
            return "sub-agent"
        current = by_sid.get(pid)
    return None


def filter_own_llm_spans(
    trace: list[dict],
    by_sid: dict[str, dict],
) -> list[dict]:
    """Get only the LLM spans that belong to the root agent, not sub-agents."""
    sub_sids = get_sub_agent_sids(trace)

    own = []
    for s in trace:
        attrs = s.get("span_attributes") or {}
        if attrs.get("type") != "llm":
            continue
        if not s.get("input") or not isinstance(s["input"], dict):
            continue
        parent = get_parent_agent(s, sub_sids, by_sid)
        if parent is None:
            own.append(s)

    own.sort(key=lambda s: s.get("created", ""))
    return own


def extract_user_message(llm_spans: list[dict]) -> str | None:
    """Extract the first user message from a list of LLM spans."""
    for llm in llm_spans:
        messages = llm.get("input", {}).get("messages", [])
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 10:
                return content
            if isinstance(content, list):
                texts = [
                    b.get("text", "") for b in content if b.get("type") == "text"
                ]
                text = " ".join(texts).strip()
                if len(text) > 10:
                    return text
    return None


def extract_last_response(llm_spans: list[dict], min_length: int = 50) -> str | None:
    """Extract the last meaningful text response from LLM spans."""
    for s in reversed(llm_spans):
        out = s.get("output")
        if (
            out
            and isinstance(out, dict)
            and out.get("text")
            and len(out["text"]) >= min_length
        ):
            return out["text"]
    return None


def extract_context(llm_spans: list[dict]) -> str:
    """Extract context from the conversation history."""
    if not llm_spans:
        return "New conversation"

    first = llm_spans[0]
    messages = first.get("input", {}).get("messages", [])

    # Try instructions first
    instructions = first.get("input", {}).get("instructions", {})
    if isinstance(instructions, dict):
        content = instructions.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    return block["text"][:2000]
        elif isinstance(content, str) and content:
            return content[:2000]

    # Fall back to prior conversation messages
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            parts.append(f"[{role}]: {content[:500]}")

    if parts:
        return "\n".join(parts[-4:])[:2000]

    return "New conversation"


def extract_examples_for_agent(
    agent_name: str,
    spans: list[dict],
    by_root: dict[str, list[dict]],
    by_sid: dict[str, dict],
    max_task_len: int = 3000,
    max_response_len: int = 5000,
    max_context_len: int = 2000,
) -> list[dict]:
    """Extract training examples for a specific agent from all traces.

    For agents that delegate to sub-agents (e.g., architecture-scanner),
    only the agent's own LLM calls are used, not the sub-agents'.

    For leaf agents (e.g., chat-assistant), all LLM calls in the trace
    belong to that agent.
    """
    # First try root spans (agents that are top-level trace owners)
    roots = find_root_spans(spans, agent_name)

    # Also find this agent as a sub-agent boundary span inside other traces
    # This handles agents like attack-surface-mapper, security-analyst, etc.
    # that mostly appear as children of architecture-scanner.
    # We always collect these — if root extraction yields nothing, we fall back.
    sub_agent_entries = []
    if True:
        for s in spans:
            attrs = s.get("span_attributes") or {}
            name = attrs.get("name", "")
            stype = attrs.get("type", "")
            if name == agent_name and stype not in ("llm", "tool") and not s.get("is_root"):
                sub_agent_entries.append(s)
        if sub_agent_entries:
            logger.info(
                f"No root traces for '{agent_name}', "
                f"found {len(sub_agent_entries)} sub-agent boundary spans"
            )

    examples = []
    seen_rids: set[str] = set()
    skipped = {
        "duplicate": 0,
        "no_llm": 0,
        "no_input": 0,
        "no_response": 0,
    }

    for root in roots:
        rid = root["root_span_id"]
        if rid in seen_rids:
            skipped["duplicate"] += 1
            continue
        seen_rids.add(rid)

        trace = by_root.get(rid, [])
        own_llm = filter_own_llm_spans(trace, by_sid)

        if not own_llm:
            skipped["no_llm"] += 1
            continue

        user_msg = extract_user_message(own_llm)
        if not user_msg:
            skipped["no_input"] += 1
            continue

        response = extract_last_response(own_llm)
        if not response:
            skipped["no_response"] += 1
            continue

        context = extract_context(own_llm)

        # Gather metadata
        sub_sids = get_sub_agent_sids(trace)
        sub_counts: dict[str, int] = {}
        for sid in sub_sids:
            s = by_sid.get(sid)
            if s:
                name = (s.get("span_attributes") or {}).get("name", "unknown")
                sub_counts[name] = sub_counts.get(name, 0) + 1

        example = {
            "inputs": {
                "context": context[:max_context_len],
                "task": user_msg[:max_task_len],
            },
            "outputs": {
                "response": response[:max_response_len],
            },
            "metadata": {
                "source": "braintrust",
                "agent": agent_name,
                "root_span_id": rid,
                "created": root.get("created", ""),
                "total_spans": len(trace),
                "own_llm_calls": len(own_llm),
                "sub_agents_spawned": sub_counts,
            },
        }
        examples.append(example)

    # Extract from sub-agent boundary spans (when agent isn't a root trace owner)
    for entry in sub_agent_entries:
        entry_sid = entry["span_id"]
        rid = entry.get("root_span_id", "")
        trace = by_root.get(rid, [])

        # Find LLM spans that are children of this sub-agent boundary
        child_llm = []
        for s in trace:
            attrs = s.get("span_attributes") or {}
            if attrs.get("type") != "llm":
                continue
            if not s.get("input") or not isinstance(s["input"], dict):
                continue
            # Walk up to see if this span is under our boundary
            current = s
            visited_p: set[str] = set()
            found = False
            while current:
                parents = current.get("span_parents", [])
                if not parents:
                    break
                pid = parents[0]
                if pid in visited_p:
                    break
                visited_p.add(pid)
                if pid == entry_sid:
                    found = True
                    break
                current = by_sid.get(pid)
            if found:
                child_llm.append(s)

        child_llm.sort(key=lambda s: s.get("created", ""))
        if not child_llm:
            continue

        user_msg = extract_user_message(child_llm)
        if not user_msg:
            continue

        response = extract_last_response(child_llm)
        if not response:
            continue

        context = extract_context(child_llm)

        example = {
            "inputs": {
                "context": context[:max_context_len],
                "task": user_msg[:max_task_len],
            },
            "outputs": {
                "response": response[:max_response_len],
            },
            "metadata": {
                "source": "braintrust",
                "agent": agent_name,
                "root_span_id": rid,
                "boundary_span_id": entry_sid,
                "created": entry.get("created", ""),
                "own_llm_calls": len(child_llm),
            },
        }
        examples.append(example)

    logger.info(
        f"Extracted {len(examples)} examples for '{agent_name}', "
        f"skipped: {json.dumps(skipped)}"
    )
    return examples


def split_and_write(
    examples: list[dict],
    output_dir: Path,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42,
    min_dev: int = 2,
    min_test: int = 2,
) -> dict[str, int]:
    """Shuffle, split, and write examples to train/dev/test JSONL files.

    Returns counts per split.
    """
    random.seed(seed)
    random.shuffle(examples)

    n = len(examples)
    if n < min_dev + min_test + 1:
        logger.warning(
            f"Only {n} examples — putting all in train, "
            f"need at least {min_dev + min_test + 1} for splits"
        )
        train, dev, test = examples, [], []
    else:
        # Ensure dev and test get at least min_dev/min_test examples
        test_count = max(int(n * (1 - train_ratio - dev_ratio)), min_test)
        dev_count = max(int(n * dev_ratio), min_dev)
        train_count = n - dev_count - test_count

        train = examples[:train_count]
        dev = examples[train_count : train_count + dev_count]
        test = examples[train_count + dev_count :]

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        counts[split_name] = len(split_data)

    logger.info(f"Written to {output_dir}: {counts}")
    return counts


def extract_all_agents(
    logs_dir: Path,
    datasets_dir: Path,
    agents: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Full ETL pipeline: load spans, extract examples for each agent, write splits.

    Args:
        logs_dir: Directory containing downloaded .jsonl.gz files
        datasets_dir: Output directory for datasets (datasets/{agent}/)
        agents: List of agent names to extract. If None, extracts all found.

    Returns:
        {agent_name: {train: N, dev: N, test: N}}
    """
    spans = load_all_spans(logs_dir)
    by_root, by_sid = build_indices(spans)

    if agents is None:
        # Discover all agent types: both root trace owners and sub-agent boundaries
        agent_counts: dict[str, int] = {}
        for s in spans:
            if s.get("is_root"):
                name = (s.get("span_attributes") or {}).get("name", "")
                if name and name not in ("test-span", "generateObject"):
                    agent_counts[name] = agent_counts.get(name, 0) + 1

        # Also discover sub-agent-only types
        sub_agent_counts: dict[str, int] = {}
        for s in spans:
            attrs = s.get("span_attributes") or {}
            name = attrs.get("name", "")
            stype = attrs.get("type", "")
            if (
                name
                and stype not in ("llm", "tool")
                and not s.get("is_root")
                and name not in ("generateObject",)
                and ":" not in name  # Skip recovery variants
            ):
                sub_agent_counts[name] = sub_agent_counts.get(name, 0) + 1

        all_agents = set(agent_counts.keys()) | set(sub_agent_counts.keys())
        agents = sorted(all_agents)
        logger.info(f"Discovered root agents: {agent_counts}")
        logger.info(f"Discovered sub-agents: {sub_agent_counts}")

    results = {}
    for agent_name in agents:
        examples = extract_examples_for_agent(
            agent_name, spans, by_root, by_sid
        )
        if not examples:
            logger.warning(f"No examples extracted for '{agent_name}', skipping")
            continue

        output_dir = datasets_dir / agent_name
        counts = split_and_write(examples, output_dir)
        results[agent_name] = counts

    return results
