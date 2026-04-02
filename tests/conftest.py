from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_agent_markdown() -> str:
    return """\
---
id: security-analyst
description: Expert cybersecurity researcher analyzing code for security issues
model: sonnet
max_output_tokens: 32768
max_iterations: 30
tools:
  - list_dir
  - read_file
  - ripgrep
  - bash
recovery_iterations: 5
---

<role>
You are an expert cybersecurity researcher.
</role>

<process>
1. Read the files
2. Trace data flows
3. Report findings
</process>
"""


@pytest.fixture
def sample_skill_markdown() -> str:
    return """\
---
id: ssrf-audit
description: How to find SSRF vulnerabilities
category: analysis
---

## Auditing for SSRF

Search for outbound HTTP calls with dynamic URLs.
"""


@pytest.fixture
def sample_agent_no_tools_markdown() -> str:
    return """\
---
id: conversation-compactor
description: Summarizes long conversations
model: haiku
max_output_tokens: 8192
max_iterations: 5
tools:
---

You are a conversation summarizer.
"""
