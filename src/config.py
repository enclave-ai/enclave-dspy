from __future__ import annotations

import os
from pathlib import Path

import dspy
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "optimization_config.yaml"


class AgentOptConfig(BaseModel):
    optimizers: list[str] = ["bootstrap", "mipro_light"]
    metric_weights: dict[str, float] = {}
    max_bootstrapped_demos: int | None = None
    max_labeled_demos: int | None = None


class OptimizationConfig(BaseModel):
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 8
    improvement_threshold: float = 0.05
    teacher_model: str = "anthropic/claude-opus-4-6"
    target_model: str = "anthropic/claude-sonnet-4-6"


class Config(BaseModel):
    defaults: OptimizationConfig = OptimizationConfig()
    agents: dict[str, AgentOptConfig] = {}

    def get_agent_config(self, agent_id: str) -> AgentOptConfig:
        return self.agents.get(agent_id, AgentOptConfig())


def load_config(path: Path | None = None) -> Config:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return Config()
    raw = yaml.safe_load(config_path.read_text())
    defaults = OptimizationConfig(**raw.get("defaults", {}))
    agents = {
        k: AgentOptConfig(**v) for k, v in raw.get("agents", {}).items()
    }
    return Config(defaults=defaults, agents=agents)


def configure_dspy(model: str | None = None) -> dspy.LM:
    model_name = model or os.getenv("DSPY_MODEL", "anthropic/claude-sonnet-4-6")
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)
    return lm


def get_enclave_repo_path() -> Path:
    path = os.getenv("ENCLAVE_REPO_PATH", "../enclave")
    return Path(path).resolve()


def get_enclave_agents_dir() -> Path:
    return get_enclave_repo_path() / "packages" / "ai-client" / "src" / "definitions" / "agents"


def get_enclave_skills_dir() -> Path:
    return get_enclave_repo_path() / "packages" / "ai-client" / "src" / "definitions" / "skills"
