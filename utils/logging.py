import os
from pathlib import Path
from typing import Any, Literal

import wandb


def init_wandb_run(
    *,
    project: str,
    config: dict[str, Any],
    notes: str | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    resume: Literal["allow", "must", "never", "auto"] = "allow",
    save_code: bool = True,
    disable_git: bool = False,
    api_key_env: str = "WANDB_API_KEY",
    login: bool = True,
):
    """Initialize a Weights & Biases run with common defaults."""

    if login:
        wandb.login(key=os.environ.get(api_key_env))

    return wandb.init(
        project=project,
        config=config,
        notes=notes,
        tags=tags,
        settings=wandb.Settings(
            resume=resume, save_code=save_code, disable_git=disable_git
        ),
        group=group,
    )


def _clean_dict(d: dict | None):
    if d is None:
        return {}
    return {k: v for k, v in d.items() if v is not None}


def log_metrics(
    wandb_run: wandb.Run, *, step: int, train_dict=None, eval_dict=None, state_dict=None
):
    """Log grouped metrics to W&B at a given step."""

    statistics: dict = {}
    statistics.update(_clean_dict(train_dict))
    statistics.update(_clean_dict(eval_dict))
    statistics.update(_clean_dict(state_dict))

    if statistics:
        wandb_run.log(statistics, step=step)


def log_artifact_dir(
    wandb_run: wandb.Run,
    *,
    local_dir: Path | str,
    name: str,
    aliases: list[str] | None = None,
    metadata: dict | None = None,
):
    """Upload a directory as a W&B artifact."""

    artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
    artifact.add_dir(local_path=str(local_dir))
    wandb_run.log_artifact(artifact, aliases=aliases or [])
