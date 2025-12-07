import importlib
import math
import dataclasses
from pathlib import Path
import flax.nnx as nnx
import jax.numpy as jnp
from flax import struct
from simple_parsing import ArgumentParser
from typing import Type

import orbax.checkpoint as ocp

import tqdm
import wandb

from dataset.ogbench_singletask import load_env_and_datasets, Dataset
from utils.jax import save_state
from utils.logging import init_wandb_run, log_metrics, log_artifact_dir


@struct.dataclass
class TrainConfig:
    learning_rate: float  # JIT 내부에서 변경/추적 가능 (Leaf)
    batch_size: int  # JIT 내부에서 사용 가능 (Leaf)
    num_updates: int
    eval_interval: int
    model_save_interval: int
    train_log_interval: int
    seed: int
    wandb_log: bool
    model_save: bool
    project_name: str = struct.field(pytree_node=False)
    algorithm_module: str = struct.field(pytree_node=False)
    algorithm_class: str = struct.field(pytree_node=False)
    dataset_name: str = struct.field(pytree_node=False)


def load_algorithm(module_path: str, class_name: str) -> Type:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f"{class_name} not found in {module_path}")
    return cls


def save(
    step: int,
    checkpointer: ocp.StandardCheckpointer,
    save_root_dir: Path,
    algorithm: nnx.Module,
    wandb_run: wandb.Run | None = None,
):
    save_dir = save_root_dir / f"{step}"
    Path.mkdir(save_dir, parents=True, exist_ok=True)
    save_state(checkpointer=checkpointer, obj=algorithm, path=save_dir)

    if wandb_run is not None:
        log_artifact_dir(
            wandb_run=wandb_run,
            local_dir=save_dir,
            name=f"{save_root_dir.name}_{step}",
            aliases=[f"{step}"],
            metadata={"step": step},
        )


def _main(config: TrainConfig):
    env, eval_env, train_dataset, val_dataset = load_env_and_datasets(
        dataset_name=config.dataset_name
    )
    algo_cls = load_algorithm(config.algorithm_module, config.algorithm_class)

    # Build example shapes for algorithm init
    example_observation = jnp.asarray(train_dataset["observations"][0])
    example_action = jnp.asarray(train_dataset["actions"][0])
    rngs = nnx.Rngs(config.seed)
    algo = algo_cls(
        config=config,
        example_observation=example_observation,
        example_action=example_action,
        rngs=rngs,
    )

    wandb_run = None
    if config.wandb_log:
        wandb_run = init_wandb_run(
            project=config.project_name,
            config=dataclasses.asdict(config),
            login=False,
        )

    save_root_dir: Path | None = None
    if wandb_run and config.model_save:
        name = wandb_run.name
        id = wandb_run.id
        save_root_dir = Path.cwd() / f"ckpt/{config.algorithm_module}/{name}_{id}"
        Path.mkdir(save_root_dir, parents=True, exist_ok=True)

    step_length = 1
    step_iterator = tqdm.tqdm(
        range(0, config.num_updates, step_length), desc="Training", dynamic_ncols=True
    )
    for cur_step in step_iterator:
        # logging
        if wandb_run is not None and cur_step % config.train_log_interval == 0:
            train_log = {f"train/{k}": float(v) for k, v in train_metrics.items()}
            log_metrics(wandb_run=wandb_run, step=cur_step, train_dict=train_log)

        # train
        batch = train_dataset.sample(config.batch_size)
        train_metrics = algo.train_step(batch)

    if wandb_run is not None:
        wandb_run.finish()


def main():
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="config")
    args = parser.parse_args()
    config = args.config

    _main(config)


if __name__ == "__main__":
    main()
