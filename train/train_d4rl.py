import dataclasses
import math
import os
import random
import sys
from pathlib import Path
import flax.nnx as nnx
import gym.vector as vector
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from tqdm.auto import tqdm

import d4rl
import wandb
from algorithms import cql


from dataset.antmaze_v2 import get_dataset_file_path, load_dataset_from_file
from utils.config import generate_experiment_hash
from utils.jax import save_state
from utils.logging import init_wandb_run, log_artifact_dir, log_metrics

print("OS getcwd", os.getcwd())
print("SYS PATH:", sys.path)

# Mirror commonly-tuned run flags from the algorithm module for convenience
wandb_log = cql.wandb_log
wandb_notes = cql.wandb_notes
wandb_tags = cql.wandb_tags
wandb_project = cql.wandb_project
# wandb_group_id = cql.wandb_group_id
machine_name = cql.machine_name
train_log_interval = cql.train_log_interval
eval_interval = cql.eval_interval
model_save = cql.model_save
model_save_interval = cql.model_save_interval
debug = cql.debug
dataset_dir = cql.dataset_dir


def _load_local_dataset(env_id: str) -> dict[str, np.ndarray]:
    """Load an offline dataset strictly from the local filesystem."""

    if dataset_dir is None:
        raise RuntimeError(
            "Set 'dataset_dir' in algorithms/cql.py to your local dataset directory before running experiments."
        )
    file_path = get_dataset_file_path(env_id=env_id, dataset_dir=dataset_dir)
    return load_dataset_from_file(file_path=file_path, env_id=env_id)


def prepare_training(config: cql.Config):
    rngs = nnx.Rngs(params=config.seed, random=config.seed + 1)
    env = vector.make(config.dataset, num_envs=config.eval_workers, asynchronous=False)

    dataset_dict = _load_local_dataset(config.dataset)
    dataset = cql.Transition(
        obs=jnp.array(dataset_dict["observations"]),
        action=jnp.array(dataset_dict["actions"]),
        reward=jnp.array(dataset_dict["rewards"]),
        next_obs=jnp.array(dataset_dict["next_observations"]),
        done=jnp.array(dataset_dict["terminals"]),
    )
    return rngs, env, dataset


def evaluate_policy(
    config: cql.Config,
    env: vector.VectorEnv,
    actor: cql.TanhGaussianPolicy,
    num_episodes: int,
):
    # Run episodes in the vectorized env using the deterministic policy (tanh(mean))
    obs = env.reset()
    episode_returns = []
    cur_returns = np.zeros(env.num_envs, dtype=float)
    while len(episode_returns) < num_episodes:
        obs_j = jnp.array(obs)
        x = actor.layer(obs_j)
        mean = actor.mean(x)
        action_j = jnp.tanh(mean)
        action = np.asarray(action_j)
        obs, reward, done, _ = env.step(action)
        cur_returns += reward
        for i, d in enumerate(done):
            if d:
                episode_returns.append(float(cur_returns[i]))
                cur_returns[i] = 0.0

    scores = d4rl.get_normalized_score(config.dataset, np.array(episode_returns)) * 100

    return cql.EvalMetrics(
        avg_return=float(np.mean(episode_returns)),
        score=scores.mean(),
        score_std=scores.std(),
    )


def evaluate(
    config: cql.Config,
    models: cql.Models,
    env: vector.VectorEnv,
):
    eval_metrics = evaluate_policy(config, env, models.actor, num_episodes=env.num_envs)
    log_data = {
        f"valid/{k}": float(v)
        for k, v in eval_metrics._asdict().items()
        if v is not None
    }
    return log_data


def save(
    step: int,
    save_root_dir: Path,
    checkpointer: ocp.StandardCheckpointer,
    models: cql.Models,
    opts: cql.Opts,
    wandb_run: wandb.Run | None = None,
):
    cur_step_dir = save_root_dir / f"{step}"
    for name, model in models._asdict().items():
        cur_save_dir = cur_step_dir / "model" / name
        save_state(checkpointer=checkpointer, obj=model, path=cur_save_dir)

    for name, opt in opts._asdict().items():
        cur_save_dir = cur_step_dir / "optimizer" / name
        save_state(checkpointer=checkpointer, obj=opt, path=cur_save_dir)

    if wandb_run is not None:
        name = wandb_run.name if wandb_run.name else wandb_run.id
        log_artifact_dir(
            wandb_run=wandb_run,
            local_dir=cur_step_dir,
            name=name,
            aliases=[f"{step}"],
            metadata={"step": step},
        )


def log_train(metrics: cql.Metrics):
    log_data = {
        f"train/{k}": float(v) for k, v in metrics._asdict().items() if v is not None
    }
    return log_data


def log_obj_stats(models: cql.Models, opts: cql.Opts):
    stats = {}
    for name, model in models._asdict().items():
        mean, std = cql.get_all_array_stats(model)
        stats[f"params/{name}_mean"], stats[f"params/{name}_std"] = (
            mean,
            std,
        )

    for name, opt in opts._asdict().items():
        (
            stats[f"opts/{name}_mean"],
            stats[f"opts/{name}_std"],
        ) = cql.get_all_array_stats(opt)
    return stats


def extract_experiment_metadata(config: cql.Config):
    global wandb_tags
    global machine_name
    wandb_config = dataclasses.asdict(config)
    wandb_config["metadata"] = {
        "machine_name": machine_name,
    }
    return wandb_config


def main(wandb_run: wandb.Run | None = None):
    run_params = {}
    if wandb_run is not None:
        valid_keys = {f.name for f in dataclasses.fields(cql.Config)}
        run_params = {k: v for k, v in wandb_run.config.items() if k in valid_keys}

    config = cql.Config(**run_params)
    wandb_config = extract_experiment_metadata(config=config)
    # print(f"Config type: {type(config)}")
    # print(f"wandb_config: {wandb_config}")
    # print(f"config: {config}")

    if wandb_log:
        if wandb_run is None:
            wandb_run = init_wandb_run(
                project=wandb_project,
                config=wandb_config,
                notes=wandb_notes,
                tags=wandb_tags,
                group=None,
            )
        else:
            wandb_run.config.update(wandb_config)

    save_root_dir: Path | None = None
    if wandb_run and model_save:
        name = wandb_run.name if wandb_run.name else wandb_run.id
        save_root_dir = Path.cwd() / f"ckpt/cql/{name}"
        Path.mkdir(save_root_dir, exist_ok=True, parents=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    checkpointer = ocp.StandardCheckpointer()

    rngs, env, dataset = prepare_training(config)
    env.seed(config.seed)
    actor_net, q_net, q_target_net, log_alpha, log_alpha_prime = cql.initialize_network(
        config, rngs, env
    )

    actor_opt = nnx.Optimizer(actor_net, optax.adam(learning_rate=config.actor_lr))
    q_opt = nnx.Optimizer(q_net, optax.adam(learning_rate=config.q_lr))
    log_alpha_opt = nnx.Optimizer(log_alpha, optax.adam(learning_rate=config.actor_lr))
    log_alpha_prime_opt = (
        nnx.Optimizer(log_alpha_prime, optax.adam(learning_rate=config.q_lr))
        if log_alpha_prime is not None
        else None
    )

    models = cql.Models(
        actor=actor_net,
        vec_q=q_net,
        vec_q_target=q_target_net,
        log_alpha=log_alpha,
        log_alpha_prime=log_alpha_prime,
    )
    opts = cql.Opts(
        actor=actor_opt,
        q=q_opt,
        log_alpha=log_alpha_opt,
        log_alpha_prime=log_alpha_prime_opt,
    )
    len_dataset = len(dataset.obs)
    step_length = math.gcd(eval_interval, model_save_interval, train_log_interval)

    step_iterator = tqdm(
        range(0, config.num_updates, step_length),
        desc="Training",
        dynamic_ncols=True,
    )

    for cur_step in step_iterator:
        train_statistics, eval_statistics, state_statistics = None, None, None
        if cur_step % eval_interval == 0:
            eval_statistics = evaluate(
                config=config,
                models=models,
                env=env,
            )

        if cur_step % model_save_interval == 0 and save_root_dir:
            save(
                step=cur_step,
                save_root_dir=save_root_dir,
                checkpointer=checkpointer,
                models=models,
                opts=opts,
                wandb_run=wandb_run,
            )

        print(f"Config type: {type(config)}")
        (rngs, models, opts), metrics = cql.train_multiple_steps(
            carry=(rngs, models, opts),
            dataset=dataset,
            config=config,
            len_dataset=len_dataset,
            length=step_length,
        )

        if cur_step % train_log_interval == 0 and wandb_run is not None:
            train_statistics = log_train(metrics=metrics)
            state_statistics = log_obj_stats(models=models, opts=opts)

        if wandb_run is not None:
            log_metrics(
                wandb_run=wandb_run,
                step=cur_step,
                train_dict=train_statistics,
                eval_dict=eval_statistics,
                state_dict=state_statistics,
            )

    eval_statistic = evaluate(
        config=config,
        models=models,
        env=env,
    )
    if wandb_run is not None:
        log_metrics(
            wandb_run=wandb_run,
            step=config.num_updates,
            eval_dict=eval_statistic,
        )
    if not debug and save_root_dir:
        save(
            step=config.num_updates,
            save_root_dir=save_root_dir,
            checkpointer=checkpointer,
            models=models,
            opts=opts,
            wandb_run=wandb_run,
        )


if __name__ == "__main__":
    main()
