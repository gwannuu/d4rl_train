import dataclasses
import getpass
import os
import random
import socket
from collections import namedtuple
import math
from dataclasses import dataclass
from pathlib import Path

from utils.jax import sym

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import d4rl
import distrax
import flax.nnx as nnx
import gym
import gym.vector as vector
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.nnx.nn.initializers import constant

import wandb
from utils.config import generate_experiment_hash, get_git_hash
from utils.jax import nnx_conditional_jit, restore_state, save_state
from dataset.antmaze_v2 import dataset as antmaze_datasets

wandb_log: bool = True
wandb_notes: str = "Add lagrangian dual & importance sampling"
wandb_tags: list[str] = ["cql"]
wandb_project: str = "d4rl_train"
wandb_group_id: str | None = None
machine_name: str = os.environ["MACHINE_NAME"]
train_log_interval: int = 10_000
eval_interval: int = 50_000
model_save: bool = True
model_save_interval: int = 500_000
debug: bool = False


@dataclass(frozen=True)
class Config:
    # Metadata
    dataset: str = antmaze_datasets[0]

    # Train
    cql_lagrange: bool = False
    cql_importance_sampling: bool = True
    seed: int = 4212
    num_critics: int = 2
    num_updates: int = 1_000_000
    polyak_step_size: float = 0.005
    batch_size: int = 256
    gamma: float = 0.99
    cql_temperature: float = 1.0
    cql_min_q_weight: float = 5.0
    actor_lr: float = 3e-5
    q_lr: float = 1e-4
    alpha_lr: float = 3e-4
    num_action_sample: int = 10
    cql_target_gap_expansion: float = 5.0

    # Eval
    eval_workers: int = 8


Models = namedtuple("Models", "actor vec_q vec_q_target alpha alpha_prime")
Opts = namedtuple("Opts", "actor q alpha alpha_prime")
Transition = namedtuple("Transition", "obs action reward next_obs done")

Metrics = namedtuple(
    "Metrics",
    "critic_loss gap_mean gap_residual actor_loss alpha_loss alpha_prime_loss entropy alpha alpha_prime q_min q_std",
)
EvalMetrics = namedtuple("EvalMetrics", "avg_return score score_std")


class LogScalar(nnx.Module):
    def __init__(self, /, *, ent_coef_init: float = 1.0):
        self.log_ent_coef = nnx.Param(jnp.log(ent_coef_init))

    def __call__(self):
        return self.log_ent_coef

    def exp(self):
        return jax.lax.stop_gradient(jnp.exp(self.log_ent_coef.value))


class SoftQNetwork(nnx.Module):
    def __init__(
        self,
        /,
        *,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        rngs: nnx.Rngs,
    ):
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(
                nnx.Linear(
                    in_features=in_dim,
                    out_features=out_dim,
                    bias_init=constant(0.1),
                    rngs=rngs,
                )
            )
            layers.append(nnx.relu)
            in_dim = out_dim

        self.net = nnx.Sequential(*layers)
        self.q_layer = nnx.Linear(
            in_features=in_dim,
            out_features=output_dim,
            # https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/misc/mlp.py#L25
            # https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/sac/policies.py#L62
            kernel_init=sym(3e-3),
            bias_init=sym(3e-3),
            rngs=rngs,
        )

    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = self.net(x)
        q = self.q_layer(x)
        return q.squeeze(-1)


class VectorQ(nnx.Module):
    def __init__(
        self,
        /,
        *,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        num_critics: int,
        rngs: nnx.Rngs,
    ):
        self.critics = [
            SoftQNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                rngs=rngs,
            )
            for _ in range(num_critics)
        ]

    def __call__(self, obs, action):
        q_values_list = [critic(obs, action) for critic in self.critics]
        q_values = jnp.stack(q_values_list, axis=-1)

        return q_values


class TanhGaussianPolicy(nnx.Module):
    def __init__(
        self,
        /,
        *,
        input_dim: int,
        hidden_dims: list[int],
        num_actions: int,
        log_std_max: float = 2.0,
        log_std_min: float = -5.0,
        rngs: nnx.Rngs,
    ):
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(
                nnx.Linear(
                    in_features=in_dim,
                    out_features=out_dim,
                    bias_init=constant(0.1),
                    rngs=rngs,
                )
            )
            layers.append(nnx.relu)
            in_dim = out_dim
        self.layer = nnx.Sequential(*layers)

        self.mean = nnx.Linear(
            in_features=in_dim,
            out_features=num_actions,
            bias_init=sym(1e-3),
            rngs=rngs,
        )
        self.log_std = nnx.Linear(
            in_features=in_dim,
            out_features=num_actions,
            bias_init=sym(1e-3),
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.layer(x)
        mean, log_std = self.mean(x), self.log_std(x)

        std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
        pi = distrax.Transformed(
            distrax.Normal(mean, std),
            distrax.Tanh(),
        )
        return pi


def restore_ckpt(wandb_run_id, models, opts, wandb_run, checkpointer):
    print(f"--- Resuming W&B Run '{wandb_run_id}' ... ---")
    artifact = wandb_run.use_artifact(f"{wandb_run_id}:latest")
    step = artifact.metadata["step"]
    ckpt_dir = artifact.download()

    restored_models = {}
    for name, model in models._asdict().items():
        restored = restore_state(
            checkpointer=checkpointer,
            obj=model,
            dir=Path(ckpt_dir) / "model" / name,
        )
        restored_models[name] = restored

    restored_opts = {}
    for name, opt in opts._asdict().items():
        restored = restore_state(
            checkpointer=checkpointer,
            obj=opt,
            dir=Path(ckpt_dir) / "optimizer" / name,
        )
        restored_opts[name] = restored

    return (step, Models(**restored_models), Opts(**restored_opts))


def prepare_training(config: Config):
    rngs = nnx.Rngs(params=config.seed, random=config.seed + 1)
    env = vector.make(config.dataset, num_envs=config.eval_workers, asynchronous=False)
    dataset = d4rl.qlearning_dataset(gym.make(config.dataset))
    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )
    return rngs, env, dataset


def initialize_network(config: Config, rngs: nnx.Rngs, env: vector.VectorEnv):
    num_actions = env.single_action_space.shape[0]
    actor_net = TanhGaussianPolicy(
        num_actions=num_actions,
        input_dim=env.single_observation_space.shape[0],
        hidden_dims=[256, 256],
        rngs=rngs,
    )

    # https://github.com/young-geng/JaxCQL/blob/bac4299194bd6ae2bc7db9034fd1a31ac43a30d7/JaxCQL/model.py#L82
    # https://github.com/hyeon1996/EPQ/blob/c56847215d748b937d9c6952cfe7a481363163a0/epq_main.py#L113
    q_net = VectorQ(
        num_critics=config.num_critics,
        input_dim=env.single_observation_space.shape[0] + num_actions,
        hidden_dims=[256, 256],
        output_dim=1,
        rngs=rngs,
    )
    q_target_net = nnx.clone(q_net)
    alpha = LogScalar()
    alpha_prime = LogScalar() if config.cql_lagrange else None
    return actor_net, q_net, q_target_net, alpha, alpha_prime


def get_all_array_stats(
    obj: nnx.Module | nnx.Optimizer,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    state = nnx.split(obj)[1]
    leaves = jax.tree_util.tree_leaves(state)
    array_leaves = [x for x in leaves if isinstance(x, jnp.ndarray)]

    if not array_leaves:
        return jnp.array(jnp.nan), jnp.array(jnp.nan)

    all_params_flat = jnp.concatenate([x.ravel() for x in array_leaves])

    if all_params_flat.size == 0:
        return jnp.array(jnp.nan), jnp.array(jnp.nan)

    return jnp.mean(all_params_flat), jnp.std(all_params_flat)


@nnx_conditional_jit(cond=debug, static_argnames=("config", "len_dataset"))
def train_batch(
    carry: tuple[nnx.Rngs, Models, Opts],
    dataset,
    config: Config,
    len_dataset: int,
) -> tuple[tuple[nnx.Rngs, Models, Opts], Metrics]:
    (rngs, agent_state, opts) = carry

    actor_net = agent_state.actor
    q_net = agent_state.vec_q
    q_target_net = agent_state.vec_q_target
    alpha_net = agent_state.alpha
    alpha_prime_net = agent_state.alpha_prime

    actor_opt = opts.actor
    q_opt = opts.q
    alpha_opt = opts.alpha
    alpha_prime_opt = opts.alpha_prime

    assert config.cql_lagrange == (
        alpha_prime_net is not None and alpha_prime_opt is not None
    )

    # draw one key for this call and split into many subkeys used below
    key = rngs.random()

    def batch_sampler(dataset):
        indices = jax.random.choice(key, len_dataset, (config.batch_size,))
        # jax.debug.print(indices)
        # Use jax.debug.print without f-strings so the tracer isn't stringified during tracing.
        # This will print concrete values at runtime (not during tracing/compilation).
        # jax.debug.print("indices[:8] : {}", indices[:8])
        return Transition(
            obs=jax.tree_util.tree_map(lambda x: x[indices], dataset.obs),
            action=jax.tree_util.tree_map(lambda x: x[indices], dataset.action),
            reward=jax.tree_util.tree_map(lambda x: x[indices], dataset.reward),
            next_obs=jax.tree_util.tree_map(lambda x: x[indices], dataset.next_obs),
            done=jax.tree_util.tree_map(lambda x: x[indices], dataset.done),
        ), indices

    batch, _ = batch_sampler(dataset)

    def alpha_loss_fn(
        alpha_net: LogScalar, actor_net: TanhGaussianPolicy, batch: Transition
    ):
        pi = actor_net(batch.obs)
        _, log_pi = pi.sample_and_log_prob(seed=key)
        target_entropy = -batch.action.shape[-1]
        loss = alpha_net() * (-log_pi.sum(-1) - target_entropy).mean()
        return loss

    alpha_grad_fn = nnx.value_and_grad(alpha_loss_fn)
    alpha_loss, alpha_grad = alpha_grad_fn(alpha_net, actor_net, batch)
    alpha_opt.update(grads=alpha_grad)
    # read updated alpha module from optimizer (nnx.Optimizer stores the updated model)
    new_alpha_net = alpha_opt.model
    alpha = new_alpha_net.exp()

    def actor_loss_fn(actor_net_param: TanhGaussianPolicy):
        pi = actor_net_param(batch.obs)
        sampled_action, log_pi = pi.sample_and_log_prob(seed=key)
        log_pi = log_pi.sum(-1)
        q_values = q_net(batch.obs, sampled_action)
        q_min = jnp.min(q_values, axis=-1)
        q_std = jnp.std(q_values, axis=-1)
        loss = (-q_min + alpha * log_pi).mean()
        return loss, (-log_pi, q_min, q_std)

    actor_grad_fn = nnx.value_and_grad(actor_loss_fn, has_aux=True)
    (actor_loss, (entropy, q_min, q_std)), actor_grad = actor_grad_fn(actor_net)
    actor_opt.update(grads=actor_grad)
    new_actor_net = actor_opt.model

    def get_param(net: nnx.Module):
        state = nnx.state(net)
        param = nnx.filter_state(state, nnx.Param)
        return param

    bs = batch.next_obs.shape[0]

    def _sample_next_v(rng_key, next_obs):
        next_pi = new_actor_net(next_obs)
        next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng_key)
        next_q = q_target_net(next_obs, next_action)
        return next_q.min(-1) - alpha * log_next_pi.sum(-1)

    next_v_target = _sample_next_v(key, batch.next_obs)
    next_v_target = jax.lax.stop_gradient(next_v_target)
    target = batch.reward + config.gamma * (1 - batch.done) * next_v_target

    key_pi, key_next_pi, key_cql = jax.random.split(key, 3)

    def _sample_actions(rng_key, obs):
        pi = new_actor_net(obs)
        return pi.sample_and_log_prob(seed=rng_key)

    # sample actions per-batch (vectorized)
    pi_actions, log_prob = jax.vmap(lambda k, o: _sample_actions(k, o))(
        jax.random.split(key_pi, bs), batch.obs
    )
    pi_next_actions, next_log_prob = jax.vmap(lambda k, o: _sample_actions(k, o))(
        jax.random.split(key_next_pi, bs), batch.next_obs
    )

    # https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/sac/cql.py#L139
    # https://github.com/young-geng/JaxCQL/blob/bac4299194bd6ae2bc7db9034fd1a31ac43a30d7/JaxCQL/conservative_sac.py#L214
    # https://github.com/EmptyJackson/unifloral/blob/0ac6fb73590436efc29214601bef12c8ab23fae3/algorithms/cql.py#L286
    cql_random_actions = jax.random.uniform(
        key_cql,
        shape=(batch.action.shape[0], config.num_action_sample, batch.action.shape[1]),
        minval=-1.0,
        maxval=1.0,
    )

    def q_loss_fn(q_net: VectorQ):
        if not config.cql_importance_sampling:
            raise NotImplementedError

        # https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/sac/cql.py#L254
        # https://github.com/EmptyJackson/unifloral/blob/0ac6fb73590436efc29214601bef12c8ab23fae3/algorithms/cql.py#L297
        beta_q = q_net(batch.obs, batch.action)
        critic_loss = jnp.square((beta_q - jnp.expand_dims(target, -1)))
        critic_loss = critic_loss.sum(-1).mean()

        rand_q = jax.vmap(q_net, in_axes=(None, 1))(batch.obs, cql_random_actions)
        pi_q = jnp.expand_dims(q_net(batch.obs, pi_actions), 0).repeat(
            config.num_action_sample, axis=0
        )
        next_pi_q = jnp.expand_dims(q_net(batch.next_obs, pi_next_actions), 0).repeat(
            config.num_action_sample, axis=0
        )

        rand_q = rand_q - jnp.expand_dims(
            jnp.log(0.5 ** batch.action.shape[-1]), (0, 1)
        )
        pi_q = pi_q - jnp.expand_dims(log_prob.sum(-1), (1,))
        next_pi_q = next_pi_q - jnp.expand_dims(next_log_prob.sum(-1), (1,))

        all_qs = jnp.concatenate([rand_q, pi_q, next_pi_q], axis=0)
        q_ood = (
            jax.scipy.special.logsumexp(all_qs / config.cql_temperature, axis=0)
            * config.cql_temperature
        )
        q_gap = q_ood - beta_q
        gap_mean = q_gap.sum(-1).mean()

        if config.cql_lagrange:
            lagrange_multiplier = jnp.exp(alpha_prime_net())
            gap_residual = gap_mean - config.cql_target_gap_expansion
            min_q_loss = lagrange_multiplier * gap_residual
        else:
            gap_residual = gap_mean
            min_q_loss = config.cql_min_q_weight * gap_mean

        critic_loss += min_q_loss
        return critic_loss, (gap_mean, gap_residual)

    q_loss_grad = nnx.value_and_grad(q_loss_fn, has_aux=True)
    (critic_loss, (gap_mean, gap_residual)), critic_grad = q_loss_grad(q_net)
    q_opt.update(grads=critic_grad)
    new_q = q_opt.model

    alpha_prime_loss = None
    alpha_prime_value = jnp.array(config.cql_min_q_weight, dtype=jnp.float32)
    new_alpha_prime_net = None

    if config.cql_lagrange:

        def alpha_prime_loss_fn(alpha_prime_param: LogScalar):
            return -alpha_prime_param() * jax.lax.stop_gradient(gap_residual)

        alpha_prime_grad_fn = nnx.value_and_grad(alpha_prime_loss_fn)
        alpha_prime_loss, alpha_prime_grad = alpha_prime_grad_fn(alpha_prime_net)
        alpha_prime_opt.update(grads=alpha_prime_grad)
        new_alpha_prime_net = alpha_prime_opt.model
        alpha_prime_value = new_alpha_prime_net.exp()

    # Polyak (soft) update target params toward updated online Q params
    q_target_net_param = optax.incremental_update(
        new_tensors=get_param(new_q),
        old_tensors=get_param(q_target_net),
        step_size=float(config.polyak_step_size),
    )
    nnx.update(q_target_net, q_target_net_param)

    agent_state = Models(
        actor=new_actor_net,
        vec_q=new_q,
        vec_q_target=q_target_net,
        alpha=new_alpha_net,
        alpha_prime=new_alpha_prime_net,
    )
    opts = Opts(
        actor=actor_opt,
        q=q_opt,
        alpha=alpha_opt,
        alpha_prime=alpha_prime_opt,
    )

    metrics = Metrics(
        critic_loss=critic_loss,
        gap_mean=gap_mean,
        gap_residual=gap_residual,
        actor_loss=actor_loss,
        alpha_loss=alpha_loss,
        alpha_prime_loss=alpha_prime_loss,
        entropy=entropy.mean(),
        alpha=alpha,
        alpha_prime=alpha_prime_value,
        q_min=q_min.mean(),
        q_std=q_std.mean(),
    )

    return (rngs, agent_state, opts), metrics


@nnx_conditional_jit(cond=debug, static_argnames=("config", "len_dataset", "length"))
def train_multiple_steps(
    carry: tuple[nnx.Rngs, Models, Opts], dataset, config, len_dataset, length: int
):
    metrics = None
    if debug:
        for _ in range(length):
            carry, metrics = train_batch(carry, dataset, config, len_dataset)

    else:
        graphdef, state = nnx.split(carry)

        def scan_fn(state, _):
            new_carry, metrics = train_batch(
                nnx.merge(graphdef, state), dataset, config, len_dataset
            )
            _, new_state = nnx.split(new_carry)
            return new_state, metrics

        final_state, stacked_metrics = jax.lax.scan(scan_fn, state, None, length=length)

        # jax.debug.print(
        #     "length: {} , last_loss: {}",
        #     len(stacked_metrics.critic_loss),
        #     stacked_metrics.critic_loss[-1],
        # )
        nnx.update(carry, final_state)
        metrics = jax.tree_util.tree_map(lambda x: x[-1], stacked_metrics)
        # jax.debug.print(
        #     "metric shape: {}, carry shape : {}",
        #     last_metrics.critic_loss.shape,
        #     len(carry[0]),
        # )

    assert metrics is not None
    return carry, metrics


def evaluate_policy(
    config: Config, env: vector.VectorEnv, actor: TanhGaussianPolicy, num_episodes: int
):
    # Run episodes in the vectorized env using the deterministic policy (tanh(mean))
    obs = env.reset()
    episode_returns = []
    cur_returns = np.zeros(env.num_envs, dtype=float)
    while len(episode_returns) < num_episodes:
        obs_j = jnp.array(obs)
        # deterministic action: tanh(mean)
        x = actor.layer(obs_j)
        mean = actor.mean(x)
        action_j = jnp.tanh(mean)
        action = np.asarray(action_j)
        obs, reward, done, info = env.step(action)
        cur_returns += reward
        for i, d in enumerate(done):
            if d:
                episode_returns.append(float(cur_returns[i]))
                cur_returns[i] = 0.0

    scores = d4rl.get_normalized_score(config.dataset, np.array(episode_returns)) * 100

    return EvalMetrics(
        avg_return=float(np.mean(episode_returns)),
        score=scores.mean(),
        score_std=scores.std(),
    )


def evaluate(
    config: Config,
    models: Models,
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
    models: Models,
    opts: Opts,
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
        artifact = wandb.Artifact(name=name, type="model", metadata={"step": step})
        artifact.add_dir(local_path=str(cur_step_dir))
        wandb_run.log_artifact(artifact, aliases=[f"{step}"])


def _clean_dict(d):
    return {k: v for k, v in d.items() if v is not None}


def log_metrics(wandb_run, step, train_dict=None, eval_dict=None, state_dict=None):
    statistics: dict = {}
    if train_dict:
        statistics.update(_clean_dict(train_dict))
    if eval_dict:
        statistics.update(_clean_dict(eval_dict))
    if state_dict:
        statistics.update(_clean_dict(state_dict))
    if statistics:
        wandb_run.log(statistics, step=step)


def log_train(metrics: Metrics):
    log_data = {
        f"train/{k}": float(v) for k, v in metrics._asdict().items() if v is not None
    }
    return log_data


def log_obj_stats(models: Models, opts: Opts):
    stats = {}
    for name, model in models._asdict().items():
        mean, std = get_all_array_stats(model)

        stats[f"params/{name}_mean"], stats[f"params/{name}_std"] = (
            mean,
            std,
        )

    for name, opt in opts._asdict().items():
        (
            stats[f"opts/{name}_mean"],
            stats[f"opts/{name}_std"],
        ) = get_all_array_stats(opt)
    return stats


def extract_experiment_metadata(config):
    global wandb_tags
    global machine_name
    git_hash = get_git_hash(length=12)
    wandb_tags.append(git_hash)
    wandb_config = dataclasses.asdict(config)
    wandb_config["metadata"] = {
        # "git_hash": git_hash,
        # "host": socket.gethostname(),
        "machine_name": machine_name,
        # "username": getpass.getuser(),
        # "mac_address": uuid.getnode(),
    }

    exp_hash = generate_experiment_hash(config_dict=wandb_config, length=12)

    return wandb_config, exp_hash


def main(sweep=False):
    run_params = {}
    if sweep:
        wandb_run = wandb.init()
        valid_keys = {f.name for f in dataclasses.fields(Config)}
        run_params = {k: v for k, v in wandb_run.config.items() if k in valid_keys}
    config = Config(**run_params)
    wandb_config, exp_hash = extract_experiment_metadata(config=config)

    wandb_run = None
    if wandb_log:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb_run = wandb.init(
            project=wandb_project,
            config=wandb_config,
            # name=f"cql/{exp_hash}",
            notes=wandb_notes,
            tags=wandb_tags,
            settings=wandb.Settings(
                resume="allow",
                save_code=True,
                disable_git=False,
            ),
            group=None if sweep else wandb_group_id,
            # id=wandb_run_id,
        )

    save_root_dir: Path | None = None
    if wandb_run and save:
        name = wandb_run.name if wandb_run.name else wandb_run.id
        save_root_dir = Path.cwd() / f"ckpt/cql/{name}"
        Path.mkdir(save_root_dir, exist_ok=True, parents=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    checkpointer = ocp.StandardCheckpointer()

    rngs, env, dataset = prepare_training(config)
    env.seed(config.seed)
    actor_net, q_net, q_target_net, alpha, alpha_prime = initialize_network(
        config, rngs, env
    )

    actor_opt = nnx.Optimizer(actor_net, optax.adam(learning_rate=config.actor_lr))
    q_opt = nnx.Optimizer(q_net, optax.adam(learning_rate=config.q_lr))
    alpha_opt = nnx.Optimizer(alpha, optax.adam(learning_rate=config.alpha_lr))
    alpha_prime_opt = (
        nnx.Optimizer(alpha_prime, optax.adam(learning_rate=config.alpha_lr))
        if alpha_prime is not None
        else None
    )

    models = Models(
        actor=actor_net,
        vec_q=q_net,
        vec_q_target=q_target_net,
        alpha=alpha,
        alpha_prime=alpha_prime,
    )
    opts = Opts(
        actor=actor_opt,
        q=q_opt,
        alpha=alpha_opt,
        alpha_prime=alpha_prime_opt,
    )
    len_dataset = len(dataset.obs)
    cur_step = 0
    step_length = math.gcd(eval_interval, model_save_interval, train_log_interval)

    while cur_step < config.num_updates:
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

        (rngs, models, opts), metrics = train_multiple_steps(
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
        cur_step += step_length

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
