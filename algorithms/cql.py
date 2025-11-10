import time
from collections import namedtuple

# from typing import NamedTuple
from dataclasses import dataclass
from functools import partial

import d4rl
import distrax
import flax.nnx as nnx
import gym
import gym.vector as vector
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.nnx.nn.initializers import constant, uniform


@dataclass(frozen=True)
class Config:
    random_seed: int = 421
    eval_workers: int = 8
    num_critics: int = 2
    dataset: str = "maze2d-umaze-v1"
    num_updates: int = 100
    eval_interval: int = 10
    batch_size: int = 64
    polyak_step_size: float = 0.005
    gamma: float = 0.99
    cql_temperature: int = 10
    cql_min_q_weight: float = 0.1
    actor_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4


AgentTrainState = namedtuple("AgentTrainState", "actor vec_q vec_q_target alpha")
Opts = namedtuple("Opts", "actor q alpha")
Transition = namedtuple("Transition", "obs action reward next_obs done")


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class EntropyCoef(nnx.Module):
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


@nnx.jit(static_argnames=("config", "len_dataset"))
def train_batch(
    carry: tuple[nnx.Rngs, AgentTrainState, Opts],
    _,
    dataset,
    config: Config,
    len_dataset: int,
):
    (rngs, agent_state, opts) = carry

    actor_net = agent_state.actor
    q_net = agent_state.vec_q
    q_target_net = agent_state.vec_q_target
    alpha_net = agent_state.alpha

    actor_opt = opts.actor
    q_opt = opts.q
    alpha_opt = opts.alpha

    # draw one key for this call and split into many subkeys used below
    key = rngs.random()

    def batch_sampler(dataset):
        indices = jax.random.choice(
            key, len_dataset, (config.batch_size,), replace=False
        )
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
        alpha_net: EntropyCoef, actor_net: TanhGaussianPolicy, batch: Transition
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

    # q_target_net_param = optax.incremental_update(
    #     new_tensors=get_param(q_target_net),
    #     old_tensors=get_param(q_net),
    #     step_size=config.polyak_step_size,
    # )
    # nnx.update(q_target_net, q_target_net_param)

    # def _sample_next_v(next_obs):
    #     next_pi = actor_net(next_obs)
    #     next_action, log_next_pi = next_pi.sample_and_log_prob(seed=key)
    #     next_q = q_target_net(next_obs, next_action)
    #     return next_q.min(-1) - alpha * log_next_pi.sum(-1)

    # next_v_target = jax.lax.stop_gradient(_sample_next_v(batch.next_obs))

    # Use updated actor (new_actor_net) to compute target V(s') with current target Q.
    # For vectorized sampling we split keys across the batch
    bs = batch.next_obs.shape[0]

    def _sample_next_v(rng_key, next_obs):
        next_pi = new_actor_net(next_obs)
        next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng_key)
        next_q = q_target_net(next_obs, next_action)
        return next_q.min(-1) - alpha * log_next_pi.sum(-1)

    next_v_target = _sample_next_v(key, batch.next_obs)
    next_v_target = jax.lax.stop_gradient(next_v_target)
    target = batch.reward + config.gamma * (1 - batch.done) * next_v_target

    # def _sample_actions(obs):
    #     pi = actor_net(obs)
    #     return pi.sample(seed=key)

    # pi_actions = _sample_actions(batch.obs)
    # pi_next_actions = _sample_actions(batch.next_obs)
    # cql_random_actions = jax.random.uniform(
    #     key, shape=batch.action.shape, minval=-1.0, maxval=1.0
    # )

    # draw keys for actor/sample and random actions
    key_pi, key_next_pi, key_cql = jax.random.split(key, 3)

    def _sample_actions(rng_key, obs):
        pi = new_actor_net(obs)
        return pi.sample(seed=rng_key)

    # sample actions per-batch (vectorized)
    pi_actions = jax.vmap(lambda k, o: _sample_actions(k, o))(
        jax.random.split(key_pi, bs), batch.obs
    )
    pi_next_actions = jax.vmap(lambda k, o: _sample_actions(k, o))(
        jax.random.split(key_next_pi, bs), batch.next_obs
    )
    cql_random_actions = jax.random.uniform(
        key_cql, shape=batch.action.shape, minval=-1.0, maxval=1.0
    )

    def q_loss_fn(q_net: VectorQ):
        q_pred = q_net(batch.obs, batch.action)
        critic_loss = jnp.square((q_pred - jnp.expand_dims(target, -1)))
        critic_loss = critic_loss.sum(-1).mean()

        rand_q = q_net(batch.obs, cql_random_actions)
        pi_q = q_net(batch.obs, pi_actions)
        next_pi_q = q_net(batch.next_obs, pi_next_actions)

        all_qs = jnp.concatenate([rand_q, pi_q, next_pi_q, q_pred], axis=1)
        q_ood = jax.scipy.special.logsumexp(all_qs / config.cql_temperature, axis=1)
        q_ood = q_ood * config.cql_temperature
        q_diff = (jnp.expand_dims(q_ood, 1) - q_pred).mean()
        min_q_loss = q_diff * config.cql_min_q_weight

        critic_loss += min_q_loss.mean()
        return critic_loss

    q_loss_grad = nnx.value_and_grad(q_loss_fn)
    critic_loss, critic_grad = q_loss_grad(q_net)
    q_opt.update(grads=critic_grad)
    new_q = q_opt.model

    # Polyak (soft) update target params toward updated online Q params
    q_target_net_param = optax.incremental_update(
        new_tensors=get_param(new_q),
        old_tensors=get_param(q_target_net),
        step_size=float(config.polyak_step_size),
    )
    nnx.update(q_target_net, q_target_net_param)

    # new_agent_state = AgentTrainState(
    #     actor=new_actor_net,
    #     vec_q=new_q,
    #     vec_q_target=q_target_net,
    #     alpha=new_alpha_net,
    # )
    # new_opts = Opts(actor=actor_opt, q=q_opt, alpha=alpha_opt)

    metrics = {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "entropy": entropy,
        "alpha": alpha,
        "q_min": q_min,
        "q_std": q_std,
    }

    # # update rngs.random so next call won't reuse keys
    # try:
    #     rngs = nnx.Rngs(
    #         params=rngs.params, random=jax.random.randint(key, (), 0, 2**31)
    #     )
    # except Exception:
    #     # fallback: leave rngs as-is if nnx.Rngs construction differs
    #     pass
    return (rngs, agent_state, opts), metrics


def evaluate_policy(env, actor: TanhGaussianPolicy, num_episodes: int):
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
    return float(np.mean(episode_returns))


if __name__ == "__main__":
    config = Config()
    rngs = nnx.Rngs(params=config.random_seed, random=config.random_seed + 1)

    env = vector.make(config.dataset, num_envs=config.eval_workers, asynchronous=False)
    dataset = d4rl.qlearning_dataset(gym.make(config.dataset))
    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )

    num_actions = env.single_action_space.shape[0]
    actor_net = TanhGaussianPolicy(
        num_actions=num_actions,
        input_dim=env.single_observation_space.shape[0],
        hidden_dims=[256, 256],
        rngs=rngs,
    )
    q_net = VectorQ(
        num_critics=config.num_critics,
        input_dim=env.single_observation_space.shape[0]
        + env.single_action_space.shape[0],
        hidden_dims=[256, 256],
        output_dim=1,
        rngs=rngs,
    )
    q_target_net = nnx.clone(q_net)
    alpha_net = EntropyCoef()

    num_evals = config.num_updates

    actor_opt = nnx.Optimizer(actor_net, optax.adam(learning_rate=config.actor_lr))
    q_opt = nnx.Optimizer(q_net, optax.adam(learning_rate=config.q_lr))
    alpha_opt = nnx.Optimizer(alpha_net, optax.adam(learning_rate=config.alpha_lr))

    agent_state = AgentTrainState(
        actor=actor_net,
        vec_q=q_net,
        vec_q_target=q_target_net,
        alpha=alpha_net,
    )
    opts = Opts(
        actor=actor_opt,
        q=q_opt,
        alpha=alpha_opt,
    )
    carry = (rngs, agent_state, opts)
    len_dataset = len(dataset.obs)

    scan_fn = partial(
        train_batch, dataset=dataset, config=config, len_dataset=len_dataset
    )

    print("Starting training")
    start_time = time.time()
    for step in range(config.num_updates):
        (rngs, agent_state, opts), metrics = train_batch(
            carry, None, dataset=dataset, config=config, len_dataset=len_dataset
        )

        # log / evaluate
        if (step % config.eval_interval == 0) or (step == config.num_updates - 1):
            avg_ret = evaluate_policy(env, agent_state.actor, num_episodes=env.num_envs)
            elapsed = time.time() - start_time
            print(f"step={step}  elapsed={elapsed:.1f}s  avg_return={avg_ret:.3f}")
    print("Training finished")
