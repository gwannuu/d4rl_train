from collections import namedtuple
from dataclasses import dataclass

import d4rl
import distrax
import flax.nnx as nnx
import flax.nnx.nn as nn
import gym
import jax
import jax.numpy as jnp
from flax.nnx.nn.initializers import constant, uniform

AgentTrainState = namedtuple("AgentTrainState", "actor vec_q vec_q_target alpha")
Transition = namedtuple("Transition", "obs action reward next_obs done")


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class EntropyCoef(nnx.Module):
    def __init__(self, /, *, ent_coef_init: float = 1.0):
        self.log_ent_coef = nnx.Param(jnp.log(ent_coef_init))

    def __call__(self) -> jnp.ndarray:
        return self.log_ent_coef


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
        # 1. split_rngs={"params": True, "dropout": True} 대체
        #    num_critics 개수만큼 'params'와 'dropout' RNG를 분리합니다.
        critic_rngs = nnx.Rngs.make_rng_tree(rngs, ["params", "dropout"], num_critics)
        self.critics = [
            SoftQNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                rngs=critic_rngs[i],
            )
            for i in range(num_critics)
        ]

    def __call__(self, obs, action):
        q_values_list = [critic(obs, action) for critic in self.critics]
        q_values = jnp.stack(q_values_list, axis=-1)

        return q_values


class TanhGaussianPolicy(nn.Module):
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


@dataclass
class Config:
    random_seed: int = 42
    num_envs: int = 8
    num_critics: int = 2


if __name__ == "__main__":
    cfg = Config()
    rng = jax.random.PRNGKey(cfg.seed)

    env = gym.vector.make(cfg.dataset, num_envs=cfg.num_envs)
    dataset = d4rl.qlearning_dataset(gym.make(cfg.dataset))
    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )

    # num_actions = env.single_action_space.shape[0]
    num_actions = env.single_action_space.n
    actor_net = TanhGaussianPolicy(num_actions=num_actions)
    q_net = VectorQ(num_critics=cfg.num_critics)
    alpha_net = EntropyCoef()
