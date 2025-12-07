from typing import Callable, Literal, Sequence, Any, Protocol

from flax import struct
import optax
from train.train_ogbench import TrainConfig


import distrax
import flax.nnx as nnx
import jax
import jax.numpy as jnp

# nnx_conditional_jit is unused here; keep simple explicit jit usage.


@struct.dataclass
class Config(TrainConfig):
    project_name: str = struct.field(pytree_node=False, default="d4rl_train")
    model_type: str = struct.field(pytree_node=False, default="transformer")
    learning_rate: float = 1e-4
    batch_size: int = 512
    hidden_dims: tuple[int] = struct.field(
        pytree_node=False, default=(512, 512, 512, 512)
    )
    gamma: float = 0
    num_ensemble: int = 2
    time_encoding: bool = True
    debug: bool = False


def map_norm_fn(
    norm_name: Literal["layer_norm", "batch_norm", "none"],
) -> Callable | None:
    if norm_name == "layer_norm":
        return nnx.LayerNorm
    elif norm_name == "batch_norm":
        return nnx.BatchNorm
    elif norm_name == "none":
        return None
    else:
        raise ValueError(f"Unknown norm function name: {norm_name}")


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int | None,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.gelu,
        norm_fn: Callable | None = None,
    ):
        self.layers = []
        self.norms = []
        self.activation_fn = activation_fn
        self.output_dim = output_dim
        self.num_hidden_dims = hidden_dims

        in_dim = input_dim

        for i, out_dim in enumerate(hidden_dims):
            self.layers.append(
                nnx.Linear(in_features=in_dim, out_features=out_dim, rngs=rngs)
            )
            self.norms.append(
                norm_fn(out_dim, rngs=rngs) if norm_fn is not None else None
            )
            in_dim = out_dim

        if output_dim is not None:
            self.layers.append(
                nnx.Linear(in_features=in_dim, out_features=output_dim, rngs=rngs)
            )
            self.norms.append(None)

    def __call__(self, x):
        sow = None

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < len(self.num_hidden_dims):
                x = self.activation_fn(x)
                if norm is not None:
                    x = norm(x)
            if i == len(self.layers) - 2:
                sow = x

        return x, sow


class Value(nnx.Module):
    def __init__(
        self,
        /,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_ensemble: int,
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.gelu,
        norm_fn: Callable | None = None,
    ):
        self.num_ensemble = num_ensemble
        self.networks = [
            MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                activation_fn=activation_fn,
                norm_fn=norm_fn,
                rngs=rngs,
            )
            for _ in range(num_ensemble)
        ]

    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)

        q_values = []
        sows = []
        for net in self.networks:
            q_value, sow = net(x)
            q_values.append(q_value)
            sows.append(sow)

        q_values_stacked = jnp.stack(q_values, axis=-1)  # (batch_size, 1, num_ensemble)
        sow_stacked = jnp.stack(sows, axis=-1)  # (batch_size, hidden_dim, num_ensemble)
        return q_values_stacked, sow_stacked


class Actor(nnx.Module):
    def __init__(
        self,
        /,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        action_dim: int,
        activation_fn: Callable = nnx.gelu,
        rngs: nnx.Rngs,
        norm_fn: Callable | None = None,
        init_scale: float = 1e-2,
        state_dependent_std: bool = False,
        const_std: bool = True,
        tanh_squash: bool = False,
        log_std_min: float = -5,
        log_std_max: float = 2,
    ):
        self.state_dependent_std = state_dependent_std
        self.const_std = const_std
        self.tanh_squash = tanh_squash
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=None,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            rngs=rngs,
        )
        self.mean_net = nnx.Linear(
            in_features=hidden_dims[-1],
            out_features=action_dim,
            kernel_init=nnx.initializers.variance_scaling(
                init_scale, "fan_avg", "uniform"
            ),
            rngs=rngs,
        )
        if self.state_dependent_std:
            self.log_std_net = nnx.Linear(
                in_features=hidden_dims[-1],
                out_features=action_dim,
                kernel_init=nnx.initializers.variance_scaling(
                    init_scale, "fan_avg", "uniform"
                ),
                rngs=rngs,
            )
        elif not self.const_std:
            self.log_stds = nnx.Param(
                nnx.initializers.zeros(rngs.params(), (action_dim,))
            )

    def __call__(self, obs, temperature=1.0):
        features, _ = self.network(obs)
        means = self.mean_net(features)
        if self.state_dependent_std:
            log_stds = self.log_std_net(features)
        elif self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_stds()
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            dist = distrax.Transformed(dist, distrax.Tanh())

        return dist


class ActorVectorField(nnx.Module):
    def __init__(
        self,
        /,
        *,
        input_dim: int,
        hidden_dims: tuple[int],
        action_dim: int,
        rngs: nnx.Rngs,
        norm_fn: Callable | None = None,
        encoder: Callable | None = None,
    ):
        self.encoder = encoder
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            norm_fn=norm_fn,
            rngs=rngs,
        )

    def __call__(self, obs, act, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            obs = self.encoder(obs)
        if times is None:
            input_values = jnp.concatenate([obs, act], axis=-1)
        else:
            input_values = jnp.concatenate([obs, act, times], axis=-1)
        v = self.mlp(input_values)
        return v


class FQL(nnx.Module):
    def __init__(
        self,
        config: Config,
        example_observation: jnp.ndarray,
        example_action: jnp.ndarray,
    ):
        self.rngs = nnx.Rngs(default=config.seed, random=config.seed + 1)
        self.config = config

        self.critic = Value(
            input_dim=example_observation.shape[-1] + example_action.shape[-1],
            hidden_dims=config.hidden_dims,
            num_ensemble=config.num_ensemble,
            rngs=self.rngs,
            activation_fn=nnx.gelu,
        )

        # Actor
        self.actor = Actor(
            input_dim=example_observation.shape[-1],
            hidden_dims=config.hidden_dims,
            action_dim=example_action.shape[-1],
            rngs=self.rngs,
        )

        time_dim = 1 if config.time_encoding else 0
        self.vector_field = ActorVectorField(
            input_dim=example_observation.shape[-1]
            + example_action.shape[-1]
            + time_dim,
            hidden_dims=config.hidden_dims,
            action_dim=example_action.shape[-1],
            rngs=self.rngs,
        )

        self.critic_opt = nnx.Optimizer(self.critic, optax.adam(config.learning_rate))
        self.actor_opt = nnx.Optimizer(self.actor, optax.adam(config.learning_rate))
        self.vf_opt = nnx.Optimizer(self.vector_field, optax.adam(config.learning_rate))

        if config.debug:
            self.train_step = nnx.jit(self.train_step)

    def _critic_loss(self, critic: Value, actor: Actor, batch, gamma: float):
        obs = batch["observations"]
        act = batch["actions"]
        rew = batch["rewards"]
        next_obs = batch["next_observations"]
        mask = batch.get("masks", jnp.ones_like(rew))

        # Current Q
        q_values, _ = critic(obs, act)  # (batch, 1, num_ensemble)

        # Next actions (deterministic mean to avoid rng plumbing)
        dist_next = actor(next_obs)
        next_actions = dist_next.mean()
        next_q_values, _ = critic(next_obs, next_actions)
        next_q = next_q_values.min(axis=-1)  # conservative target

        target_q = rew + gamma * mask * next_q
        target_q = jax.lax.stop_gradient(target_q)

        loss = jnp.mean((q_values - target_q) ** 2)
        return loss

    def _actor_loss(self, actor: Actor, critic: Value, batch):
        obs = batch["observations"]
        dist = actor(obs)
        actions = dist.mean()
        q_values, _ = critic(obs, actions)
        q = q_values.mean(axis=-1)
        return -jnp.mean(q)

    def _vf_loss(self, vf: ActorVectorField, batch):
        obs = batch["observations"]
        act = batch["actions"]
        batch_size, action_dim = act.shape

        x_0 = jax.random.normal(self.rngs.random(), (batch_size, action_dim))
        x_1 = act
        t = jax.random.uniform(self.rngs.random(), (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        v_pred, _ = vf(obs, x_t, times=t)
        return jnp.mean((v_pred - vel) ** 2)

    def total_loss(self, batch: dict[str, jnp.ndarray]):
        gamma = getattr(self.config, "gamma", 0.99)
        critic_loss = self._critic_loss(self.critic, self.actor, batch, gamma)
        actor_loss = self._actor_loss(self.actor, self.critic, batch)
        vf_loss = self._vf_loss(self.vector_field, batch)
        loss = critic_loss + actor_loss + vf_loss
        return loss, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "vf_loss": vf_loss,
        }

    def train_step(self, batch: dict[str, jnp.ndarray]):
        # Critic update
        critic_loss, critic_grads = nnx.value_and_grad(self._critic_loss)(
            self.critic, self.actor, batch, self.config.gamma
        )
        self.critic_opt.update(critic_grads)

        # Actor update
        actor_loss, actor_grads = nnx.value_and_grad(self._actor_loss)(
            self.actor, self.critic, batch
        )
        self.actor_opt.update(actor_grads)

        # Vector field update
        vf_loss, vf_grads = nnx.value_and_grad(self._vf_loss)(self.vector_field, batch)
        self.vf_opt.update(vf_grads)

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "vf_loss": vf_loss,
        }

    @nnx.jit
    def sample_actions(self, obs, temperature=1.0):
        dist = self.actor(obs, temperature=temperature)
        return dist.sample(seed=self.rngs.random())
