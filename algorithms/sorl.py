from collections import namedtuple
from typing import Callable, Sequence
import os

import flax.nnx as nnx
import chex
import jax.numpy as jnp

from dataset.ogbench_singletask import load_env_and_datasets


wandb_log: bool = True
wandb_notes: str | None = None
wandb_tags: list[str] = ["fql"]
wandb_project: str = "d4rl_train"
wandb_group_id: str | None = None
machine_name: str = os.environ["MACHINE_NAME"]
train_log_interval: int = 10_000
eval_interval: int = 50_000
model_save: bool = True
model_save_interval: int = 500_000
debug: bool = False


@chex.dataclass
class Config:
    env_name: str
    seed: int = 42
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99


Models = namedtuple("Models", "actor")
Opts = namedtuple("Opts", "actor q log_alpha log_alpha_prime")
Transition = namedtuple("Transition", "obs action reward next_obs done")

Metrics = namedtuple(
    "Metrics",
    "critic_loss gap_mean gap_residual actor_loss alpha_loss alpha_prime_loss entropy alpha alpha_prime q_min q_std q_max",
)
EvalMetrics = namedtuple("EvalMetrics", "avg_return score score_std")

EncoderFn = Callable[[jnp.ndarray], jnp.ndarray]


class MLP(nnx.Module):
    """Simple MLP that lazily infers the input dimensionality."""

    def __init__(
        self,
        /,
        *,
        hidden_dims: Sequence[int],
        activate_final: bool = True,
        layer_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.rngs = rngs
        self._built = False
        self._input_features: int | None = None
        self.layers: list[nnx.Linear] = []
        self.norm_layers: list[nnx.LayerNorm | None] = []

    def _build(self, input_dim: int) -> None:
        self.layers.clear()
        self.norm_layers.clear()
        prev_dim = input_dim
        for idx, out_dim in enumerate(self.hidden_dims):
            self.layers.append(
                nnx.Linear(
                    in_features=prev_dim,
                    out_features=out_dim,
                    rngs=self.rngs,
                )
            )
            is_last = idx == len(self.hidden_dims) - 1
            if self.layer_norm and (self.activate_final or not is_last):
                self.norm_layers.append(
                    nnx.LayerNorm(
                        num_features=out_dim,
                        rngs=self.rngs,
                    )
                )
            else:
                self.norm_layers.append(None)
            prev_dim = out_dim

        self._input_features = input_dim
        self._built = True

    def __call__(self, x):
        feature_dim = int(x.shape[-1])
        if not self._built:
            self._build(feature_dim)
        elif self._input_features != feature_dim:
            raise ValueError(
                f"MLP expected inputs with last dimension {self._input_features}, got {feature_dim}."
            )

        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            norm = self.norm_layers[idx]
            if norm is not None:
                h = norm(h)
            is_last = idx == len(self.layers) - 1
            if not is_last or self.activate_final:
                h = nnx.relu(h)

        return h


def _ensure_feature_rank(value, reference):
    """Match the rank of aux inputs (times, step sizes, etc.) with main inputs."""

    if value is None:
        return jnp.zeros((*reference.shape[:-1], 1), dtype=reference.dtype)

    arr = jnp.asarray(value)
    if arr.ndim == 0:
        arr = jnp.broadcast_to(arr, (*reference.shape[:-1], 1))
    elif arr.ndim == reference.ndim - 1:
        arr = arr[..., None]
    return arr


class ShortcutModel(nnx.Module):
    """Shortcut model for flow matching with step-size conditioning."""

    def __init__(
        self,
        /,
        *,
        hidden_dims: Sequence[int],
        action_dim: int,
        layer_norm: bool = False,
        encoder: EncoderFn | None = None,
        rngs: nnx.Rngs,
    ):
        self.encoder = encoder
        self.layer_norm = layer_norm
        self.mlp = MLP(
            hidden_dims=(*hidden_dims, action_dim),
            activate_final=False,
            layer_norm=layer_norm,
            rngs=rngs,
        )

    def __call__(
        self,
        observations,
        actions,
        times=0,
        step_sizes=0,
        *,
        is_encoded: bool = False,
    ):
        """Return shortcut vectors for the given states, actions, and steps."""

        observations = jnp.asarray(observations)
        actions = jnp.asarray(actions)

        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        obs_rank_ref = observations
        times = _ensure_feature_rank(times, obs_rank_ref)
        step_sizes = _ensure_feature_rank(step_sizes, obs_rank_ref)

        inputs = jnp.concatenate([observations, actions, times, step_sizes], axis=-1)

        return self.mlp(inputs)


class Value(nnx.Module):
    """Value/critic network supporting ensembles."""

    def __init__(
        self,
        /,
        *,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        num_ensembles: int = 2,
        encoder: EncoderFn | None = None,
        rngs: nnx.Rngs,
    ):
        if num_ensembles < 1:
            raise ValueError("num_ensembles must be >= 1")

        self.encoder = encoder
        self.layer_norm = layer_norm
        self.num_ensembles = num_ensembles
        final_dims = tuple(hidden_dims) + (1,)
        self.value_nets = [
            MLP(
                hidden_dims=final_dims,
                activate_final=False,
                layer_norm=layer_norm,
                rngs=rngs,
            )
            for _ in range(num_ensembles)
        ]

    def __call__(self, observations, actions=None, timesteps=None):
        """Return V(s) or Q(s, a) style estimates depending on the inputs."""

        obs = jnp.asarray(observations)
        if self.encoder is not None:
            obs = self.encoder(obs)

        features = [obs]
        if actions is not None:
            features.append(jnp.asarray(actions))
        if timesteps is not None:
            features.append(_ensure_feature_rank(timesteps, obs))

        inputs = jnp.concatenate(features, axis=-1)
        outputs = [net(inputs) for net in self.value_nets]
        values = jnp.concatenate(outputs, axis=-1)

        if self.num_ensembles == 1:
            return values.squeeze(-1)
        return values


if __name__ == "__main__":
    load_env_and_datasets
