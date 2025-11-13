from collections import namedtuple
from algorithms.cql import Config, prepare_training, initialize_network, Models

import orbax.checkpoint as ocp
import flax.nnx as nnx

from pathlib import Path

from utils.jax import restore_state


def restore_models(checkpointer, models):
    restored = {}
    for name, model in models._asdict().items():
        graphdef, abs_state = nnx.split(nnx.eval_shape(lambda: model))
        state = checkpointer.restore(Path(ckpt_dir) / "model" / name, abs_state)
        restored_model = nnx.merge(graphdef, state)
        restored[name] = restored_model

    restored_models = Models(**restored)
    return restored_models


if __name__ == "__main__":
    ckpt_dir = "/home/gwanwoo/developer/d4rl_train/ckpt/cql/c4b4795d80a5/100000"
    checkpointer = ocp.StandardCheckpointer()

    config = Config()
    rngs, env, _ = prepare_training(config)
    actor, q, q_target, alpha = initialize_network(config, rngs, env)

    models = Models(actor=actor, vec_q=q, vec_q_target=q_target, alpha=alpha)
    restored_models = restore_models(checkpointer, models)
