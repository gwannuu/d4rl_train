import os
from functools import partial

import wandb
from algorithms import cql_w as cql_w_module
from utils.file import clear_ckpt_dir

sweep_id: str | None = None
cuda_visible_devices: int = -1
dataset_dir: str | None = cql_w_module.dataset_dir


if cuda_visible_devices == -1 and sweep_id is None:
    raise RuntimeError(
        "Set 'dataset_dir' in algorithms/cql_w.py before launching sweeps."
    )

if dataset_dir is None:
    raise RuntimeError(
        "Set 'dataset_dir' in algorithms/cql_w.py before launching sweeps."
    )

assert cuda_visible_devices != -1
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_visible_devices}"

clear_ckpt_dir()

if not sweep_id:
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "valid/score"},
        "parameters": {
            # "batch_size": {"values": [256]},
            "train_alpha_prime": {"values": [False]},
            "alpha_prime": {"values": [0.3, 0.1, 0.03, 0.01, 0.003, 1e-12]},
            "num_action_sample": {"values": [10]},
            "dataset": {
                "values": [
                    "antmaze-umaze-v2",
                    "antmaze-umaze-diverse-v2",
                    "antmaze-medium-play-v2",
                    "antmaze-medium-diverse-v2",
                    "antmaze-large-play-v2",
                    "antmaze-large-diverse-v2",
                ]
            },
            "seed": {"values": [20251, 30251, 40251, 50251]},
            "q_lr": {"values": [3e-4, 1e-4]},
            "actor_lr": {"values": [3e-5]},
            # "cql_importance_sampling": {"values": [True, False]},
            # "cql_lagrange": {"values": [True, False]},
            # "cql_target_gap_expansion": {"values": [5.0, 10.0]},
            # "epochs": {"values": [5, 10, 15]},
            # "lr": {"max": 0.1, "min": 0.0001},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project=cql_w_module.wandb_project
    )

wandb.agent(
    sweep_id=sweep_id,
    function=partial(cql_w_module.main, sweep=True),
    project=cql_w_module.wandb_project,
)
