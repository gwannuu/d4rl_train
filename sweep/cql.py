from functools import partial
import os
import wandb
from algorithms.cql import extract_experiment_metadata, wandb_project, main

sweep_id: str | None = None
cuda_visible_devices: int = -1

assert cuda_visible_devices != -1
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_visible_devices}"

if not sweep_id:
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "valid/score"},
        "parameters": {
            "seed": {
                "values": [20251, 30251, 40251, 50251, 60251, 70251, 80251, 90251]
            },
            # "batch_size": {"values": [256]},
            "dataset": {
                "values": [
                    "antmaze-large-diverse-v2",
                    "antmaze-large-play-v2",
                    "antmaze-medium-play-v2",
                    "antmaze-medium-diverse-v2",
                    "antmaze-umaze-v2",
                    "antmaze-umaze-diverse-v2",
                ]
            },
            "cql_importance_sampling": {"values": [True, False]},
            "cql_lagrange": {"values": [True, False]},
            "cql_target_gap_expansion": {"values": [5.0, 10.0]},
            # "epochs": {"values": [5, 10, 15]},
            # "lr": {"max": 0.1, "min": 0.0001},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project)

wandb.agent(
    sweep_id=sweep_id, function=partial(main, sweep=True), project=wandb_project
)
