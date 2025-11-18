from functools import partial
import os
import wandb
from algorithms.cql import extract_experiment_metadata, wandb_project, main

sweep_id: str | None = "avk6l0af"
cuda_visible_devices: int | None = None

if cuda_visible_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_visible_devices}"

if not sweep_id:
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "valid/avg_return"},
        "parameters": {
            "seed": {"values": [2025, 3025, 4025, 5025]},
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
            # "epochs": {"values": [5, 10, 15]},
            # "lr": {"max": 0.1, "min": 0.0001},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project)

wandb.agent(
    sweep_id=sweep_id, function=partial(main, sweep=True), project=wandb_project
)
