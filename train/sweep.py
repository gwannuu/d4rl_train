from train import train_d4rl
import wandb
import dataclasses
from algorithms import cql


def main():
    wandb_run = wandb.init()
    valid_keys = {f.name for f in dataclasses.fields(cql.Config)}
    run_params = {k: v for k, v in wandb_run.config.items() if k in valid_keys}
    print(run_params)
    config = cql.Config(**run_params)
    wandb_config = train_d4rl.extract_experiment_metadata(config=config)
    wandb_run.config.update(wandb_config)
    train_d4rl.main(wandb_run=wandb_run)


if __name__ == "__main__":
    main()
