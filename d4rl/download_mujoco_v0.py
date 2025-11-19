import minari
import gymnasium


def download_dataset(dataset_name: str):
    dataset = minari.load_dataset(dataset_name, download=True)
    # eval_env_spec = dataset._eval_env_spec #None
    # env_spec = dataset.env_spec
    # vec_env = gymnasium.make_vec(
    #     id=env_spec,
    #     num_envs=8,
    #     vectorization_mode="sync",
    # )
    # env = dataset.recover_environment()
    # eval_env = dataset.recover_environment(
    #     eval_env=True,
    # )
    # print(eval_env)


if __name__ == "__main__":
    agents = ["hopper", "walker2d", "halfcheetah"]
    levels = ["simple", "expert", "medium"]

    datasets = []

    for agent in agents:
        for level in levels:
            datasets.append(f"mujoco/{agent}/{level}-v0")

    for dataset in datasets:
        download_dataset(dataset)
