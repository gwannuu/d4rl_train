import d4rl
import gym


def check_and_download_dataset(dataset: list[str]):
    for d in dataset:
        env = gym.make(d)
        data = d4rl.qlearning_dataset(env)
        print(f"d4rl {d} dataset shape: {len(data['actions'])}")


if __name__ == "__main__":
    dataset = [
        "antmaze-large-diverse-v2",
        "antmaze-large-play-v2",
        "antmaze-medium-play-v2",
        "antmaze-medium-diverse-v2",
        "antmaze-umaze-v2",
        "antmaze-umaze-diverse-v2",
    ]
    check_and_download_dataset(dataset)
