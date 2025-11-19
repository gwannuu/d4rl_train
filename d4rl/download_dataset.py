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
        "halfcheetah-random-v0",
        "halfcheetah-medium-v0",
        "halfcheetah-expert-v0",
        "halfcheetah-medium-replay-v0",
        "halfcheetah-medium-expert-v0",
        "walker2d-random-v0",
        "walker2d-medium-v0",
        "walker2d-expert-v0",
        "walker2d-medium-replay-v0",
        "walker2d-medium-expert-v0",
        "hopper-random-v0",
        "hopper-medium-v0",
        "hopper-expert-v0",
        "hopper-medium-replay-v0",
        "hopper-medium-expert-v0",
        "ant-random-v0",
        "ant-medium-v0",
        "ant-expert-v0",
        "ant-medium-replay-v0",
        "ant-medium-expert-v0",
    ]
    check_and_download_dataset(dataset)
