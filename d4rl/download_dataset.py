import d4rl
import gym


def check_and_download_dataset(dataset: list[str]):
    for d in dataset:
        env = gym.make(d)
        data = d4rl.qlearning_dataset(env)
        print(f"d4rl {d} dataset shape: {len(data['actions'])}")


if __name__ == "__main__":
    dataset = [
        # "antmaze-large-diverse-v2",
        # "antmaze-large-play-v2",
        # "antmaze-medium-play-v2",
        # "antmaze-medium-diverse-v2",
        # "antmaze-umaze-v2",
        # "antmaze-umaze-diverse-v2",
        "halfcheetah-random-v2",
        "halfcheetah-medium-v2",
        "halfcheetah-expert-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-expert-v2",
        "walker2d-random-v2",
        "walker2d-medium-v2",
        "walker2d-expert-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-expert-v2",
        "hopper-random-v2",
        "hopper-medium-v2",
        "hopper-expert-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-expert-v2",
        "ant-random-v2",
        "ant-medium-v2",
        "ant-expert-v2",
        "ant-medium-replay-v2",
        "ant-medium-expert-v2",
    ]
    check_and_download_dataset(dataset)
