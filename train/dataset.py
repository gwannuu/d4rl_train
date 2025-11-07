from dataclasses import dataclass

import d4rl
import gym
import numpy as np

# OBSERVATIONS = "observations"
# ACTIONS = "actions"
# NEXT_OBSERVATIONS = "next_observations"
# REWARDS = "rewards"
# DONE = "dones"


@dataclass(slots=True)
class ReplayBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray


def get_d4rl_dataset(env: str | gym.Env):
    if isinstance(env, str):
        env = gym.make(env)
    dataset = d4rl.qlearning_dataset(env)
    return ReplayBuffer(**dataset)


def sample_dataset(dataset: ReplayBuffer, size: int):
    indices = np.random.randint(len(dataset.observations), size=size)
    return ReplayBuffer(
        observations=dataset.observations[indices],
        actions=dataset.actions[indices],
        rewards=dataset.rewards[indices],
        next_observations=dataset.next_observations[indices],
        terminals=dataset.terminals[indices],
    )


class OfflineDataset:
    def __init__(self, env: str | gym.Env, batch_size: int):
        self.batch_size: int = batch_size
        self.dataset: ReplayBuffer = get_d4rl_dataset(env)

    def sample_batch(self) -> ReplayBuffer:
        return sample_dataset(self.dataset, self.batch_size)
