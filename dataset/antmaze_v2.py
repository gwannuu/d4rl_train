from os import PathLike
from pathlib import Path
from typing import Mapping

import d4rl
import gym
import h5py
import numpy as np
from tqdm import tqdm

# dataset = [
#     "antmaze-large-diverse-v2",
#     "antmaze-large-play-v2",
#     "antmaze-medium-play-v2",
#     "antmaze-medium-diverse-v2",
#     "antmaze-umaze-v2",
#     "antmaze-umaze-diverse-v2",
# ]

ANTMAZE_DATASETS: dict[str, str] = {
    "antmaze-large-diverse-v2": "Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
    "antmaze-large-play-v2": "Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
    "antmaze-medium-play-v2": "Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
    "antmaze-medium-diverse-v2": "Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
    "antmaze-umaze-v2": "Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5",
    "antmaze-umaze-diverse-v2": "Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5",
}


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load_dataset_from_file(
    file_path: str | PathLike[str],
    observation_shape: tuple[int, ...] | None = None,
    action_shape: tuple[int, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Load an AntMaze dataset from disk following d4rl.offline_env logic."""

    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data_dict: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as dataset_file:
        for key in tqdm(get_keys(dataset_file), desc=f"load {path.name}"):
            data_array = dataset_file[key]
            if not isinstance(data_array, h5py.Dataset):
                raise TypeError(f"Key '{key}' did not resolve to an h5py.Dataset")
            try:
                data_dict[key] = data_array[:]
            except ValueError:
                data_dict[key] = data_array[()]

    required_keys = ["observations", "actions", "rewards", "terminals"]
    for key in required_keys:
        if key not in data_dict:
            raise KeyError(f"Dataset is missing key '{key}'")

    n_samples = data_dict["observations"].shape[0]
    if observation_shape is not None:
        obs_shape = data_dict["observations"].shape[1:]
        if obs_shape != observation_shape:
            raise ValueError(
                f"Observation shape mismatch: {obs_shape} vs expected {observation_shape}"
            )
    if action_shape is not None:
        act_shape = data_dict["actions"].shape[1:]
        if act_shape != action_shape:
            raise ValueError(
                f"Action shape mismatch: {act_shape} vs expected {action_shape}"
            )

    def squeeze_if_needed(key: str):
        arr = data_dict[key]
        if arr.shape == (n_samples, 1):
            data_dict[key] = arr[:, 0]
        elif arr.shape != (n_samples,):
            raise ValueError(f"{key} has wrong shape: {arr.shape}")

    squeeze_if_needed("rewards")
    squeeze_if_needed("terminals")
    return data_dict


def process_dataset_mapping(
    dataset_mapping: Mapping[str, str], dataset_dir: str | PathLike[str] | None
) -> dict[str, str | Path]:
    """Process dataset_mapping to determine loading sources.

    Returns a dict where values are either env_ids (for online loading) or file paths (for local loading).
    """
    if dataset_dir is None:
        return {env_id: env_id for env_id in dataset_mapping.keys()}

    root = Path(dataset_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    return {env_id: root / file_name for env_id, file_name in dataset_mapping.items()}


def load_datasets(sources: dict[str, str | Path]) -> dict[str, dict[str, np.ndarray]]:
    """Load datasets from the provided sources."""
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for env_id, source in sources.items():
        if isinstance(source, str):  # Online loading via env_id
            env = gym.make(source)
            data = d4rl.qlearning_dataset(env)
            loaded[env_id] = data
            print(f"Loaded {env_id} via d4rl API ({len(data['actions'])} transitions)")
        else:  # Local loading via file path
            data = load_dataset_from_file(source)
            loaded[env_id] = data
            print(f"Loaded {env_id}: {source} ({len(data['actions'])} transitions)")
    return loaded


def check_and_load_dataset(
    dataset_mapping: Mapping[str, str], dataset_dir: str | PathLike[str] | None
) -> dict[str, dict[str, np.ndarray]]:
    """Load every dataset referenced in dataset_mapping.

    If dataset_dir is provided, files are read locally. Otherwise, envs are
    instantiated via gym/d4rl so the datasets are fetched using the original
    API (which may download if missing).
    """
    sources = process_dataset_mapping(dataset_mapping, dataset_dir)
    return load_datasets(sources)


if __name__ == "__main__":
    dataset = load_dataset_from_file(
        file_path="/home/gwanwoo/datasets/d4rl/datasets/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5"
    )
    datasets = check_and_load_dataset(
        ANTMAZE_DATASETS, dataset_dir="/home/gwanwoo/datasets/d4rl/datasets"
    )
    print(datasets)
