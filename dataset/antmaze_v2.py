import os
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
    *,
    env_id: str,
    observation_shape: tuple[int, ...] | None = None,
    action_shape: tuple[int, ...] | None = None,
    terminate_on_end: bool = False,
) -> dict[str, np.ndarray]:
    """Load an AntMaze HDF5 dataset and reproduce d4rl.qlearning_dataset output.

    The dataset is first loaded exactly like d4rl.offline_env.OfflineEnv.get_dataset
    and then passed to d4rl.qlearning_dataset for processing so the returned
    dictionary matches the online-loading behavior bit-for-bit.
    """

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
            # For arrays like observations/actions we expect a 2D shape, so
            # only rewards/terminals are squeezed; for others we keep original.
            return

    squeeze_if_needed("rewards")
    squeeze_if_needed("terminals")

    env = gym.make(env_id)
    try:
        processed = d4rl.qlearning_dataset(
            env,
            dataset=data_dict,
            terminate_on_end=terminate_on_end,
        )
    finally:
        env.close()

    return processed


def load_dataset_via_d4rl(
    env_id: str,
    *,
    terminate_on_end: bool = False,
) -> dict[str, np.ndarray]:
    """Fetch a dataset directly from d4rl for cases where local files are absent."""

    env = gym.make(env_id)
    try:
        return d4rl.qlearning_dataset(
            env,
            terminate_on_end=terminate_on_end,
        )
    finally:
        env.close()


def get_dataset_file_path(
    env_id: str, dataset_dir: str | PathLike[str] | None
) -> Path:
    """Return the local HDF5 path for a dataset.

    If dataset_dir is a directory, the path is resolved via ANTMAZE_DATASETS.
    If it is a file, that file is returned as-is. Raises when anything is
    missing to keep the caller honest.
    """

    if dataset_dir is None:
        raise ValueError("dataset_dir must be provided for local loading")

    candidate = Path(dataset_dir).expanduser()
    if candidate.is_file():
        return candidate

    if not candidate.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {candidate}")

    if env_id not in ANTMAZE_DATASETS:
        raise KeyError(f"Unknown AntMaze dataset: {env_id}")

    file_path = candidate / ANTMAZE_DATASETS[env_id]
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found for '{env_id}': {file_path}"
        )

    return file_path


# def load_antmaze_dataset(
#     env_id: str,
#     dataset_dir: str | PathLike[str] | None,
#     observation_shape: tuple[int, ...] | None = None,
#     action_shape: tuple[int, ...] | None = None,
# ) -> dict[str, np.ndarray]:
#     """Load a single dataset either from disk or via the default API."""

#     if dataset_dir is None:
#         env = gym.make(env_id)
#         try:
#             data = d4rl.qlearning_dataset(env)
#         finally:
#             env.close()
#         print(
#             f"Loaded {env_id} via d4rl API ({len(data['actions'])} transitions)"
#         )
#         return data

#     file_path = get_dataset_file_path(env_id=env_id, dataset_dir=dataset_dir)
#     print(f"Loading {env_id} from {file_path}")
#     return load_dataset_from_file(
#         file_path=file_path,
#         env_id=env_id,
#         observation_shape=observation_shape,
#         action_shape=action_shape,
#     )


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


def load_datasets(
    env_id: str,
    source: str | Path,
    *,
    allow_online: bool = False,
) -> dict[str, np.ndarray]:
    """Load a single dataset from a provided source.

    This accepts an `env_id` and a `source` which may be either an env id
    (string) for online/d4rl loading or a local `Path`/file path for HDF5.
    Returns the dataset dict for that env.
    """
    # Prefer local file if the source resolves to an existing path.
    file_path = Path(source)
    if file_path.exists():
        data = load_dataset_from_file(file_path, env_id=env_id)
        print(f"Loaded {env_id}: {file_path} ({len(data['actions'])} transitions)")
        return data

    # Otherwise optionally allow online loading.
    if allow_online and isinstance(source, str):
        data = load_dataset_via_d4rl(source)
        print(f"Loaded {env_id} via d4rl API ({len(data['actions'])} transitions)")
        return data

    raise FileNotFoundError(
        f"Dataset source not found for {env_id}: {source}. "
        "Pass allow_online=True explicitly if remote download is intended."
    )


def check_and_load_datasets(
    dataset_mapping: Mapping[str, str],
    dataset_dir: str | PathLike[str] | None,
    *,
    allow_online: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Load every dataset referenced in dataset_mapping by delegating to
    `load_datasets` for each single-source mapping.

    Returns a dict mapping env_id -> loaded dataset dict.
    """
    sources = process_dataset_mapping(dataset_mapping, dataset_dir)
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for env_id, src in sources.items():
        loaded_dataset = load_datasets(env_id, src, allow_online=allow_online)
        loaded[env_id] = loaded_dataset
    return loaded


if __name__ == "__main__":
    datasets = check_and_load_datasets(
        ANTMAZE_DATASETS, dataset_dir="/home/gwanwoo/datasets/d4rl"
    )
    print(datasets)
