import os
from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from flax.core import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = (
            False  # Whether to additionally return next actions; set outside the class.
        )
        # Compute terminal and initial locations.

        terminals = cast(np.ndarray, self["terminals"])
        self.terminal_locs = np.nonzero(terminals > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, idxs, side="right") - 1
            ]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(
                    jax.tree_util.tree_map(
                        lambda arr: arr[cur_idxs], self["observations"]
                    )
                )
                if i != self.frame_stack - 1:
                    next_obs.append(
                        jax.tree_util.tree_map(
                            lambda arr: arr[cur_idxs], self["observations"]
                        )
                    )
            next_obs.append(
                jax.tree_util.tree_map(lambda arr: arr[idxs], self["next_observations"])
            )

            batch["observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *obs
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *next_obs
            )
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding))
                if len(arr.shape) == 4
                else arr,
                batch[key],
            )


def make_maze_task_name(
    agent: Literal["point", "ant", "humanoid"],
    maze_type: Literal["medium", "large", "giant"],
    dataset_type: Literal["navigate", "stitch", "explore"],
    task_id: int | None = None,
    visual: bool = False,
) -> str:
    dataset_name = ""
    if visual:
        dataset_name += "visual-"
    dataset_name += f"{agent}maze-{maze_type}-{dataset_type}"
    if task_id is not None:
        assert 1 <= task_id <= 5, "task_id must be in [1, 5]"
        dataset_name += f"-singletask-task{task_id}"
    dataset_name += "-v0"
    return dataset_name


def make_antsoccer_task_name(
    maze_type: Literal["arena", "medium"],
    dataset_type: Literal["navigate", "stitch"],
    task_id: int | None = None,
):
    dataset_name = f"antsoccer-{maze_type}-{dataset_type}"
    if task_id is not None:
        assert 1 <= task_id <= 5, "task_id must be in [1, 5]"
        dataset_name += f"-singletask-task{task_id}"
    dataset_name += "-v0"
    return dataset_name


# def make_cube_task_name(
#     task_type: Literal["single", "double", "triple", "quadruple"],

# )


def download_ogbench_datasets(dataset_names, dataset_dir=".ogbench/data"):
    resolved_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    ogbench.download_datasets(dataset_names, dataset_dir=resolved_dir)
    print(f"OGBench datasets saved to: {resolved_dir}")
    return resolved_dir


def load_env_and_datasets(
    dataset_name: str,
    dataset_dir: str = ".ogbench/data",
    render_mode: str | None = None,
    action_clip_eps: float | None = 1e-5,
):
    resolved_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    result = ogbench.make_env_and_datasets(
        dataset_name,
        dataset_dir=resolved_dir,
        env_only=False,
        dataset_only=False,
        render_mode=render_mode,
    )

    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError(
            f"Expected (train_dataset, val_dataset) tuple from make_env_and_datasets "
            f"with dataset_only=True, got: {type(result)} / {result}"
        )

    env, train_dataset, val_dataset = result
    if action_clip_eps is not None:
        train_dataset["actions"] = np.clip(
            train_dataset["actions"],
            -1 + action_clip_eps,
            1 - action_clip_eps,
        )
        if val_dataset is not None:
            val_dataset["actions"] = np.clip(
                val_dataset["actions"],
                -1 + action_clip_eps,
                1 - action_clip_eps,
            )

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    eval_env = ogbench.make_env_and_datasets(
        dataset_name,
        env_only=True,
    )
    return env, eval_env, train_dataset, val_dataset


if __name__ == "__main__":
    # download_ogbench_datasets(
    #     [
    #         "humanoidmaze-medium-navigate-v0",
    #         "visual-puzzle-3x3-play-v0",
    #         "powderworld-easy-play-v0",
    #     ],
    #     dataset_dir=".ogbench/data",
    # )

    # dataset_name = make_maze_name(
    #     agent="ant",
    #     maze_type="medium",
    #     dataset_type="navigate",
    #     task_id=5,
    #     visual=True,
    # )

    dataset_name = "antsoccer-medium-stitch-singletask-task1-v0"
    env, eval_env, train_dataset, val_dataset = load_env_and_datasets(
        dataset_name=dataset_name,
        render_mode="human",
    )
    print(train_dataset.sample(1).shape)
