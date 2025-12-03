import os
from typing import Literal

import numpy as np
import ogbench


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
    return env, train_dataset, val_dataset


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
    env, train_dataset, val_dataset = load_env_and_datasets(
        dataset_name=dataset_name,
        render_mode="human",
    )
    print(train_dataset["actions"].shape)
