import hashlib
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

# https://github.com/google/flax/issues/4423


def _compute_state_hash(state: Any) -> str:
    """Compute SHA256 hash of state pytree for comparison."""
    import pickle

    state_bytes = pickle.dumps(state)
    return hashlib.sha256(state_bytes).hexdigest()


def save_state(
    checkpointer: ocp.StandardCheckpointer,
    obj: nnx.Module | nnx.Optimizer,
    path,
    overwrite: bool = False,
):
    """
    Save state to checkpoint.

    Returns True if saved, False if skipped (already exists with same content).
    Raises ValueError if destination exists with different content.
    """
    path = Path(path)
    state = nnx.split(obj)[1]
    new_hash = _compute_state_hash(state)

    # Check if destination already exists
    if path.exists():
        try:
            old_state = checkpointer.restore(
                path, target=nnx.split(nnx.eval_shape(lambda: obj))[1]
            )
            old_hash = _compute_state_hash(old_state)

            if new_hash == old_hash:
                print(f"Checkpoint already exists with identical content: {path}")
                return
            else:
                raise ValueError(
                    f"Checkpoint {path} exists but content differs. "
                    f"Old hash: {old_hash}, New hash: {new_hash}. "
                    f"This suggests a code bug or training divergence."
                )
        except Exception as e:
            if "content differs" in str(e):
                raise
            # If restore fails, fall through and try to overwrite
            if not overwrite:
                raise ValueError(
                    f"Destination {path} exists but cannot be read. "
                    f"Pass overwrite=True to force save. Error: {e}"
                )

    # Save new state
    checkpointer.save(path, state=state)
    checkpointer.wait_until_finished()
    print(f"✓ Saved checkpoint: {path}")
    return


def restore_state(
    checkpointer: ocp.StandardCheckpointer, obj: nnx.Module | nnx.Optimizer, dir: Path
):
    graphdef, abs_state = nnx.split(nnx.eval_shape(lambda: obj))
    state = checkpointer.restore(dir, abs_state)
    restored = nnx.merge(graphdef, state)
    return restored


def nnx_conditional_jit(cond, *args, **kwargs):
    """JIT decorator that can be disabled for debugging."""
    if cond:

        def identity_decorator(f):
            return f

        return identity_decorator
    else:
        return nnx.jit(*args, **kwargs)
