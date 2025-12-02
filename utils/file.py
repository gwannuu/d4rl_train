import shutil
from pathlib import Path


def clear_ckpt_dir():
    ckpt_root = Path(__file__).resolve().parent.parent / "ckpt"
    if not ckpt_root.exists():
        return
    for child in ckpt_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
