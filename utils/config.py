import hashlib
import json
import subprocess


def get_git_hash(length: int | None = None):
    stream = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )
    if length is not None:
        stream = stream[:length]
    return stream


def generate_experiment_hash(config_dict: dict, length) -> str:
    # git_hash = get_git_hash()
    config_string = json.dumps(config_dict, sort_keys=True)

    hash_object = hashlib.sha256(f"{config_string}".encode("utf-8"))
    return hash_object.hexdigest()[:length]
