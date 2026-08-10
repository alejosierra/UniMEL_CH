import argparse
from pathlib import Path
from omegaconf import OmegaConf


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True)
    known_args, remaining_args = parser.parse_known_args()

    overrides = []
    index = 0
    while index < len(remaining_args):
        token = remaining_args[index]
        if token.startswith('--'):
            token = token[2:]
            if '=' in token:
                overrides.append(token)
                index += 1
                continue

            if index + 1 < len(remaining_args) and not remaining_args[index + 1].startswith('-'):
                overrides.append(f'{token}={remaining_args[index + 1]}')
                index += 2
                continue

            overrides.append(f'{token}=true')
            index += 1
            continue

        overrides.append(token)
        index += 1

    args = OmegaConf.merge(OmegaConf.load(known_args.config), OmegaConf.from_dotlist(overrides))
    return args

def refine_checkpoint_path(checkpoint_path:Path):
    """
    Check if the path contains acheckpoint file (adapter_model.safetensors or adapter_model.bin) return it as it is.

    If not, this means its a swift fine tuning folder.

    It must find the latest folder (the name of the folders start v0-...,v1-...,v2-... etc). After finding the latest folder, it must read the logging.jsonl file, get the last line and the parse it as json.

    From that json, it must get the value of the key "best_model_checkpoint" and return it as an absolute path.
    """
    
    if (checkpoint_path / "adapter_model.safetensors").exists() or (checkpoint_path / "adapter_model.bin").exists():
        return checkpoint_path

    # Find the latest folder using numeric version parsing (handles multi-digit versions)
    import re

    candidates = []
    for p in checkpoint_path.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r'^v(\d+)', p.name)
        if m:
            ver = int(m.group(1))
            candidates.append((ver, p))

    if not candidates:
        raise FileNotFoundError(f"No valid versioned folders found in {checkpoint_path}")

    latest_folder = max(candidates, key=lambda x: x[0])[1]

    logging_file = latest_folder / "logging.jsonl"
    if not logging_file.exists():
        raise FileNotFoundError(f"Logging file not found in {latest_folder}")

    with open(logging_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            raise ValueError(f"Logging file {logging_file} is empty")

        last_line = lines[-1]
        import json
        log_data = json.loads(last_line)
        best_model_checkpoint = log_data.get("best_model_checkpoint")
        if not best_model_checkpoint:
            raise KeyError(f"'best_model_checkpoint' not found in the last log entry of {logging_file}")

    return Path(best_model_checkpoint).absolute()