import csv
import hashlib
import json
import os
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MACE_MP_KWARG_KEYS = ("model", "device", "default_dtype",
                      "head", "dispersion", "damping", "dispersion_xc")
SITE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "site_config.toml"
# keys every stage accepts in addition to its own REQUIRED and DEFAULTS
GENERAL_SETTING_KEYS = {"name", "slurm"}


def write_text(path, text):
    """Write text to path and fsync it."""
    with open(path, "w") as output_file:
        output_file.write(text)
        output_file.flush()
        os.fsync(output_file.fileno())


def write_json(path, data):
    """Write data as indented JSON to path and fsync it."""
    with open(path, "w") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.flush()
        os.fsync(output_file.fileno())


def write_csv(path, header, rows):
    """Write header and rows to a CSV file and fsync it."""
    with open(path, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(header)
        csv_writer.writerows(rows)
        output_file.flush()
        os.fsync(output_file.fileno())


def cache_key(**key_fields):
    """16-character hash of the JSON of key_fields; identical fields give the identical key."""
    return hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:16]


def trajectory_cache_fields(trajectory_path):
    """Path, size and modification time of a trajectory as cache_key fields; a trajectory overwritten under the same path gets a new key."""
    trajectory_stat = os.stat(trajectory_path)
    return {"traj": trajectory_path, "size": trajectory_stat.st_size, "mtime": trajectory_stat.st_mtime_ns}


def load_or_compute_cached(path, compute_arrays):
    """Dict of arrays from the npz at path if it exists, otherwise compute_arrays() saved to path."""
    if os.path.exists(path):
        print(f"using cache {path}", flush=True)
        with np.load(path) as cached_arrays:
            return {key: cached_arrays[key] for key in cached_arrays.files}
    computed_arrays = compute_arrays()
    # write and rename, so an interruption never leaves a truncated cache file
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "wb") as output_file:
        np.savez(output_file, **computed_arrays)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)
    return computed_arrays


def savefig_fsync(figure, path, **savefig_kwargs):
    """Save figure to path, close it and fsync the file."""
    figure.savefig(path, **savefig_kwargs)
    plt.close(figure)
    with open(path, "rb") as saved_file:
        os.fsync(saved_file.fileno())


def frames_from_ps(time_ps, dt_fs):
    """Number of frames closest to time_ps for a frame spacing of dt_fs."""
    return round(time_ps * 1000.0 / dt_fs)


def deep_merge(base, override):
    """Copy of base with override merged in; nested dicts are merged recursively, other values replaced. Neither input is modified."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_stage_settings(pipeline_config, stage_config):
    """Settings of one [[stage]]: its presets deep-merged in order, then the stage's own keys on top, without "presets"."""
    merged_settings = {}
    for preset_name in stage_config.get("presets", []):
        merged_settings = deep_merge(
            merged_settings, pipeline_config["preset"][preset_name])
    return deep_merge(merged_settings, {key: value for key, value in stage_config.items() if key != "presets"})


def standalone_settings(config_path, module_name, command_line_inputs):
    """Settings and inputs of a standalone run: the one [[stage]] of module_name in a pipeline config, merged with its presets.

    The inputs are command_line_inputs, or the stage's inputs if none are given on the command line.
    """
    pipeline_config = tomllib.loads(Path(config_path).read_text())
    stage_configs = [stage_config for stage_config in pipeline_config["stage"]
                     if stage_config["module"] == module_name]
    if len(stage_configs) != 1:
        raise ValueError(
            f"{config_path}: a standalone {module_name} run needs exactly one [[stage]] of it, found {len(stage_configs)}")
    settings = merge_stage_settings(pipeline_config, stage_configs[0])
    settings.pop("module")
    stage_inputs = settings.pop("inputs", [])
    return settings, command_line_inputs or stage_inputs


def load_site_config():
    """Machine-specific settings from site_config.toml in the repository root; raises if the file does not exist."""
    if not SITE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{SITE_CONFIG_PATH} not found; copy site_config.template.toml to site_config.toml and adapt it")
    return tomllib.loads(SITE_CONFIG_PATH.read_text())


def mace_mp_kwargs(settings):
    """Keyword arguments for mace_mp: the keys of MACE_MP_KWARG_KEYS present in settings."""
    return {key: settings[key] for key in MACE_MP_KWARG_KEYS if key in settings}


def setting_problems(stage_module, settings):
    """List of problems with a stage's settings: unknown keys, missing required keys and the module's nested checks; empty if valid."""
    allowed_keys = set(stage_module.REQUIRED) | set(
        stage_module.DEFAULTS) | GENERAL_SETTING_KEYS
    if stage_module.USES_MACE_CALCULATOR:
        allowed_keys |= set(MACE_MP_KWARG_KEYS)
    problems = []
    unknown_keys = set(settings) - allowed_keys
    if unknown_keys:
        problems.append(f"unknown settings {sorted(unknown_keys)}")
    missing_keys = set(stage_module.REQUIRED) - set(settings)
    if missing_keys:
        problems.append(f"missing required settings {sorted(missing_keys)}")
    if hasattr(stage_module, "nested_setting_problems"):
        problems += stage_module.nested_setting_problems(settings)
    return problems


def write_merged_settings(stage_module, settings, output_dir, inputs):
    """Check a stage's settings, merge them over the module's DEFAULTS and write <output_dir>/<name>_settings.json.

    Raises on any setting problem. Returns (merged settings, output prefix <output_dir>/<name>).
    """
    problems = setting_problems(stage_module, settings)
    if problems:
        raise ValueError(f"{settings['name']}: " + "; ".join(problems))
    merged_settings = {**stage_module.DEFAULTS, **settings}
    output_prefix = os.path.join(output_dir, merged_settings["name"])
    write_json(f"{output_prefix}_settings.json", {
               **merged_settings, "inputs": inputs})
    return merged_settings, output_prefix


def one_item_per_input(settings, inputs):
    """Fan-out hook: one item per input file, labelled and named by the file name without extension.

    Returns [(label, item settings, [input path])]; raises if two inputs give the same label.
    """
    item_labels = [os.path.splitext(os.path.basename(input_path))[
        0] for input_path in inputs]
    if len(set(item_labels)) != len(item_labels):
        raise ValueError(
            f"inputs {inputs} give non-unique labels {item_labels}")
    return [(item_label, {**settings, "name": item_label}, [input_path]) for item_label, input_path in zip(item_labels, inputs)]


def split_name(path):
    """Split name of a file: the text after the last "_" of the file name without extension (selection_train.traj -> train)."""
    return os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]


def unique_split_names(split_files):
    """Split name of each file, in input order; raises if two files give the same split name."""
    split_names = [split_name(split_file) for split_file in split_files]
    if len(set(split_names)) != len(split_names):
        raise ValueError(
            f"frame files {split_files} give non-unique split names {split_names}")
    return split_names
