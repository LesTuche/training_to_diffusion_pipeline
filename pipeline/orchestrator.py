import importlib
import json
import os
import sys
import tomllib
from pathlib import Path

from mace.calculators import mace_mp
from mace.calculators.foundations_models import mace_mp_names

from pipeline import slurm_jobs
from pipeline.stage_utils import (load_site_config, mace_mp_kwargs, merge_stage_settings, setting_problems, write_json,
                                  write_text)

SLURM_KEYS = {"gpu_class", "cpus_per_task", "mem_per_cpu",
              "time", "sbatch_args", "poll_interval_s"}
SLURM_DEFAULT_TIME = "24:00:00"

cached_mace_calculator, cached_mace_calculator_kwargs = None, None


def run_stage_in_process(module_name, settings, output_dir, inputs):
    """Run one stage item in this process and return its output paths.

    The MACE calculator is built from the mace_mp kwargs in settings (none without "model") and reused while they are unchanged.
    """
    global cached_mace_calculator, cached_mace_calculator_kwargs
    requested_mace_kwargs = mace_mp_kwargs(settings)
    if "model" not in requested_mace_kwargs:
        cached_mace_calculator, cached_mace_calculator_kwargs = None, None
    elif requested_mace_kwargs != cached_mace_calculator_kwargs:
        cached_mace_calculator, cached_mace_calculator_kwargs = mace_mp(
            **requested_mace_kwargs), requested_mace_kwargs
    return importlib.import_module(f"pipeline.{module_name}").run(settings, output_dir, inputs, cached_mace_calculator)


def slurm_job_script(settings, job_dir, job_name):
    """job.sh text for running one stage item as a SLURM job.

    job_dir holds stage_input.json and receives the SLURM logs. Requests one GPU, of gpu_class if given, when device starts with
    "cuda"; the job_setup lines of site_config.toml run before the stage command.
    """
    slurm_settings = settings["slurm"]
    unknown_slurm_keys = set(slurm_settings) - SLURM_KEYS
    if unknown_slurm_keys:
        raise ValueError(
            f"unknown slurm keys {sorted(unknown_slurm_keys)}; allowed {sorted(SLURM_KEYS)}")
    requests_gpu = str(settings.get("device", "cpu")).startswith("cuda")
    if "gpu_class" in slurm_settings and not requests_gpu:
        raise ValueError(
            f"slurm.gpu_class given but device is {settings.get('device', 'cpu')!r}")
    job_dir_absolute = os.path.abspath(job_dir)
    job_script_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={slurm_settings.get('cpus_per_task', 1)}",
        f"#SBATCH --mem-per-cpu={slurm_settings.get('mem_per_cpu', '6G')}",
        f"#SBATCH --time={slurm_settings.get('time', SLURM_DEFAULT_TIME)}",
        # relative paths refer to the orchestrator's working directory, not job_dir
        f"#SBATCH --chdir={os.getcwd()}",
        f"#SBATCH --output={job_dir_absolute}/slurm_%j.out",
        f"#SBATCH --error={job_dir_absolute}/slurm_%j.err",
    ]
    if requests_gpu:
        job_script_lines.append(
            f"#SBATCH --gpus={slurm_settings['gpu_class']}:1" if "gpu_class" in slurm_settings else "#SBATCH --gpus=1")
    job_script_lines += [
        f"#SBATCH {extra_sbatch_line}" for extra_sbatch_line in slurm_settings.get("sbatch_args", [])]
    job_script_lines.append("set -euo pipefail")
    job_script_lines += load_site_config()["job_setup"]
    job_script_lines.append(
        f"{sys.executable} -m pipeline.orchestrator --run-stage {job_dir}")
    return "\n".join(job_script_lines) + "\n"


def check_pipeline(pipeline_config):
    """Check every stage before anything runs and raise with the full list of problems.

    Checks the settings and that every referenced file outside run_dir exists: explicit inputs, model files, PLUMED templates,
    exclude_provenance, POTCARs.
    """
    run_dir = os.path.abspath(pipeline_config["run_dir"])
    problems = []
    earlier_stage_names = []
    for stage_index, stage_config in enumerate(pipeline_config["stage"]):
        stage_settings = merge_stage_settings(pipeline_config, stage_config)
        log_tag = f"[{stage_index:02d}_{stage_settings.get('name')}]"
        if "name" not in stage_settings:
            problems.append(f"{log_tag} missing required setting 'name'")
        stage_module = importlib.import_module(
            f"pipeline.{stage_settings.pop('module')}")
        explicit_inputs = stage_settings.pop("inputs", [])
        referenced_stage_names = [
            entry for entry in explicit_inputs if entry in earlier_stage_names]
        input_files = [
            entry for entry in explicit_inputs if entry not in earlier_stage_names]
        ambiguous_inputs = [
            entry for entry in referenced_stage_names if os.path.exists(entry)]
        if ambiguous_inputs:
            problems.append(
                f"{log_tag} inputs {ambiguous_inputs} are both earlier stage names and existing files")
        earlier_stage_names.append(stage_settings.get("name"))
        item_settings_list = []
        if hasattr(stage_module, "items"):
            known_inputs = [] if referenced_stage_names else input_files
            item_settings_list = [item_settings for _, item_settings,
                                  _ in stage_module.items(stage_settings, known_inputs)]
        # items fed by earlier stages are unknown before the run; check the stage settings instead
        for item_settings in item_settings_list or [stage_settings]:
            problems += [f"{log_tag} {problem}" for problem in setting_problems(
                stage_module, item_settings)]

        referenced_files = list(input_files)
        model = stage_settings.get("model")
        if model not in mace_mp_names and not str(model).startswith("https:"):
            referenced_files.append(model)
        plumed_templates = stage_settings.get("plumed")
        if isinstance(plumed_templates, dict):
            referenced_files += list(plumed_templates.values())
        elif plumed_templates is not None:
            referenced_files.append(plumed_templates)
        if stage_settings.get("exclude_provenance") is not None:
            referenced_files.append(stage_settings["exclude_provenance"])
        for potcar_subdir in stage_settings.get("potcars", {}).values():
            referenced_files.append(os.path.join(
                load_site_config()["vasp"]["potcar_dir"], potcar_subdir, "POTCAR"))
        for referenced_file in referenced_files:
            inside_run_dir = os.path.commonpath(
                [os.path.abspath(referenced_file), run_dir]) == run_dir
            if not inside_run_dir and not os.path.exists(referenced_file):
                problems.append(f"{log_tag} file not found: {referenced_file}")
    if problems:
        raise RuntimeError(
            "pipeline config check failed:\n" + "\n".join(problems))


def move_aside(stage_dir):
    """Rename stage_dir to the first free <stage_dir>_old_<n> and return the new path."""
    old_run_number = 1
    while os.path.exists(f"{stage_dir}_old_{old_run_number}"):
        old_run_number += 1
    os.rename(stage_dir, f"{stage_dir}_old_{old_run_number}")
    return f"{stage_dir}_old_{old_run_number}"


def run_stage(module_name, settings, stage_dir, inputs):
    """Run all items of one stage, as SLURM jobs if settings has "slurm", otherwise in this process.

    Fan-out items run in <stage_dir>/<label>; items with an outputs.json are skipped. Raises if any item failed, otherwise writes
    <stage_dir>/outputs.json with the outputs of all items and returns them.
    """
    stage_module = importlib.import_module(f"pipeline.{module_name}")
    if hasattr(stage_module, "items"):
        stage_items = []
        for item_label, item_settings, item_inputs in stage_module.items(settings, inputs):
            item_dir = os.path.join(stage_dir, item_label)
            if not os.path.exists(item_dir):
                os.makedirs(item_dir)
                write_json(os.path.join(item_dir, "stage_input.json"), {
                           "module": module_name, "settings": item_settings, "inputs": item_inputs})
            stage_items.append(
                (item_label, item_settings, item_inputs, item_dir))
    else:
        stage_items = [(None, settings, inputs, stage_dir)]

    failure_reason_by_item_dir = {}
    if "slurm" in settings:
        for item_label, item_settings, _, item_dir in stage_items:
            if not os.path.exists(os.path.join(item_dir, "job.sh")):
                job_name = settings["name"] if item_label is None else f"{settings['name']}_{item_label}"
                write_text(os.path.join(item_dir, "job.sh"),
                           slurm_job_script(item_settings, item_dir, job_name))
        item_dirs = [item_dir for *_, item_dir in stage_items]
        poll_interval_s = settings["slurm"].get("poll_interval_s", 60)
        _, failed_job_state_by_dir = slurm_jobs.submit_and_wait_for_jobs(
            settings["name"], item_dirs, poll_interval_s)
        failure_reason_by_item_dir = {item_dir: f"SLURM job {job_state}, see {item_dir}/slurm_*.err"
                                      for item_dir, job_state in failed_job_state_by_dir.items()}
    else:
        for _, item_settings, item_inputs, item_dir in stage_items:
            if os.path.exists(os.path.join(item_dir, "outputs.json")):
                continue
            try:
                run_stage_in_process(
                    module_name, item_settings, item_dir, item_inputs)
            except Exception as error:
                failure_reason_by_item_dir[item_dir] = f"{type(error).__name__}: {error}"

    for item_dir, failure_reason in failure_reason_by_item_dir.items():
        print(f"WARNING: {item_dir} failed: {failure_reason}",
              file=sys.stderr, flush=True)
    if failure_reason_by_item_dir:
        raise RuntimeError(
            f"{settings['name']}: {len(failure_reason_by_item_dir)} of {len(stage_items)} items failed")
    stage_outputs = [output_path for *_, item_dir in stage_items
                     for output_path in json.loads(Path(item_dir, "outputs.json").read_text())]
    write_json(os.path.join(stage_dir, "outputs.json"), stage_outputs)
    return stage_outputs


def run_pipeline(pipeline_config):
    """Run the [[stage]] list of a config in order, skipping, resuming or rerunning each stage; returns the last stage's outputs.

    A stage gets its "inputs", with names of earlier stages replaced by their outputs, else the previous stage's outputs.
    """
    check_pipeline(pipeline_config)
    upstream_outputs = []
    outputs_by_stage_name = {}
    # inputs are compared by path, not content: once a stage has run, every later stage must rerun
    earlier_stage_ran = False
    for stage_index, stage_config in enumerate(pipeline_config["stage"]):
        stage_settings = merge_stage_settings(pipeline_config, stage_config)
        module_name = stage_settings.pop("module")
        explicit_inputs = stage_settings.pop("inputs", None)
        if explicit_inputs is not None:
            upstream_outputs = [input_path for entry in explicit_inputs
                                for input_path in outputs_by_stage_name.get(entry, [entry])]
        stage_dir = os.path.join(
            pipeline_config["run_dir"], f"{stage_index:02d}_{stage_settings['name']}")
        log_tag = f"[{stage_index:02d}_{stage_settings['name']}]"
        stage_outputs_file = os.path.join(stage_dir, "outputs.json")
        stage_input_file = os.path.join(stage_dir, "stage_input.json")
        # JSON round trip, so it compares equal to the stage_input.json read from disk
        current_stage_input = json.loads(json.dumps(
            {"module": module_name, "settings": stage_settings, "inputs": upstream_outputs}))
        recorded_stage_input = json.loads(
            Path(stage_input_file).read_text()) if os.path.exists(stage_input_file) else None
        if not earlier_stage_ran and recorded_stage_input == current_stage_input:
            if os.path.exists(stage_outputs_file):
                upstream_outputs = json.loads(
                    Path(stage_outputs_file).read_text())
                outputs_by_stage_name[stage_settings["name"]
                                      ] = upstream_outputs
                print(
                    f"{log_tag} {module_name} already done: {upstream_outputs}", flush=True)
                continue
            print(f"{log_tag} {module_name} resuming in {stage_dir}", flush=True)
        else:
            if os.path.exists(stage_dir):
                print(
                    f"{log_tag} {module_name} settings or inputs changed, moved old run to {move_aside(stage_dir)}", flush=True)
            os.makedirs(stage_dir)
            write_json(stage_input_file, current_stage_input)
            print(f"{log_tag} {module_name} -> {stage_dir}", flush=True)
        earlier_stage_ran = True
        upstream_outputs = run_stage(
            module_name, stage_settings, stage_dir, upstream_outputs)
        outputs_by_stage_name[stage_settings["name"]] = upstream_outputs
        print(f"{log_tag} done: {upstream_outputs}", flush=True)
    return upstream_outputs


if __name__ == "__main__":
    # entry point of the SLURM jobs written by slurm_job_script
    if sys.argv[1] == "--run-stage":
        stage_input = json.loads(
            Path(sys.argv[2], "stage_input.json").read_text())
        run_stage_in_process(
            stage_input["module"], stage_input["settings"], sys.argv[2], stage_input["inputs"])
        sys.exit(0)
    print(
        f"finished: {run_pipeline(tomllib.loads(Path(sys.argv[1]).read_text()))}", flush=True)
