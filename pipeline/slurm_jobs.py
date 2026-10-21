import os
import subprocess
import time
from pathlib import Path

from pipeline.stage_utils import write_text

ACTIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED"}


def submit_job(job_dir):
    """Submit job_dir/job.sh with sbatch from inside job_dir and write the job ID to job_dir/job_id."""
    sbatch_result = subprocess.run(["sbatch", "--parsable", "job.sh"], cwd=job_dir, capture_output=True, text=True)
    if sbatch_result.returncode != 0:
        raise RuntimeError(f"sbatch failed in {job_dir}: {sbatch_result.stderr.strip()}")
    # --parsable prints "<id>" or "<id>;<cluster>" on multi-cluster setups
    write_text(os.path.join(job_dir, "job_id"), sbatch_result.stdout.strip().split(";")[0] + "\n")


def query_job_states(job_ids):
    """{job ID: SLURM state} from sacct; a job that sacct does not know yet is missing."""
    # with -j, sacct searches from Unix epoch 0 instead of only jobs since midnight
    sacct_result = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(job_ids)], capture_output=True, text=True)
    if sacct_result.returncode != 0:
        raise RuntimeError(f"sacct failed: {sacct_result.stderr.strip()}")
    state_by_job_id = {}
    for sacct_line in sacct_result.stdout.splitlines():
        job_id, state_text = sacct_line.split("|")
        # the state can carry extra text, e.g. "CANCELLED by <uid>"; keep only the state name
        state_by_job_id[job_id] = state_text.split()[0]
    return state_by_job_id


def submit_and_wait_for_jobs(stage_name, job_dirs, poll_interval_s):
    """Submit every job not submitted yet and poll sacct until no job is pending or running.

    A job succeeded if it wrote job_dir/outputs.json. A job that had already failed before this call is resubmitted once; a job
    that fails during this call stays failed.
    Returns ({job_dir: job ID}, {failed job_dir: SLURM state}).
    """
    submitted_in_this_call = set()
    while True:
        job_id_by_dir = {job_dir: (Path(job_dir) / "job_id").read_text().strip() for job_dir in job_dirs
                         if os.path.exists(os.path.join(job_dir, "job_id"))}
        state_by_job_id = query_job_states(list(job_id_by_dir.values())) if job_id_by_dir else {}
        done_dirs, active_dirs, failed_state_by_dir, dirs_to_submit = [], [], {}, []
        for job_dir in job_dirs:
            if os.path.exists(os.path.join(job_dir, "outputs.json")):
                done_dirs.append(job_dir)
            elif job_dir not in job_id_by_dir:
                dirs_to_submit.append(job_dir)
            # a job submitted moments ago can be missing from sacct; count it as pending
            elif state_by_job_id.get(job_id_by_dir[job_dir], "PENDING") in ACTIVE_STATES:
                active_dirs.append(job_dir)
            elif job_dir in submitted_in_this_call:
                failed_state_by_dir[job_dir] = state_by_job_id[job_id_by_dir[job_dir]]
            else:
                print(f"{stage_name}: resubmitting {job_dir} (previous job {job_id_by_dir[job_dir]} "
                      f"{state_by_job_id[job_id_by_dir[job_dir]]})", flush=True)
                dirs_to_submit.append(job_dir)
        for job_dir in dirs_to_submit:
            submit_job(job_dir)
            submitted_in_this_call.add(job_dir)
        print(f"{stage_name}: {len(done_dirs)} done, {len(failed_state_by_dir)} failed, "
              f"{len(active_dirs) + len(dirs_to_submit)} active", flush=True)
        if not dirs_to_submit and not active_dirs:
            return {job_dir: (Path(job_dir) / "job_id").read_text().strip() for job_dir in job_dirs}, failed_state_by_dir
        time.sleep(poll_interval_s)
