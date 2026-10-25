import ase
import numpy as np
from packaging.version import Version
from ase import units
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from tqdm import tqdm

ASE_MAX_VERSION_CHECKED = "3.30.0b1"
if Version(ase.__version__) > Version(ASE_MAX_VERSION_CHECKED):
    raise RuntimeError(
        f"ASE {ase.__version__} > {ASE_MAX_VERSION_CHECKED}; re-check the Bussi ndof patch")


def relax_with_bfgs(atoms, mace_calculator, fmax, max_steps, output_prefix, progress_label):
    """Relax atoms in place with BFGS; stopping unconverged at max_steps is not an error.

    Writes <output_prefix>_opt.traj and _opt.log.
    Returns (converged, number of steps, final max force in eV/Å).
    """
    atoms.calc = mace_calculator
    bfgs_optimizer = BFGS(
        atoms, trajectory=f"{output_prefix}_opt.traj", logfile=f"{output_prefix}_opt.log")
    progress_bar = tqdm(total=max_steps, unit="step", desc=progress_label)
    bfgs_optimizer.attach(lambda: progress_bar.update(
        bfgs_optimizer.nsteps - progress_bar.n))
    converged = bfgs_optimizer.run(fmax=fmax, steps=max_steps)
    progress_bar.close()
    final_max_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
    return converged, bfgs_optimizer.nsteps, final_max_force


def run_bussi_md(atoms, mace_calculator, temperature_K, timestep_fs, taut_fs, nsteps, write_interval_fs, log_interval_fs,
                 remove_rigid_body_interval_fs, random_number_generator, output_prefix, progress_label):
    """Bussi NVT MD of an isolated system, in place, with fresh Maxwell-Boltzmann velocities at temperature_K.

    Centre-of-mass translation and rotation are removed every remove_rigid_body_interval_fs. random_number_generator draws the
    velocities and the thermostat noise. Writes <output_prefix>.traj and <output_prefix>.log.
    """
    atoms.calc = mace_calculator

    def remove_com_translation_and_rotation():
        Stationary(atoms)
        ZeroRotation(atoms)

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, rng=random_number_generator)
    remove_com_translation_and_rotation()

    bussi_dynamics = Bussi(atoms, timestep=timestep_fs * units.fs, temperature_K=temperature_K, taut=taut_fs * units.fs,
                           rng=random_number_generator)

    # Bussi counts 3N dof, but the 6 rigid-body modes are removed; constraints are already counted by ASE
    ase_thermostat_ndof = bussi_dynamics.ndof
    bussi_dynamics.ndof = atoms.get_number_of_degrees_of_freedom() - 6
    bussi_dynamics.target_kinetic_energy = 0.5 * \
        bussi_dynamics.temp * bussi_dynamics.ndof
    print(f"{progress_label}: T {temperature_K} K, dt {timestep_fs} fs, taut {taut_fs} fs, nsteps {nsteps} "
          f"({nsteps * timestep_fs / 1000:g} ps), thermostat ndof ASE {ase_thermostat_ndof} -> patched {bussi_dynamics.ndof}", flush=True)

    progress_bar = tqdm(total=nsteps, unit="step", desc=progress_label)

    def show_internal_temperature():
        internal_temperature_K = 2 * atoms.get_kinetic_energy() / \
            (bussi_dynamics.ndof * units.kB)
        progress_bar.set_postfix(
            T_int=f"{internal_temperature_K:.0f} K", refresh=False)

    trajectory_writer = Trajectory(f"{output_prefix}.traj", "w", atoms)
    log_interval_steps = round(log_interval_fs / timestep_fs)
    bussi_dynamics.attach(MDLogger(
        bussi_dynamics, atoms, f"{output_prefix}.log", mode="w"), interval=log_interval_steps)
    bussi_dynamics.attach(trajectory_writer, interval=round(
        write_interval_fs / timestep_fs))
    bussi_dynamics.attach(remove_com_translation_and_rotation, interval=round(
        remove_rigid_body_interval_fs / timestep_fs))
    bussi_dynamics.attach(lambda: progress_bar.update(
        bussi_dynamics.nsteps - progress_bar.n))
    bussi_dynamics.attach(show_internal_temperature,
                          interval=log_interval_steps)
    bussi_dynamics.run(nsteps)
    progress_bar.close()
    trajectory_writer.close()
