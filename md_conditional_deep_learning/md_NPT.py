#!/usr/bin/env python
"""
all_in_one_openmm_htmd.py

A single script that:
1) Reads a protein PDB file.
2) Reads all ligand PDB files in a specified folder.
3) For each ligand:
   - Combines the protein and ligand using a Modeller.
   - Adds missing hydrogens.
   - Solvates the system.
   - Builds the system and runs an MD simulation (using NPT optionally, with HBonds constraints so that the peptide is flexible).
   - At every data collection interval, records:
       • Total potential energy (kJ/mol)
       • RMSD relative to the initial conformation (nm)
       • Δ Binding energy (protein-ligand interaction, relative to the first frame, kJ/mol)
       • Positions (coordinates)
   - Writes out a CSV file with per-step simulation info.
   - Saves an NPZ file (training_data.npz) containing steps, energies, RMSDs, Δ binding energies, and positions.
4) Also writes a global CSV that collects energies across ligands.
5) Runs each ligand simulation in parallel (using GPU if available).

Usage:
  python all_in_one_openmm_htmd.py
"""

import os
import glob
import csv
import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from openmm.app import *
from openmm import *
from openmm.unit import *

from openmmforcefields.generators import GAFFTemplateGenerator

# ------------------------
#  USER-ADJUSTABLE PARAMS
# ------------------------
PROTEIN_PDB = "1a1m_A.pdb"          # Protein PDB file
LIGANDS_DIR = "./peptide-tests"     # Directory holding ligand PDB files

# Simulation parameters
NUM_STEPS   = 10000                # Total number of MD steps (for demonstration)
TEMPERATURE = 300 * kelvin

# Standard Amber force fields
FORCEFIELD_FILES = ['amber14-all.xml', 'amber14/tip3pfb.xml']
NONBONDED_CUTOFF = 1.0 * nanometer

# Flags to control simulation behavior:
USE_NPT = False                   # Use NPT ensemble (adds MonteCarloBarostat) if True
REMOVE_CONSTRAINTS = False        # Using HBonds constraints to allow a larger timestep and keep peptide flexible
SAVE_FRAMES = False               # Save each frame as a PDB file (optional)
GENERATE_TRAJECTORY = False       # Merge saved frames into a trajectory (if SAVE_FRAMES is True)

# Data collection: update state every COLLECTION_INTERVAL steps.
COLLECTION_INTERVAL = 10          # Collect every 10 steps

# Integrator parameters:
TIMESTEP = 0.002 * picoseconds    # 2 fs timestep (with constraints enabled)
LOW_FRICTION = 0.001 / picosecond # Lower friction for faster diffusion

# Reporting frequency (print progress every PROGRESS_INTERVAL steps)
PROGRESS_INTERVAL = max(1, NUM_STEPS // 100)

# Main output directory for simulation files
MD_OUTPUT_DIR = "md_output"

# ------------------------
#  Helper functions
# ------------------------
def positions_to_numpy(positions):
    """Convert OpenMM positions to a NumPy array (in nm)."""
    return np.array([[p.x, p.y, p.z] for p in positions.value_in_unit(nanometers)])

def compute_rmsd(ref, current):
    """
    Compute RMSD between two sets of coordinates using the Kabsch algorithm.
    Both ref and current are numpy arrays of shape (N, 3).
    A small regularization is applied if SVD fails.
    """
    ref_centroid = np.mean(ref, axis=0)
    current_centroid = np.mean(current, axis=0)
    ref_centered = ref - ref_centroid
    current_centered = current - current_centroid
    C = np.dot(ref_centered.T, current_centered)
    try:
        V, S, Wt = np.linalg.svd(C)
    except np.linalg.LinAlgError:
        epsilon = 1e-8
        C_reg = C + epsilon * np.eye(C.shape[0])
        V, S, Wt = np.linalg.svd(C_reg)
    d = np.sign(np.linalg.det(np.dot(Wt.T, V.T)))
    D = np.diag([1, 1, d])
    U = np.dot(Wt.T, np.dot(D, V.T))
    current_aligned = np.dot(current_centered, U)
    diff = ref_centered - current_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

def get_num_gpus():
    """Return the number of GPUs available (using nvidia-smi) or 0 if none."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        )
        devices = output.decode("utf-8").strip().split("\n")
        return len(devices)
    except Exception:
        return 0

def merge_pdb_frames(ligand_folder):
    """
    Merge individual PDB frames (named frame_XXX.pdb) in ligand_folder into a single
    multi-model PDB file (trajectory.pdb) using the MODEL/ENDMDL format.
    """
    pdb_files = sorted(glob.glob(os.path.join(ligand_folder, "frame_*.pdb")))
    if not pdb_files:
        print(f"No PDB frames found in {ligand_folder}.")
        return None
    traj_file = os.path.join(ligand_folder, "trajectory.pdb")
    with open(traj_file, "w") as outfile:
        for i, pdb_file in enumerate(pdb_files):
            outfile.write(f"MODEL     {i+1}\n")
            with open(pdb_file, "r") as infile:
                for line in infile:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        outfile.write(line)
            outfile.write("ENDMDL\n")
    print(f"Trajectory file created for ligand in {ligand_folder}: {traj_file}")
    return traj_file

def compute_binding_energy(positions, nb_force, num_protein_atoms, num_ligand_atoms):
    """
    Compute the binding energy between protein and ligand.
    positions: numpy array of shape (N_total, 3) in nm.
    nb_force: the NonbondedForce used in the system.
    num_protein_atoms: number of protein atoms.
    num_ligand_atoms: number of ligand atoms.
    
    Returns the sum of Coulomb and Lennard-Jones interactions for protein-ligand pairs.
    """
    coulomb_const = 138.935456  # kJ mol^-1 nm e^-2
    binding_energy = 0.0
    for i in range(num_protein_atoms):
        qi, sigma_i, epsilon_i = nb_force.getParticleParameters(i)
        qi = qi.value_in_unit(elementary_charge)
        sigma_i = sigma_i.value_in_unit(nanometer)
        epsilon_i = epsilon_i.value_in_unit(kilojoules_per_mole)
        for j in range(num_protein_atoms, num_protein_atoms + num_ligand_atoms):
            qj, sigma_j, epsilon_j = nb_force.getParticleParameters(j)
            qj = qj.value_in_unit(elementary_charge)
            sigma_j = sigma_j.value_in_unit(nanometer)
            epsilon_j = epsilon_j.value_in_unit(kilojoules_per_mole)
            sigma_ij = (sigma_i + sigma_j) / 2.0
            epsilon_ij = np.sqrt(epsilon_i * epsilon_j)
            r_ij = np.linalg.norm(positions[i] - positions[j])
            if r_ij < 1e-3:
                continue
            e_coul = coulomb_const * qi * qj / r_ij
            r_ratio = sigma_ij / r_ij
            e_lj = 4 * epsilon_ij * (r_ratio**12 - r_ratio**6)
            binding_energy += (e_coul + e_lj)
    return binding_energy

# ------------------------
#  Simulation function
# ------------------------
def run_sim_for_protein_ligand(protein_pdbfile, ligand_pdbfile, nsteps=NUM_STEPS, device_index=None):
    """
    Build a solvated protein-ligand system, add missing hydrogens, and run MD.
    Records:
      - Total potential energy (kJ/mol)
      - RMSD (nm)
      - Relative binding energy (delta binding energy, kJ/mol)
      - Positions (for each collection step)
    Writes out a CSV file and NPZ (training_data.npz) with these data.
    Returns energies and RMSDs.
    """
    # 1) Read protein and ligand.
    protein_pdb = PDBFile(protein_pdbfile)
    ligand_pdb  = PDBFile(ligand_pdbfile)
    num_protein_atoms = len(protein_pdb.positions)
    num_ligand_atoms = len(ligand_pdb.positions)
    
    # 2) Create ForceField and register GAFF for ligand.
    forcefield = ForceField(*FORCEFIELD_FILES)
    gaff = GAFFTemplateGenerator(forcefield="gaff-2.11")
    forcefield.registerTemplateGenerator(gaff.generator)
    
    # 3) Combine protein and ligand using Modeller; add hydrogens.
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_pdb.topology, ligand_pdb.positions)
    modeller.addHydrogens(forcefield)
    
    # 4) Solvate the system.
    modeller.addSolvent(forcefield,
                        model='tip3p',
                        padding=1.0 * nanometer,
                        ionicStrength=0.15 * molar)
    # Note: No repositioning; the ligand is kept free.
    
    # 5) Create the System using HBonds constraints.
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=NONBONDED_CUTOFF,
        constraints=HBonds  # Using constraints for speed and stability.
    )
    
    # 5.1 Get the NonbondedForce for binding energy calculation.
    nb_force = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nb_force = force
            break
    if nb_force is None:
        raise ValueError("No NonbondedForce found in the system!")
    
    # 6) Add MonteCarloBarostat if NPT is enabled.
    if USE_NPT:
        system.addForce(MonteCarloBarostat(1*atmosphere, TEMPERATURE, 25))
    
    # 7) Define the integrator.
    integrator = LangevinMiddleIntegrator(TEMPERATURE, LOW_FRICTION, TIMESTEP)
    
    # 8) Choose platform/properties.
    if device_index is not None:
        try:
            platform = Platform.getPlatformByName("CUDA")
            properties = {"DeviceIndex": str(device_index)}
        except Exception:
            platform = Platform.getPlatformByName("CPU")
            properties = {}
    else:
        platform = None
        properties = {}
    
    # 9) Create Simulation and set initial positions.
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    
    # 10) Minimize energy and set velocities.
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    
    # Data lists.
    energies = []
    rmsds = []
    binding_energies = []
    all_positions = []
    
    # Get reference positions and reference binding energy.
    state = simulation.context.getState(getPositions=True)
    ref_positions = positions_to_numpy(state.getPositions())
    ref_bind_energy = compute_binding_energy(ref_positions, nb_force, num_protein_atoms, num_ligand_atoms)
    
    # Create output folder.
    ligand_name = os.path.splitext(os.path.basename(ligand_pdbfile))[0]
    frames_dir = os.path.join(MD_OUTPUT_DIR, ligand_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Main MD loop using COLLECTION_INTERVAL.
    for step in range(0, nsteps, COLLECTION_INTERVAL):
        simulation.step(COLLECTION_INTERVAL)
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        current_positions = positions_to_numpy(state.getPositions())
        rmsd_val = compute_rmsd(ref_positions, current_positions)
        current_bind_energy = compute_binding_energy(current_positions, nb_force, num_protein_atoms, num_ligand_atoms)
        # Relative binding energy: difference from the reference frame.
        delta_bind_energy = current_bind_energy - ref_bind_energy
        
        energies.append(energy)
        rmsds.append(rmsd_val)
        binding_energies.append(delta_bind_energy)
        all_positions.append(current_positions)
        
        if (step + COLLECTION_INTERVAL) % PROGRESS_INTERVAL == 0:
            print(f"Step {step+COLLECTION_INTERVAL}/{nsteps} for ligand {ligand_name}: "
                  f"Energy={energy:.2f} kJ/mol, RMSD={rmsd_val:.3f} nm, ΔBindingEnergy={delta_bind_energy:.2f} kJ/mol")
    
    # Write per-ligand CSV.
    csv_filename = os.path.join(frames_dir, "simulation_info.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "Potential Energy (kJ/mol)", "RMSD (nm)", "Δ Binding Energy (kJ/mol)"])
        for i, (e, r, be) in enumerate(zip(energies, rmsds, binding_energies)):
            writer.writerow([i, e, r, be])
    
    # Save NPZ file with training data.
    training_data_npz = os.path.join(frames_dir, "training_data.npz")
    steps_array = np.arange(0, nsteps, COLLECTION_INTERVAL)
    all_positions_arr = np.array(all_positions)  # shape (n_points, n_atoms, 3)
    np.savez(
        training_data_npz,
        steps=steps_array,
        energies=np.array(energies),
        rmsds=np.array(rmsds),
        binding_energies=np.array(binding_energies),
        positions=all_positions_arr,
        protein_file=protein_pdbfile,
        ligand_file=ligand_pdbfile
    )
    
    return energies, rmsds

# ------------------------
#  Main function with parallel execution
# ------------------------
def main():
    os.makedirs(MD_OUTPUT_DIR, exist_ok=True)
    ligand_files = sorted(glob.glob(os.path.join(LIGANDS_DIR, "*.pdb")))
    if not ligand_files:
        print(f"No PDB ligands found in folder {LIGANDS_DIR}!")
        return
    
    num_gpus = get_num_gpus()
    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s).")
    else:
        print("No GPUs detected; simulations will run on CPU.")
    
    max_workers = num_gpus if num_gpus > 0 else min(len(ligand_files), os.cpu_count() or 1)
    all_energies = {}
    
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, ligpdb in enumerate(ligand_files):
            device_index = i % num_gpus if num_gpus > 0 else None
            future = executor.submit(run_sim_for_protein_ligand,
                                     PROTEIN_PDB, ligpdb, NUM_STEPS, device_index)
            ligand_name = os.path.splitext(os.path.basename(ligpdb))[0]
            futures[future] = ligand_name
        
        for future in as_completed(futures):
            ligand_name = futures[future]
            try:
                energies, rmsds = future.result()
                all_energies[ligand_name] = energies
                print(f"Finished simulation for ligand: {ligand_name}")
            except Exception as e:
                print(f"Simulation for ligand {ligand_name} generated an exception: {e}")
    
    # Write global CSV with energies across ligands.
    out_csv = os.path.join(MD_OUTPUT_DIR, "all_ligands_energy.csv")
    ligand_names = list(all_energies.keys())
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Step"] + ligand_names
        writer.writerow(header)
        for i in range(len(all_energies[ligand_names[0]])):
            row = [i]
            for ln in ligand_names:
                row.append(all_energies[ln][i])
            writer.writerow(row)
    print(f"\nFinished simulations! Global energies saved to {out_csv}.")
    
    if GENERATE_TRAJECTORY and SAVE_FRAMES:
        ligand_folders = [f for f in os.listdir(MD_OUTPUT_DIR) if os.path.isdir(os.path.join(MD_OUTPUT_DIR, f))]
        for folder in ligand_folders:
            ligand_folder = os.path.join(MD_OUTPUT_DIR, folder)
            merge_pdb_frames(ligand_folder)
    
    print(f"Simulation files (CSV, NPZ) are saved under '{MD_OUTPUT_DIR}'.")
    print("Each ligand folder contains 'training_data.npz' with steps, energies, RMSDs, Δ binding energies, and positions.")

if __name__ == "__main__":
    main()
