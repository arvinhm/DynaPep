#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

import mdtraj as md
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import collections

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a basic protein-in-water OpenMM simulation with in-memory clustering and save cluster representatives containing only protein plus polar hydrogens."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input PDB file (e.g., GZMB.pdb)"
    )
    return parser.parse_args()

class MemoryStorageReporter:
    """
    A custom Reporter that saves simulation snapshots (positions and box vectors)
    in memory at a specified interval.
    """
    def __init__(self, reportInterval):
        self.reportInterval = reportInterval
        self.frames = []  # Will hold tuples of (positions, box_vectors)
    
    def describeNextReport(self, simulation):
        return (self.reportInterval, True, False, False, False, False, False)
    
    def report(self, simulation, state):
        positions = state.getPositions(asNumpy=True)
        box_vectors = state.getPeriodicBoxVectors()
        self.frames.append((positions, box_vectors))

def angle_degrees(v1, v2):
    """Compute the angle in degrees between two vectors."""
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    cosang = dot / (mag1 * mag2)
    cosang = np.clip(cosang, -1.0, 1.0)
    rad = np.arccos(cosang)
    return np.degrees(rad)

def box_lengths_and_angles_from_vectors(box_vectors):
    """
    Convert a 3x3 box vector matrix into (lengths, angles).
    Returns:
      lengths: array([a, b, c])
      angles:  array([alpha, beta, gamma]) in degrees.
    """
    a = np.linalg.norm(box_vectors[0])
    b = np.linalg.norm(box_vectors[1])
    c = np.linalg.norm(box_vectors[2])
    alpha = angle_degrees(box_vectors[1], box_vectors[2])
    beta  = angle_degrees(box_vectors[0], box_vectors[2])
    gamma = angle_degrees(box_vectors[0], box_vectors[1])
    return np.array([a, b, c]), np.array([alpha, beta, gamma])

def filter_protein_and_polar_hydrogens(traj):
    """
    Given a single-frame MDTraj trajectory containing the full system,
    returns a new trajectory with only protein atoms.
    For hydrogen atoms, only include them if they are polar (i.e. bonded to N, O, or S).
    This version uses `atom_slice()` so that the topology is correctly updated.
    """
    # Build a dictionary mapping each atom index to its bonded atom indices.
    bond_dict = {}
    for bond in traj.topology.bonds:
        a1 = bond.atom1.index
        a2 = bond.atom2.index
        bond_dict.setdefault(a1, []).append(a2)
        bond_dict.setdefault(a2, []).append(a1)
    
    # Define residue names to exclude (solvent and common ions).
    exclude_resnames = ["HOH", "WAT", "NA", "CL", "K", "MG", "CA"]
    
    # Select only atoms belonging to protein residues.
    protein_indices = traj.topology.select("protein")
    selected_indices = []
    for idx in protein_indices:
        atom = traj.topology.atom(idx)
        if atom.residue.name in exclude_resnames:
            continue
        # Keep non-hydrogen atoms.
        if atom.element is None or atom.element.symbol != "H":
            selected_indices.append(idx)
        else:
            # For hydrogen, keep it only if bonded to N, O, or S.
            partners = bond_dict.get(idx, [])
            if any(traj.topology.atom(p).element is not None and 
                   traj.topology.atom(p).element.symbol in ["N", "O", "S"]
                   for p in partners):
                selected_indices.append(idx)
    selected_indices = np.sort(selected_indices)
    if len(selected_indices) == 0:
        raise ValueError("No protein atoms (plus polar hydrogens) were selected.")
    # Use atom_slice() to update the topology accordingly.
    filtered_traj = traj.atom_slice(selected_indices)
    return filtered_traj

def main():
    args = parse_args()
    pdb_file = args.input

    # --------------------------- USER SETTINGS ---------------------------
    OUTPUT_DIR = "output_md"
    CLUSTER_DIR = os.path.join(OUTPUT_DIR, "clusters")

    # Simulation parameters
    PADDING_DISTANCE = 2.0 * nanometer
    NONBONDED_CUTOFF = 1.0 * nanometer
    TEMPERATURE = 300 * kelvin
    PRESSURE = 1 * bar
    TIMESTEP = 0.001 * picoseconds  # 1 fs (reduce further if needed)
    FRICTION_COEFF = 1 / picosecond
    NVT_STEPS = 5000
    NPT_STEPS = 5000
    REPORT_INTERVAL = 10  # Frequency for in-memory frame storage

    # Clustering parameters
    N_CLUSTERS = 100

    # Ensure output directories exist.
    for d in [OUTPUT_DIR, CLUSTER_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 1) Load the PDB file.
    if not os.path.isfile(pdb_file):
        print(f"[ERROR] Cannot find PDB file: {pdb_file}")
        sys.exit(1)
    print(f"Loading PDB file: {pdb_file}")
    pdb = PDBFile(pdb_file)

    # 2) Define the force field.
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # 3) Clean up: remove water & add hydrogens.
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.addHydrogens(forcefield)

    # 4) Re-center the protein by its geometric center.
    coords = np.array([[atom.x, atom.y, atom.z] for atom in modeller.positions.value_in_unit(nanometer)])
    center = coords.mean(axis=0)
    shift_vec = -center
    shifted_coords = coords + shift_vec
    new_positions = [Vec3(*xyz) for xyz in shifted_coords]
    modeller.positions = Quantity(new_positions, nanometer)
    print(f"Protein re-centered by shift = {shift_vec} nm")

    # 5) Solvate the system.
    modeller.addSolvent(forcefield, model="tip3p", padding=PADDING_DISTANCE)
    print("Solvation complete.")

    # 6) Create the system.
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=NONBONDED_CUTOFF,
        constraints=HBonds
    )

    # 7) Set up the integrator and simulation.
    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION_COEFF, TIMESTEP)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # 8) Minimize energy with a higher iteration limit.
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    
    # Check energy after minimization.
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    if np.isnan(energy):
        print("Error: Energy is NaN after minimization. Check your system (e.g., overlapping atoms).")
        sys.exit(1)
    else:
        print(f"Energy after minimization: {energy} kJ/mol")

    # 9) Set up reporters: log to stdout and CSV.
    log_path = os.path.join(OUTPUT_DIR, "log.csv")
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            REPORT_INTERVAL,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True
        )
    )
    simulation.reporters.append(
        StateDataReporter(
            log_path,
            REPORT_INTERVAL,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            separator=","
        )
    )
    # Add a custom reporter to store frames in memory.
    mem_reporter = MemoryStorageReporter(REPORT_INTERVAL)
    simulation.reporters.append(mem_reporter)

    # 10) Run NVT simulation.
    print(f"Running NVT for {NVT_STEPS} steps...")
    simulation.step(NVT_STEPS)

    # 11) Run NPT simulation.
    barostat = MonteCarloBarostat(PRESSURE, TEMPERATURE)
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    print(f"Running NPT for {NPT_STEPS} steps...")
    simulation.step(NPT_STEPS)
    print("Simulation complete.")

    # 12) Save the final structure.
    final_positions = simulation.context.getState(getPositions=True).getPositions()
    final_pdb = os.path.join(OUTPUT_DIR, "final_structure.pdb")
    with open(final_pdb, "w") as f:
        PDBFile.writeFile(modeller.topology, final_positions, f)
    print(f"Final structure saved to: {final_pdb}")

    # 13) Plot energy, temperature, and volume vs. simulation step.
    try:
        data = np.loadtxt(log_path, delimiter=",", skiprows=1)
        steps = data[:, 0]
        pot_energy = data[:, 1]
        temp = data[:, 2]
        vol = data[:, 3]

        plt.figure()
        plt.plot(steps, pot_energy, label="Potential Energy (kJ/mol)")
        plt.xlabel("Step")
        plt.ylabel("Energy (kJ/mol)")
        plt.title("Potential Energy vs Step")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "potential_energy.png"))
        plt.close()

        plt.figure()
        plt.plot(steps, temp, label="Temperature (K)")
        plt.xlabel("Step")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature vs Step")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "temperature.png"))
        plt.close()

        plt.figure()
        plt.plot(steps, vol, label="Volume (nm^3)")
        plt.xlabel("Step")
        plt.ylabel("Volume (nm^3)")
        plt.title("Box Volume vs Step")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "volume.png"))
        plt.close()

        print("Plots saved in", OUTPUT_DIR)
    except Exception as e:
        print(f"[WARNING] Could not plot data from log: {e}")

    # 14) In-memory clustering using stored frames.
    try:
        print("Clustering the in-memory frames...")

        # Build an MDTraj Topology from the OpenMM modeller topology.
        md_top = md.Topology.from_openmm(modeller.topology)

        # Gather positions and box vectors from our memory reporter.
        all_xyz = []  # shape: (n_frames, n_atoms, 3)
        all_box = []  # shape: (n_frames, 3, 3)
        for (pos, box) in mem_reporter.frames:
            all_xyz.append(pos.value_in_unit(nanometer))
            box_matrix = np.array([
                [box[0][0].value_in_unit(nanometer), box[0][1].value_in_unit(nanometer), box[0][2].value_in_unit(nanometer)],
                [box[1][0].value_in_unit(nanometer), box[1][1].value_in_unit(nanometer), box[1][2].value_in_unit(nanometer)],
                [box[2][0].value_in_unit(nanometer), box[2][1].value_in_unit(nanometer), box[2][2].value_in_unit(nanometer)]
            ])
            all_box.append(box_matrix)

        all_xyz = np.array(all_xyz)
        all_box = np.array(all_box)
        if all_xyz.shape[0] < 2:
            raise ValueError("Not enough frames collected for clustering.")

        # For older MDTraj versions, convert box vectors to unit cell lengths and angles.
        n_frames = all_box.shape[0]
        unitcell_lengths = np.zeros((n_frames, 3), dtype=np.float32)
        unitcell_angles = np.zeros((n_frames, 3), dtype=np.float32)
        for i in range(n_frames):
            lengths, angles = box_lengths_and_angles_from_vectors(all_box[i])
            unitcell_lengths[i] = lengths
            unitcell_angles[i] = angles

        # Create the MDTraj Trajectory from in-memory data.
        traj_in_memory = md.Trajectory(
            xyz=all_xyz,
            topology=md_top,
            unitcell_lengths=unitcell_lengths,
            unitcell_angles=unitcell_angles
        )

        # Re-image molecules so they are inside the primary box.
        traj_in_memory.image_molecules()

        # Align all frames on alpha carbons.
        alpha_indices = traj_in_memory.topology.select("name CA")
        if len(alpha_indices) == 0:
            raise ValueError("No alpha-carbons found; cannot cluster on CA atoms.")
        traj_in_memory.superpose(traj_in_memory, 0, atom_indices=alpha_indices)

        # Flatten the CA coordinates for clustering.
        xyz_CA = traj_in_memory.xyz[:, alpha_indices, :]
        n_frames, n_ca, _ = xyz_CA.shape
        X = xyz_CA.reshape(n_frames, n_ca * 3)

        print(f"Clustering {n_frames} frames into {N_CLUSTERS} clusters...")
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0)
        kmeans.fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Use PCA to visualize clustering.
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure()
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=10)
        plt.colorbar(scatter, label="Cluster ID")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("K-means Clustering on CA (PCA projection)")
        plt.savefig(os.path.join(OUTPUT_DIR, "clustering_pca.png"))
        plt.close()

        counts = collections.Counter(labels)
        # Save representative frame for each cluster—filter so that only protein (plus polar hydrogens) remains.
        for cid in range(N_CLUSTERS):
            idxs = np.where(labels == cid)[0]
            cluster_points = X[idxs]
            dists = np.linalg.norm(cluster_points - centers[cid], axis=1)
            rep_idx = idxs[np.argmin(dists)]
            rep_traj = traj_in_memory.slice(rep_idx)
            # Filter: retain only protein atoms and polar hydrogens.
            filtered_rep = filter_protein_and_polar_hydrogens(rep_traj)
            rep_filename = os.path.join(CLUSTER_DIR, f"cluster_{cid}_rep.pdb")
            filtered_rep.save_pdb(rep_filename)

        # Save a bar plot of cluster sizes.
        plt.figure()
        plt.bar(range(N_CLUSTERS), [counts[i] for i in range(N_CLUSTERS)])
        plt.xlabel("Cluster ID")
        plt.ylabel("Frame Count")
        plt.title("Cluster Sizes")
        plt.savefig(os.path.join(OUTPUT_DIR, "cluster_counts.png"))
        plt.close()

        print(f"Clustering analysis complete. Representative frames saved in {CLUSTER_DIR}")
    except Exception as e:
        print(f"[WARNING] Clustering step failed: {e}")

if __name__ == "__main__":
    main()
