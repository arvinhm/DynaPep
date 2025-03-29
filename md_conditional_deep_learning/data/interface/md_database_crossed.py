#!/usr/bin/env python
"""
md_cross_database_enhanced.py

Usage Example:
    python md_cross_database_enhanced.py /path/to/complex_folder /path/to/output_folder
         --steps 10000 --clusters 10 --metad-freq 5000

Enhancements:
1. (Optional) Metadynamics stage using a distance-based CV between chain A and chain B centers.
2. All the usual cross docking, clustering, and data aggregation.

You may install the needed packages (including an up-to-date OpenMM) so that:
    from openmm.app.metadynamics import Metadynamics, BiasVariable
works properly.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mdtraj as md

from openmm import Platform, LangevinIntegrator, unit, Vec3
from openmm import CustomCentroidBondForce
from openmm.app import (PDBFile, Modeller, Simulation, DCDReporter, 
                        StateDataReporter, PME)
from openmm.app.metadynamics import Metadynamics, BiasVariable

from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer

import concurrent.futures
from tqdm import tqdm
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

# -------------------------
# Helper Functions
# -------------------------
def get_platform():
    """Return the fastest available OpenMM platform (CUDA/OpenCL if possible)."""
    os_platform = os.getenv('PLATFORM')
    if os_platform:
        platform = Platform.getPlatformByName(os_platform)
    else:
        speed = 0
        platform = None
        for i in range(Platform.getNumPlatforms()):
            p = Platform.getPlatform(i)
            if p.getSpeed() > speed:
                platform = p
                speed = p.getSpeed()
    if platform is None:
        platform = Platform.getPlatformByName("Reference")
    else:
        if platform.getName() in ['CUDA', 'OpenCL']:
            platform.setPropertyDefaultValue('Precision', 'mixed')
    return platform

def bounding_box_coords(positions):
    """
    Return the bounding box array [minx, maxx, miny, maxy, minz, maxz]
    for an Nx3 array of positions.
    """
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]])

def fix_complex_modeller(input_pdbfile, keep_chains=('A','B')):
    """
    Create a Modeller from the given PDB file, removing all residues 
    not in keep_chains.
    """
    pdb_in = PDBFile(input_pdbfile)
    modeller = Modeller(pdb_in.topology, pdb_in.positions)
    to_delete = [res for res in modeller.topology.residues() if res.chain.id not in keep_chains]
    modeller.delete(to_delete)
    return modeller

# -------------------------
# Clustering
# -------------------------
def cluster_and_save_complex(minimised_pdb, dcd_file, output_folder, num_clusters):
    """
    Load the full trajectory (pdb + dcd), cluster chain B, and compute:
      - docking_score (=-#contacts)
      - binding_energy (=-0.5 * #contacts) 
      - RMSD (vs. frame 0)
      - mean COM distance
    Save "clusters.csv" and representative cluster structures in "clusters_full/".
    """
    try:
        import mdtraj as md
        traj = md.load(dcd_file, top=minimised_pdb)
        pep_inds = traj.topology.select("chainid 1")  # chain B
        if pep_inds.size == 0:
            return None
        pep_traj = traj.atom_slice(pep_inds)
        n_frames = pep_traj.n_frames
        if n_frames < 1:
            return None
        
        coords_2d = pep_traj.xyz.reshape(n_frames, -1)
        if n_frames < num_clusters:
            num_clusters = n_frames
        
        with threadpool_limits(limits=1, user_api='blas'):
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(coords_2d)
        
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        rep_indices = []
        for c in range(num_clusters):
            idxs = np.where(labels == c)[0]
            center = centers[c]
            dists = np.linalg.norm(coords_2d[idxs] - center, axis=1)
            rep_idx = idxs[np.argmin(dists)]
            rep_indices.append(rep_idx)
        
        cluster_folder = os.path.join(output_folder, "clusters_full")
        os.makedirs(cluster_folder, exist_ok=True)
        
        def compute_contacts(frame, prot_inds, pep_inds, cutoff=0.4):
            prot_coords = frame[prot_inds]
            pep_coords = frame[pep_inds]
            diff = prot_coords[:, None, :] - pep_coords[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            return np.sum(dists < cutoff)
        
        ref_pep_coords = traj.xyz[0][pep_inds]
        rec_inds = traj.topology.select("chainid 0")
        
        rows = []
        for c_id, frame_idx in enumerate(rep_indices, start=1):
            rep_snap = traj[frame_idx]
            out_pdb = os.path.join(cluster_folder, f"cluster_{c_id}_rep.pdb")
            rep_snap.save_pdb(out_pdb)
            
            contacts = compute_contacts(traj.xyz[frame_idx], rec_inds, pep_inds, cutoff=0.4)
            docking_score = -contacts
            binding_energy = -0.5 * contacts
            
            current_pep_coords = traj.xyz[frame_idx][pep_inds]
            rmsd_val = np.sqrt(np.mean((current_pep_coords - ref_pep_coords)**2))
            
            rec_coords = traj.xyz[frame_idx][rec_inds]
            pep_coords = traj.xyz[frame_idx][pep_inds]
            rec_com = rec_coords.mean(axis=0)
            pep_com = pep_coords.mean(axis=0)
            mean_dist = np.linalg.norm(rec_com - pep_com)
            
            rows.append({
                "cluster_id": c_id,
                "frame_idx": frame_idx,
                "filename": out_pdb,
                "docking_score": docking_score,
                "binding_energy_kcal_per_mol": binding_energy,
                "rmsd_nm": rmsd_val,
                "mean_distance_nm": mean_dist
            })
        
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_folder, "clusters.csv"), index=False)
        return True
    except Exception as e:
        print("Error in clustering:", e)
        return None


# -------------------------
# (Optional) Pulling Force
# -------------------------
def apply_pull_force(system, topology, chainA_id=0, chainB_id=1, k_harm=5.0):
    """
    Add a CustomCentroidBondForce to pull chain B away from chain A 
    after production, for measuring detachment force.
    """
    from openmm import CustomCentroidBondForce
    force = CustomCentroidBondForce(2, "0.5* k * (distance(g1,g2) - r0)^2")
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")
    
    chainA_inds = []
    chainB_inds = []
    for chain in topology.chains():
        if chain.index == chainA_id:
            chainA_inds.extend([atom.index for atom in chain.atoms()])
        elif chain.index == chainB_id:
            chainB_inds.extend([atom.index for atom in chain.atoms()])
    
    force.addGroup(chainA_inds)
    force.addGroup(chainB_inds)
    force.addBond([0, 1], [k_harm, 0.0])
    system.addForce(force)
    return force

def update_pull_rest_length(simulation, pull_force, r0_new):
    pull_force.setBondParameters(0, [0,1],
         [pull_force.getBondParameters(0)[2][0], r0_new])
    pull_force.updateParametersInContext(simulation.context)


# -------------------------
# (Optional) Metadynamics
# -------------------------
def setup_metadynamics(system, topology, temperature, deposit_period=1000,
                       bias_factor=5.0, height=1.0, distance_min=0.1, distance_max=3.0):
    """
    Create a Metadynamics object for a single distance-based CV 
    between the alpha-C of chain A and alpha-C of chain B (example).
    
    - deposit_period: How often (in integration steps) to deposit bias
    - bias_factor: The well-tempered bias factor
    - height: Gaussian height in kJ/mol
    - distance_min, distance_max: The range for the CV
    """
    from openmm.app.metadynamics import Metadynamics, BiasVariable
    
    # For demonstration, pick the first alpha carbon in chain A and chain B
    # or just the first heavy atom if no alpha carbons found.
    chainA_ca = None
    chainB_ca = None
    
    for chain in topology.chains():
        if chain.index == 0:  # chain A
            for atom in chain.atoms():
                # find alpha carbon if possible
                if ('CA' in atom.name) or ('C' in atom.name and atom.element.symbol=='C'):
                    chainA_ca = atom.index
                    break
        elif chain.index == 1: # chain B
            for atom in chain.atoms():
                if ('CA' in atom.name) or ('C' in atom.name and atom.element.symbol=='C'):
                    chainB_ca = atom.index
                    break
    
    if chainA_ca is None or chainB_ca is None:
        print("[WARNING] Could not find alpha carbons for Metadynamics. Skipping.")
        return None
    
    # Distance-based CV
    distance_var = BiasVariable(
        lambda state: state.getPositions(asNumpy=True), 
        minValue=distance_min, 
        maxValue=distance_max,
        biasWidth=0.1,
        periodic=False,
        gridWidth=300,
        name="distAB"
    )
    
    # The above "lambda" won't handle the distance itself. We can do an official approach:
    # Actually, the built-in "Distance" CV from openmm.app.metadynamics uses atom indices directly:
    from openmm.app.metadynamics import Distance
    distance_cv = Distance(atom1=chainA_ca, atom2=chainB_ca,
                           minValue=distance_min, maxValue=distance_max, biasWidth=0.1, periodic=False)
    
    # well-tempered metad with specified bias factor
    # "temperature" is in K, so we convert it for the object as well
    metad = Metadynamics(system=system,
                         variables=[distance_cv],
                         temperature=temperature*unit.kelvin,
                         biasFactor=bias_factor,
                         height=height*unit.kilojoule_per_mole,
                         frequency=deposit_period,
                         saveFrequency=0,      # no checkpoint for bias
                         biasDir=None,
                         reportInterval=deposit_period)
    return metad


# -------------------------
# The main MD pipeline
# -------------------------
def run_full_md_pipeline(modeller, outdir, steps, step_size, friction, temperature,
                         equil_steps, clusters, pull_k, pull_steps,
                         metad_freq, metad_factor, metad_height):
    """
    1) Write out initial PDB
    2) Fix with PDBFixer
    3) Create System
    4) Minimize + Equilibrate + Production MD
    5) (Optional) run Metadynamics if metad_freq>0
    6) (Optional) pulling simulation if pull_k>0
    7) cluster chain B
    """
    os.makedirs(outdir, exist_ok=True)
    
    # 1) Write the input complex PDB
    complex_out = os.path.join(outdir, "complex.pdb")
    with open(complex_out, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    
    # 2) PDBFixer
    fixer = PDBFixer(filename=complex_out)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    fixed_topology = fixer.topology
    fixed_positions = fixer.positions
    fixed_pdb = os.path.join(outdir, "complex_fixed.pdb")
    with open(fixed_pdb, 'w') as f:
        PDBFile.writeFile(fixed_topology, fixed_positions, f)
    
    # 3) System
    ff_files = ['amber/ff14SB.xml']
    system_generator = SystemGenerator(forcefields=ff_files)
    try:
        system = system_generator.create_system(fixed_topology)
    except Exception as e:
        if "No template found for residue" in str(e):
            # remove heterogens and retry
            fixer.removeHeterogens(True)
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            fixed_topology = fixer.topology
            fixed_positions = fixer.positions
            with open(fixed_pdb, 'w') as f:
                PDBFile.writeFile(fixed_topology, fixed_positions, f)
            system = system_generator.create_system(fixed_topology)
        else:
            raise e
    
    integrator = LangevinIntegrator(temperature*unit.kelvin,
                                    friction/unit.picosecond,
                                    step_size*unit.picoseconds)
    platform = get_platform()
    simulation = Simulation(fixed_topology, system, integrator, platform=platform)
    simulation.context.setPositions(fixed_positions)
    
    # Minimization
    simulation.minimizeEnergy()
    minpdb = os.path.join(outdir, "minimised.pdb")
    with open(minpdb, 'w') as f:
        PDBFile.writeFile(
            fixed_topology,
            simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
            f, keepIds=True
        )
    
    # bounding box for chain A
    import mdtraj as md
    tmp_top = md.Topology.from_openmm(fixed_topology)
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    rec_inds = tmp_top.select("chainid 0")
    if len(rec_inds)>0:
        rec_coords = pos[rec_inds]
        box_arr = bounding_box_coords(rec_coords)
        box_csv = os.path.join(outdir, "md_box.csv")
        with open(box_csv, 'w') as bf:
            bf.write("xmin,xmax,ymin,ymax,zmin,zmax\n")
            bf.write(",".join(str(x) for x in box_arr)+"\n")
    
    # Equilibration
    simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)
    simulation.step(equil_steps)
    
    # -----------------------------------------------------------
    # Production MD
    dcd_file = os.path.join(outdir, "traj.dcd")
    energy_file = open(os.path.join(outdir, "energy.csv"), 'w')
    energy_file.write("step,potential_energy_kjmol,temperature_k\n")
    simulation.reporters.append(DCDReporter(dcd_file, 100, enforcePeriodicBox=False))
    simulation.reporters.append(StateDataReporter(energy_file, 1000, 
                                                  step=True, potentialEnergy=True, temperature=True))
    
    # (Optional) If metad_freq>0, set up metad
    metad = None
    if metad_freq>0:
        metad = setup_metadynamics(system, fixed_topology, temperature,
                                   deposit_period=metad_freq, 
                                   bias_factor=metad_factor,
                                   height=metad_height,
                                   distance_min=0.1, distance_max=3.0)
    
    # If metad is active, we can't just do simulation.step(steps)
    # We do a loop so that metad.update(simulation) is called each iteration.
    steps_done = 0
    if metad is None:
        # normal production
        simulation.step(steps)
    else:
        # run with metad updates
        for s in range(steps):
            simulation.step(1)
            metad.update(simulation)
            steps_done += 1
    energy_file.close()
    
    # -----------------------------------------------------------
    # 5) (Optional) pulling
    if pull_k>0 and pull_steps>0:
        pull_force = apply_pull_force(system, fixed_topology, chainA_id=0, chainB_id=1, k_harm=pull_k)
        simulation.context.reinitialize()
        simulation.context.setPositions(simulation.context.getState(getPositions=True).getPositions())
        simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)
        
        # We do a linear ramp from current distance to +2 nm
        state = simulation.context.getState(getPositions=True)
        coords = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        Ainds = tmp_top.select("chainid 0")
        Binds = tmp_top.select("chainid 1")
        if len(Ainds)>0 and len(Binds)>0:
            Acom = coords[Ainds].mean(axis=0)
            Bcom = coords[Binds].mean(axis=0)
            init_dist = np.linalg.norm(Acom - Bcom)
            final_dist = init_dist + 2.0
            step_increase = (final_dist - init_dist)/pull_steps
            
            with open(os.path.join(outdir, "energy.csv"), 'a') as ef:
                ef.write("# Pulling stage begins\n")
                simulation.reporters = []
                simulation.reporters.append(StateDataReporter(ef, 1000, step=True,
                                                              potentialEnergy=True, temperature=True))
                for pull_step in range(1, pull_steps+1):
                    r0_new = init_dist + step_increase*pull_step
                    update_pull_rest_length(simulation, pull_force, r0_new)
                    simulation.step(1000)
    
    # -----------------------------------------------------------
    # 6) Cluster chain B
    cluster_and_save_complex(minpdb, dcd_file, outdir, clusters)
    
    return fixed_pdb, minpdb, dcd_file


# -------------------------
# process_complex
# -------------------------
def process_complex(args_tuple):
    """
    Clean/fix each original complex, run MD pipeline, store chain B for cross docking.
    """
    complex_file, out_subdir, args, all_peptides = args_tuple
    base_name = os.path.splitext(os.path.basename(complex_file))[0]
    try:
        mod = fix_complex_modeller(complex_file, keep_chains=('A','B'))
        if mod is None:
            return (base_name, "Error")
        
        run_full_md_pipeline(
            modeller=mod,
            outdir=out_subdir,
            steps=args.steps,
            step_size=args.step_size,
            friction=args.friction_coeff,
            temperature=args.temperature,
            equil_steps=args.equil_steps,
            clusters=args.clusters,
            pull_k=args.pull_k,
            pull_steps=args.pull_steps,
            metad_freq=args.metad_freq,
            metad_factor=args.metad_factor,
            metad_height=args.metad_height
        )
        
        # Extract chain B
        fixed_pdb = os.path.join(out_subdir, "complex_fixed.pdb")
        pep_mod = fix_complex_modeller(fixed_pdb, keep_chains=('B',))
        if pep_mod:
            all_peptides[base_name] = pep_mod
        else:
            return (base_name, "Error")
        
        return (base_name, "Success")
    except Exception as e:
        return (base_name, "Error")

# -------------------------
# Cross Docking
# -------------------------
def cross_processing(all_peptides, all_complexes, out_root, args):
    """
    For each successfully processed receptor complex, 
    cross with every foreign peptide in all_peptides.
    """
    from openmm.app import Modeller
    from openmm import Vec3
    
    for rec_name in all_complexes:
        rec_folder = os.path.join(out_root, rec_name)
        fix_pdb_path = os.path.join(rec_folder, "complex_fixed.pdb")
        if not os.path.exists(fix_pdb_path):
            continue
        
        rec_full = fix_complex_modeller(fix_pdb_path, keep_chains=('A','B'))
        if not rec_full:
            continue
        
        # get chain B center
        b_inds = []
        for i, atom in enumerate(rec_full.topology.atoms()):
            if atom.residue.chain.id == 'B':
                b_inds.append(i)
        if not b_inds:
            continue
        import numpy as np
        b_coords = np.array([[rec_full.positions[i].x,
                              rec_full.positions[i].y,
                              rec_full.positions[i].z] for i in b_inds])
        rec_b_center = b_coords.mean(axis=0)
        
        # remove chain B
        rec_A = fix_complex_modeller(fix_pdb_path, keep_chains=('A',))
        
        for pep_name, pep_mod in all_peptides.items():
            if pep_name == rec_name:
                # skip self cross
                continue
            cross_name = f"{rec_name}_{pep_name}"
            cross_out = os.path.join(out_root, cross_name)
            if not os.path.exists(cross_out):
                os.makedirs(cross_out, exist_ok=True)
            
            pep_inds = []
            for i, atom in enumerate(pep_mod.topology.atoms()):
                pep_inds.append(i)
            pep_coords = np.array([[pep_mod.positions[i].x,
                                    pep_mod.positions[i].y,
                                    pep_mod.positions[i].z] for i in pep_inds])
            pep_center = pep_coords.mean(axis=0)
            shift_vec = rec_b_center - pep_center
            shifted_positions = pep_coords + shift_vec
            
            from openmm import Vec3
            new_positions = []
            for row in shifted_positions:
                new_positions.append(Vec3(*row)*unit.nanometer)
            
            cross_mod = Modeller(rec_A.topology, rec_A.positions)
            cross_mod.add(pep_mod.topology, new_positions)
            
            try:
                run_full_md_pipeline(
                    modeller=cross_mod,
                    outdir=cross_out,
                    steps=args.steps,
                    step_size=args.step_size,
                    friction=args.friction_coeff,
                    temperature=args.temperature,
                    equil_steps=args.equil_steps,
                    clusters=args.clusters,
                    pull_k=args.pull_k,
                    pull_steps=args.pull_steps,
                    metad_freq=args.metad_freq,
                    metad_factor=args.metad_factor,
                    metad_height=args.metad_height
                )
            except Exception:
                pass

# -------------------------
# Aggregation
# -------------------------
def aggregate_grid_boxes(out_root, complexes):
    rows = []
    for comp in complexes:
        csv_path = os.path.join(out_root, comp, "md_box.csv")
        if os.path.exists(csv_path):
            df_box = pd.read_csv(csv_path)
            if not df_box.empty:
                row = df_box.iloc[0].to_dict()
                row["complex"] = comp
                rows.append(row)
    if rows:
        agg_df = pd.DataFrame(rows)
        agg_csv = os.path.join(out_root, "all_grid_boxes.csv")
        agg_df.to_csv(agg_csv, index=False)
        print(f"[INFO] All bounding boxes saved to {agg_csv}")

def aggregate_energy_data(out_root):
    aggregated = []
    for folder in sorted(os.listdir(out_root)):
        folder_path = os.path.join(out_root, folder)
        if not os.path.isdir(folder_path):
            continue
        
        energy_file = os.path.join(folder_path, "energy.csv")
        if not os.path.exists(energy_file):
            continue
        
        parts = folder.split("_")
        if len(parts)>1:
            receptor = parts[0]
            peptide = "_".join(parts[1:])
        else:
            receptor = folder
            peptide = "-"
        
        try:
            lines = []
            with open(energy_file, 'r') as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    lines.append(line)
            if not lines:
                continue
            from io import StringIO
            csv_str = "".join(lines)
            df = pd.read_csv(StringIO(csv_str))
            df["receptor"] = receptor
            df["peptide"]  = peptide
            df["simulation_folder"] = folder
            aggregated.append(df)
        except Exception:
            pass
    
    if aggregated:
        bigdf = pd.concat(aggregated, ignore_index=True)
        out_csv = os.path.join(out_root, "all_cross_energy.csv")
        bigdf.to_csv(out_csv, index=False)
        print(f"[INFO] Aggregated energy saved to {out_csv}")


def aggregate_cluster_data(out_root):
    import pandas as pd
    clusters_list = []
    for folder in os.listdir(out_root):
        folder_path = os.path.join(out_root, folder)
        if os.path.isdir(folder_path):
            clusters_file = os.path.join(folder_path, "clusters.csv")
            if os.path.exists(clusters_file):
                try:
                    df = pd.read_csv(clusters_file)
                    df["simulation_folder"] = folder
                    clusters_list.append(df)
                except Exception as e:
                    print(f"Error reading {clusters_file}: {e}")
    if clusters_list:
        merged_df = pd.concat(clusters_list, ignore_index=True)
        merged_csv = os.path.join(out_root, "all_clusters.csv")
        merged_df.to_csv(merged_csv, index=False)
        print(f"[INFO] Aggregated clusters data saved to {merged_csv}")


# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MD Cross Docking with optional pulling & metadynamics for enhanced sampling"
    )
    parser.add_argument("complex_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--step-size", type=float, default=0.001)
    parser.add_argument("--friction-coeff", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--equil-steps", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--pull-k", type=float, default=0, help="Force constant for pulling")
    parser.add_argument("--pull-steps", type=int, default=0)
    
    # Enhanced sampling arguments
    parser.add_argument("--metad-freq", type=int, default=0,
                        help="If >0, deposit bias every these many steps for metadynamics.")
    parser.add_argument("--metad-factor", type=float, default=10.0,
                        help="Well-tempered bias factor for metadynamics")
    parser.add_argument("--metad-height", type=float, default=1.0,
                        help="Gaussian height (kJ/mol) for metadynamics")
    
    args = parser.parse_args()
    
    all_pdbs = sorted([f for f in os.listdir(args.complex_folder) if f.lower().endswith('.pdb')])
    if args.subset>0:
        all_pdbs = all_pdbs[:args.subset]
    if not all_pdbs:
        print("No pdb found in the input folder.")
        sys.exit(1)
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    from multiprocessing import Manager
    manager = Manager()
    all_peptides = manager.dict()  # store chain B modellers
    
    job_args = []
    for pdbf in all_pdbs:
        base = os.path.splitext(pdbf)[0]
        inf = os.path.join(args.complex_folder, pdbf)
        outd = os.path.join(args.output_folder, base)
        os.makedirs(outd, exist_ok=True)
        job_args.append((inf, outd, args, all_peptides))
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_complex, job): job for job in job_args}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(job_args),
                        desc="Processing original complexes"):
            results.append(fut.result())
    
    processed_complexes = [r[0] for r in results if r[1]=="Success"]
    
    # Cross docking
    cross_processing(all_peptides, processed_complexes, args.output_folder, args)
    
    # Aggregation
    aggregate_grid_boxes(args.output_folder, processed_complexes)
    aggregate_energy_data(args.output_folder)
    aggregate_cluster_data(args.output_folder)

if __name__ == "__main__":
    main()
