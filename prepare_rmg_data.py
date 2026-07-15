import argparse
import os
import yaml
import numpy as np
from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import AllChem
from molecule.molecule import Molecule


def get_desorbed_with_map(mol):
    molcopy = mol.copy(deep=True)
    init_map = {i: a for i, a in enumerate(molcopy.atoms)}

    for bd in molcopy.get_all_edges():
        if bd.atom1.is_surface_site():
            bd.atom2.radical_electrons += round(bd.order)
            molcopy.remove_bond(bd)
            molcopy.remove_atom(bd.atom1)
        elif bd.atom2.is_surface_site():
            bd.atom1.radical_electrons += round(bd.order)
            molcopy.remove_bond(bd)
            molcopy.remove_atom(bd.atom2)

    molcopy.sort_atoms()
    out_map = {i: molcopy.atoms.index(a) for i, a in init_map.items() if a in molcopy.atoms}
    return molcopy, out_map


def get_conformer(desorbed):
    try:
        rdmol, rdmap = desorbed.to_rdkit_mol(remove_h=False, return_mapping=True)
    except Exception as exc:
        syms = [a.symbol for a in desorbed.atoms]
        indmap = {i: i for i in range(len(desorbed.atoms))}
        if len(desorbed.atoms) == 1:
            atoms = Atoms(syms[0], positions=[(0, 0, 0)])
            return atoms, indmap
        if len(desorbed.atoms) == 2:
            atoms = Atoms(syms[0] + syms[1], positions=[(0, 0, 0), (1.3, 0, 0)])
            return atoms, indmap
        raise exc

    indmap = {i: rdmap[a] for i, a in enumerate(desorbed.atoms)}
    Chem.AllChem.EmbedMultipleConfs(rdmol, numConfs=1, randomSeed=1)
    conf = rdmol.GetConformer()
    pos = conf.GetPositions()
    syms = [a.GetSymbol() for a in rdmol.GetAtoms()]
    atoms = Atoms(symbols=syms, positions=pos)
    return atoms, indmap


def get_adsorbate(mol):
    desorbed, mol_to_desorbed_map = get_desorbed_with_map(mol)
    atoms, desorbed_to_atoms_map = get_conformer(desorbed)
    mol_to_atoms_map = {key: desorbed_to_atoms_map[val] for key, val in mol_to_desorbed_map.items()}
    return atoms, mol_to_atoms_map


def get_name(mol):
    try:
        return mol.to_smiles()
    except Exception:
        return mol.to_adjacency_list().replace("\n", " ")[:-1]


def get_surface_atom_map(mol, mol_to_atoms_map):
    ad_idx = [mol.atoms.index(a) for a in mol.get_adatoms()]
    ad_idx = [mol_to_atoms_map[i] for i in ad_idx]
    return ad_idx


def build_reaction_and_species(rxns_file):
    with open(rxns_file, "r") as f:
        rxns = yaml.safe_load(f)

    mols = []
    for rxn in rxns:
        rxn["reactant_mols"] = []
        rxn["product_mols"] = []

        react = Molecule().from_adjacency_list(rxn["reactant"])
        prod = Molecule().from_adjacency_list(rxn["product"])

        bond = []
        for idx, atom in enumerate(react.atoms):
            if atom.label == "*":
                bond.append(idx)
        rxn["broken_bond"] = bond

        react.clear_labeled_atoms()
        prod.clear_labeled_atoms()

        for mol in react.split():
            mol.multiplicity = mol.get_radical_count() + 1
            if not mol.is_surface_site():
                mols.append(mol)
                rxn["reactant_mols"].append(mol)

        for mol in prod.split():
            mol.multiplicity = mol.get_radical_count() + 1
            if not mol.is_surface_site():
                mols.append(mol)
                rxn["product_mols"].append(mol)

    unique_mols = []
    for mol in mols:
        for existing in unique_mols:
            if mol.is_isomorphic(existing):
                break
        else:
            unique_mols.append(mol)

    species = {}
    for idx, mol in enumerate(unique_mols):
        sp_id = f"sp_{idx:03d}"
        species[sp_id] = {
            "name": get_name(mol),
            "adjlist": mol.to_adjacency_list(),
            "mol_obj": mol,
        }

    for rxn in rxns:
        rxn["reactant_species"] = []
        rxn["product_species"] = []

        for i, rmol in enumerate(rxn["reactant_mols"]):
            for sp_id, rec in species.items():
                mol = rec["mol_obj"]
                if mol is rmol or mol.is_isomorphic(rmol, save_order=True):
                    rxn["reactant_mols"][i] = mol
                    rxn["reactant_species"].append(sp_id)
                    break
            else:
                raise ValueError("Unmatched reactant molecule")

        for i, pmol in enumerate(rxn["product_mols"]):
            for sp_id, rec in species.items():
                mol = rec["mol_obj"]
                if mol is pmol or mol.is_isomorphic(pmol, save_order=True):
                    rxn["product_mols"][i] = mol
                    rxn["product_species"].append(sp_id)
                    break
            else:
                raise ValueError("Unmatched product molecule")

    return rxns, species


def write_prepared_dataset(workdir, rxns_file, out_dir):
    rxns, species = build_reaction_and_species(rxns_file)

    os.makedirs(out_dir, exist_ok=True)
    template_dir = os.path.join(out_dir, "ads_templates")
    os.makedirs(template_dir, exist_ok=True)

    species_dump = {}
    for sp_id, record in species.items():
        mol = record["mol_obj"]
        ads, mol_to_atoms_map = get_adsorbate(mol)

        surf_sites = mol.get_surface_sites()
        if len(surf_sites) == 1:
            ad_idx = [mol.atoms.index(a) for a in mol.get_adatoms()]
            ad_idx = [mol_to_atoms_map[i] for i in ad_idx]
            ad_pos = ads[ad_idx[0]].position
            pos_average = np.mean(ads.positions, axis=0)
            if np.linalg.norm(ad_pos - pos_average) > 1e-6:
                ads.rotate(a=-ad_pos + pos_average, v=[0, 1, 0], center=[0, 0, 0])
            min_y = np.min(ads.positions[:, 1])
            if min_y < 0:
                ads.translate([0, -min_y, 0])
        ad_idx = get_surface_atom_map(mol, mol_to_atoms_map)

        template_path = os.path.join(template_dir, f"{sp_id}.xyz")
        write(template_path, ads)

        species_dump[sp_id] = {
            "name": record["name"],
            "adjlist": record["adjlist"],
            "template_xyz": os.path.relpath(template_path, out_dir),
            "ad_idx": ad_idx,
        }

    rxns_dump = []
    for rxn in rxns:
        rxns_dump.append(
            {
                "reactant": rxn["reactant"],
                "product": rxn["product"],
                "broken_bond": rxn["broken_bond"],
                "reactant_species": rxn["reactant_species"],
                "product_species": rxn["product_species"],
            }
        )

    payload = {
        "rxns": rxns_dump,
        "species": species_dump,
    }

    out_file = os.path.join(out_dir, "prepared_rmg_data.yaml")
    with open(out_file, "w") as f:
        yaml.safe_dump(payload, f, allow_unicode=True)

    print(f"Prepared data written to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RMG-derived species/adsorbate templates for DP-only runs")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--rxns", default="./test.yaml")
    parser.add_argument("--out", default="./prepared_data")
    args = parser.parse_args()

    write_prepared_dataset(
        workdir=args.workdir,
        rxns_file=args.rxns,
        out_dir=args.out,
    )
