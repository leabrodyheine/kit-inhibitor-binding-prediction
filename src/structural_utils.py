"""Structural context helpers for the PDB-based sanity check (Phase 6).

Design Doc §5.7 / §4.2: used qualitatively only -- no docking. Functions here
either (a) do pure, offline-testable geometry/parsing logic on an
already-loaded Biopython structure, or (b) are thin network wrappers around
RCSB's file download and Chemical Component Dictionary REST API (download
PDB structures, fetch a ligand's canonical SMILES) -- the latter are
exercised by actually running notebooks/06_structural_context.ipynb rather
than by a offline unit test, the same pattern Phase 1 used for the ChEMBL API
pull.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # this module only ever saves figures, never shows them interactively

import matplotlib.pyplot as plt
import numpy as np
import requests

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_code}"

# Okabe-Ito colorblind-safe pair (validated: scripts/validate_palette.js
# "#D55E00,#009E73" --mode light -- all checks pass), matching the palette
# already validated and used for Phase 4's diagnostic plots.
_LIGAND_COLOR = "#D55E00"
_RESIDUE_COLOR = "#009E73"


def download_pdb(pdb_id, dest_dir):
    """Download a PDB structure file from RCSB, caching it in `dest_dir` so
    repeat runs don't re-download. Returns the local file path.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{pdb_id}.pdb"
    if not dest_path.exists():
        response = requests.get(RCSB_PDB_URL.format(pdb_id=pdb_id), timeout=30)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
    return dest_path


def fetch_ligand_smiles(ligand_code):
    """Fetch a ligand's canonical SMILES from RCSB's Chemical Component
    Dictionary, given its 2-5 character PDB ligand code (e.g. "647").
    Raises ValueError if no SMILES descriptor is present for that component.
    """
    response = requests.get(RCSB_CHEMCOMP_URL.format(ligand_code=ligand_code), timeout=30)
    response.raise_for_status()
    data = response.json()

    descriptors = data.get("pdbx_chem_comp_descriptor", [])
    for descriptor in descriptors:
        if descriptor.get("type") == "SMILES_CANONICAL":
            return descriptor["descriptor"]
    for descriptor in descriptors:
        if "SMILES" in descriptor.get("type", ""):
            return descriptor["descriptor"]
    raise ValueError(f"No SMILES descriptor found for ligand code {ligand_code!r}")


def get_hetero_ligand_codes(model, exclude=frozenset({"HOH"})):
    """Return the set of non-water heteroatom residue names (ligand/cofactor
    codes) across all chains of a Biopython `Model`. `exclude` defaults to
    just water; callers filter out cofactors like ADP/MG themselves, since
    "is this a real inhibitor" is a judgment call, not a parsing question.
    """
    codes = set()
    for chain in model:
        for residue in chain:
            hetero_flag = residue.id[0].strip()
            if hetero_flag and residue.resname not in exclude:
                codes.add(residue.resname)
    return codes


def get_residue_coord(model, chain_id, residue_number, atom_name="CA"):
    """Return (coord, resname) for a specific atom of a specific residue
    (e.g. residue 816's CA, to locate the D816V mutation site) in a
    Biopython `Model`. Raises KeyError if the chain/residue/atom is absent.
    """
    chain = model[chain_id]
    residue = chain[(" ", residue_number, " ")]
    atom = residue[atom_name]
    return np.array(atom.coord, dtype=float), residue.resname


def ligand_atom_coords(model, chain_id, ligand_resname):
    """Return an (n_atoms, 3) array of coordinates for a named hetero
    residue (e.g. the co-crystallized inhibitor) in a given chain. Raises
    KeyError if no matching hetero residue is found.
    """
    chain = model[chain_id]
    for residue in chain:
        if residue.resname == ligand_resname and residue.id[0].strip():
            return np.array([atom.coord for atom in residue], dtype=float)
    raise KeyError(f"Ligand {ligand_resname!r} not found in chain {chain_id!r}")


def min_distance_to_point(coords, point):
    """Minimum Euclidean distance from any row of `coords` (n_atoms, 3) to a
    single 3D `point` -- a simple, docking-free proxy for how close a
    ligand's binding site is to a specific residue (e.g. the D816 Cα)."""
    coords = np.asarray(coords, dtype=float)
    point = np.asarray(point, dtype=float)
    return float(np.min(np.linalg.norm(coords - point, axis=1)))


def get_backbone_trace(model, chain_id):
    """Return the ordered (n_residues, 3) array of Cα coordinates for a
    chain's protein residues (hetero residues excluded), for drawing a
    simple backbone trace. Residues missing a CA atom (rare, e.g. some
    disordered loops) are skipped."""
    chain = model[chain_id]
    coords = [
        residue["CA"].coord
        for residue in chain
        if residue.id[0].strip() == "" and "CA" in residue
    ]
    return np.array(coords, dtype=float)


def plot_binding_site_context(model, chain_id, ligand_resname, residue_number=816, title=None, save_path=None):
    """Static, qualitative 3D plot: the protein's Cα backbone trace, a
    co-crystallized ligand's atoms, and the mutation-site residue
    highlighted. A docking-free visual sanity check (Phase 6 / Design Doc
    §5.7) -- not a binding-pose prediction.

    Returns the matplotlib Figure; also saves it to `save_path` if given.
    """
    backbone = get_backbone_trace(model, chain_id)
    ligand_coords = ligand_atom_coords(model, chain_id, ligand_resname)
    residue_coord, resname = get_residue_coord(model, chain_id, residue_number)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        backbone[:, 0], backbone[:, 1], backbone[:, 2],
        color="0.7", linewidth=1, label="Protein backbone (Cα trace)",
    )
    ax.scatter(*ligand_coords.T, color=_LIGAND_COLOR, s=30, label=f"Ligand {ligand_resname}")
    ax.scatter(
        *residue_coord, color=_RESIDUE_COLOR, s=150, marker="*",
        label=f"Residue {residue_number} ({resname}) Cα",
    )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
