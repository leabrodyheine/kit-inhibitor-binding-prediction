"""Data loading and cleaning helpers for KIT bioactivity data (Phase 1-2), plus
train/test splitting utilities used before modeling (Phase 4)."""

import random
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def bemis_murcko_scaffold(smiles):
    """Return the Bemis-Murcko scaffold (generic core, no side chains) of a SMILES
    string as a canonical SMILES string. Stereochemistry is stripped, so
    stereoisomers of the same scaffold are grouped together."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles_list, frac_train=0.8, seed=0):
    """Split row indices into train/test sets grouped by Bemis-Murcko scaffold,
    so no scaffold appears on both sides (Design Doc §5.4: random splits on
    cheminformatics data overestimate generalization because near-identical
    analogues end up in both train and test).

    Rows sharing a scaffold (e.g. WT/D816V pairs of the same compound, which
    share identical SMILES and therefore identical scaffolds) always land in
    the same split.

    Algorithm follows the standard MoleculeNet/DeepChem scaffold split:
    scaffold groups are shuffled (for a seeded but non-alphabetical tie-break
    order) then sorted largest-first, and assigned whole to train until the
    train fraction target is met; the remaining (smaller, rarer-scaffold)
    groups go to test. This deliberately makes test the harder, less
    chemically-similar-to-train split, rather than a softer random sample.

    Returns (train_idx, test_idx) as sorted numpy arrays of positional indices
    into `smiles_list`.
    """
    scaffold_to_indices = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        scaffold_to_indices[bemis_murcko_scaffold(smiles)].append(i)

    groups = list(scaffold_to_indices.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_train_target = round(frac_train * len(smiles_list))
    train_idx, test_idx = [], []
    for group in groups:
        if len(train_idx) < n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))
