"""Data loading and cleaning helpers for KIT bioactivity data (Phase 1-2), plus
train/test splitting utilities used before modeling (Phase 4)."""

import random
from collections import defaultdict

import numpy as np
import pandas as pd
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


def get_variant_pairs(df, variant_col="kit_variant", id_col="molecule_chembl_id", value_col="p_value"):
    """Pivot the cleaned bioactivity table (Phase 2) to one row per compound
    that has *both* a WT and a D816V record, and compute the WT/D816V
    selectivity ratio (Design Doc §5.6).

    `value_col` is on the p-scale (-log10 molar), so potency is proportional
    to 10**value_col. `selectivity_ratio` = WT potency / D816V potency =
    10**(p_value_wt - p_value_d816v): > 1 means more potent against WT (loses
    efficacy against the mutant, e.g. imatinib); < 1 means more potent
    against the mutant; ~1 means comparable potency against both (e.g.
    dasatinib, per the §4.3 anchors) -- though dasatinib itself has no D816V
    row in the current cleaned dataset (didn't survive Phase 2's
    aggregation), so it won't appear in this table; that's a real data-
    coverage gap, not a bug in this function.

    Compounds with only one variant measured are excluded (a ratio needs
    both sides). Returns a DataFrame indexed by `id_col` with columns:
    canonical_smiles, p_value_wt, p_value_d816v, censored_wt, censored_d816v,
    log_selectivity, selectivity_ratio, both_censored.

    `both_censored` flags rows where *both* sides are censored (e.g. both
    capped at ">10000 nM" -- Phase 2's capping policy). A ratio computed from
    two censored caps can land near 1.0 purely because both sides hit the
    same bound, not because the compound is genuinely equipotent -- that's a
    real data-quality caveat callers should check, not a computation bug.
    """
    wt = df[df[variant_col] == "WT"].set_index(id_col)
    mut = df[df[variant_col] == "D816V"].set_index(id_col)
    common_ids = wt.index.intersection(mut.index)

    pairs = pd.DataFrame(
        {
            "canonical_smiles": wt.loc[common_ids, "canonical_smiles"],
            "p_value_wt": wt.loc[common_ids, value_col],
            "p_value_d816v": mut.loc[common_ids, value_col],
            "censored_wt": wt.loc[common_ids, "censored"],
            "censored_d816v": mut.loc[common_ids, "censored"],
        }
    )
    pairs["log_selectivity"] = pairs["p_value_wt"] - pairs["p_value_d816v"]
    pairs["selectivity_ratio"] = 10 ** pairs["log_selectivity"]
    pairs["both_censored"] = pairs["censored_wt"] & pairs["censored_d816v"]
    return pairs
