"""Unit tests for src/data_utils.py (Phase 4 step 1: scaffold split; Phase 5
step 1: WT/D816V pairing and selectivity ratio)."""

import numpy as np
import pandas as pd
import pytest

from data_utils import bemis_murcko_scaffold, get_variant_pairs, scaffold_split

BENZENE_SCAFFOLD = "c1ccccc1"
CYCLOHEXANE_SCAFFOLD = "C1CCCCC1"


class TestBemisMurckoScaffold:
    def test_strips_substituents_to_bare_ring(self):
        # Toluene -> its only ring, benzene, with the methyl substituent removed.
        assert bemis_murcko_scaffold("Cc1ccccc1") == BENZENE_SCAFFOLD

    def test_different_substituents_same_scaffold(self):
        # Toluene, benzyl alcohol, chlorobenzene: different molecules, same ring core.
        assert bemis_murcko_scaffold("Cc1ccccc1") == BENZENE_SCAFFOLD
        assert bemis_murcko_scaffold("OCc1ccccc1") == BENZENE_SCAFFOLD
        assert bemis_murcko_scaffold("Clc1ccccc1") == BENZENE_SCAFFOLD

    def test_different_ring_systems_different_scaffold(self):
        benzene_scaffold = bemis_murcko_scaffold("Cc1ccccc1")
        naphthalene_scaffold = bemis_murcko_scaffold("Cc1ccc2ccccc2c1")
        assert benzene_scaffold != naphthalene_scaffold

    def test_acyclic_molecule_has_empty_scaffold(self):
        # Bemis-Murcko scaffolds are defined by ring systems; a molecule with no
        # rings at all (hexane) has no scaffold.
        assert bemis_murcko_scaffold("CCCCCC") == ""

    def test_chirality_ignored(self):
        # include_chirality=False: stereoisomers of the same ring core must
        # collapse to the same scaffold string.
        achiral = bemis_murcko_scaffold("OC1CCCCC1")
        chiral = bemis_murcko_scaffold("O[C@H]1CCCCC1")
        assert achiral == chiral == CYCLOHEXANE_SCAFFOLD

    def test_invalid_smiles_raises_value_error(self):
        with pytest.raises(ValueError, match="not_a_real_smiles"):
            bemis_murcko_scaffold("not_a_real_smiles!!!")


@pytest.fixture
def synthetic_smiles():
    # Rows 0-4: five distinct molecules sharing one large scaffold group (benzene).
    # Rows 5-9: five singleton molecules, each on its own distinct scaffold.
    benzene_group = [
        "Cc1ccccc1",  # toluene
        "OCc1ccccc1",  # benzyl alcohol
        "Clc1ccccc1",  # chlorobenzene
        "Brc1ccccc1",  # bromobenzene
        "Nc1ccccc1",  # aniline
    ]
    singletons = [
        "Cc1ccc2ccccc2c1",  # methylnaphthalene
        "C1CCCCC1",  # cyclohexane
        "c1ccncc1",  # pyridine
        "c1ccoc1",  # furan
        "C1CCCC1",  # cyclopentane
    ]
    return benzene_group + singletons


class TestScaffoldSplit:
    def test_split_sizes_and_disjointness(self, synthetic_smiles):
        train_idx, test_idx = scaffold_split(synthetic_smiles, frac_train=0.8, seed=0)
        assert len(train_idx) + len(test_idx) == len(synthetic_smiles)
        assert set(train_idx).isdisjoint(set(test_idx))

    def test_large_scaffold_group_stays_together_in_train(self, synthetic_smiles):
        # The 5-member benzene group is strictly larger than every singleton
        # group, so it is always assigned to train first regardless of the
        # shuffle order of same-size groups -- true for any seed.
        for seed in range(5):
            train_idx, _ = scaffold_split(synthetic_smiles, frac_train=0.8, seed=seed)
            assert set(range(5)).issubset(set(train_idx))

    def test_no_scaffold_leakage(self, synthetic_smiles):
        train_idx, test_idx = scaffold_split(synthetic_smiles, frac_train=0.8, seed=0)
        train_scaffolds = {bemis_murcko_scaffold(synthetic_smiles[i]) for i in train_idx}
        test_scaffolds = {bemis_murcko_scaffold(synthetic_smiles[i]) for i in test_idx}
        assert train_scaffolds.isdisjoint(test_scaffolds)

    def test_deterministic_given_seed(self, synthetic_smiles):
        train_a, test_a = scaffold_split(synthetic_smiles, frac_train=0.8, seed=0)
        train_b, test_b = scaffold_split(synthetic_smiles, frac_train=0.8, seed=0)
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)

    def test_different_seeds_can_change_split(self, synthetic_smiles):
        splits = {
            seed: tuple(scaffold_split(synthetic_smiles, frac_train=0.8, seed=seed)[1])
            for seed in range(5)
        }
        assert len(set(splits.values())) > 1

    def test_duplicate_smiles_land_on_the_same_side(self, synthetic_smiles):
        # Mirrors real WT/D816V rows: two rows with the identical SMILES (e.g.
        # a compound assayed against both KIT variants) must never be split
        # across train and test.
        smiles_with_duplicate = synthetic_smiles + [synthetic_smiles[0]]
        train_idx, test_idx = scaffold_split(smiles_with_duplicate, frac_train=0.8, seed=0)
        side = {i: "train" for i in train_idx}
        side.update({i: "test" for i in test_idx})
        assert side[0] == side[len(smiles_with_duplicate) - 1]


@pytest.fixture
def bioactivity_df():
    # Mirrors kit_bioactivity_clean.csv's shape: one row per
    # (molecule_chembl_id, kit_variant). CHEMBL_A: paired, WT more potent
    # (like imatinib). CHEMBL_B: paired, comparable potency (like dasatinib),
    # one side censored. CHEMBL_C: WT-only, no D816V row (can't have a
    # ratio). CHEMBL_D: paired, *both* sides censored at the same cap --
    # ratio=1.0 here is a censoring artifact, not real equipotency.
    return pd.DataFrame(
        {
            "molecule_chembl_id": [
                "CHEMBL_A", "CHEMBL_A", "CHEMBL_B", "CHEMBL_B", "CHEMBL_C", "CHEMBL_D", "CHEMBL_D",
            ],
            "kit_variant": ["WT", "D816V", "WT", "D816V", "WT", "WT", "D816V"],
            "canonical_smiles": ["CCO", "CCO", "CCN", "CCN", "CCC", "CCF", "CCF"],
            "p_value": [7.0, 6.0, 7.0, 7.0, 5.0, 5.0, 5.0],
            "censored": [False, False, False, True, False, True, True],
        }
    )


class TestGetVariantPairs:
    def test_only_compounds_with_both_variants_are_included(self, bioactivity_df):
        pairs = get_variant_pairs(bioactivity_df)
        assert set(pairs.index) == {"CHEMBL_A", "CHEMBL_B", "CHEMBL_D"}
        assert "CHEMBL_C" not in pairs.index

    def test_p_values_assigned_to_correct_variant_column(self, bioactivity_df):
        pairs = get_variant_pairs(bioactivity_df)
        assert pairs.loc["CHEMBL_A", "p_value_wt"] == 7.0
        assert pairs.loc["CHEMBL_A", "p_value_d816v"] == 6.0

    def test_selectivity_ratio_greater_than_one_means_more_potent_against_wt(self, bioactivity_df):
        # CHEMBL_A: p_value_wt=7.0 > p_value_d816v=6.0 -- more potent against
        # WT, i.e. loses efficacy against the mutant (the imatinib case).
        pairs = get_variant_pairs(bioactivity_df)
        assert pairs.loc["CHEMBL_A", "selectivity_ratio"] == pytest.approx(10.0)
        assert pairs.loc["CHEMBL_A", "log_selectivity"] == pytest.approx(1.0)

    def test_equal_potency_gives_ratio_of_one(self, bioactivity_df):
        # CHEMBL_B: identical p_value both variants -- comparable potency
        # (the dasatinib case, in spirit).
        pairs = get_variant_pairs(bioactivity_df)
        assert pairs.loc["CHEMBL_B", "selectivity_ratio"] == pytest.approx(1.0)

    def test_censored_flags_carried_through_per_variant(self, bioactivity_df):
        pairs = get_variant_pairs(bioactivity_df)
        assert pairs.loc["CHEMBL_B", "censored_wt"] == False
        assert pairs.loc["CHEMBL_B", "censored_d816v"] == True

    def test_both_censored_flag(self, bioactivity_df):
        pairs = get_variant_pairs(bioactivity_df)
        assert pairs.loc["CHEMBL_D", "both_censored"] == True
        assert pairs.loc["CHEMBL_A", "both_censored"] == False
        assert pairs.loc["CHEMBL_B", "both_censored"] == False  # only one side censored

    def test_no_shared_compounds_gives_empty_result(self):
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["A", "B"],
                "kit_variant": ["WT", "D816V"],
                "canonical_smiles": ["CCO", "CCN"],
                "p_value": [7.0, 6.0],
                "censored": [False, False],
            }
        )
        pairs = get_variant_pairs(df)
        assert len(pairs) == 0

    def test_frac_train_one_puts_everything_in_train(self, synthetic_smiles):
        train_idx, test_idx = scaffold_split(synthetic_smiles, frac_train=1.0, seed=0)
        assert len(train_idx) == len(synthetic_smiles)
        assert len(test_idx) == 0

    def test_frac_train_zero_puts_everything_in_test(self, synthetic_smiles):
        train_idx, test_idx = scaffold_split(synthetic_smiles, frac_train=0.0, seed=0)
        assert len(train_idx) == 0
        assert len(test_idx) == len(synthetic_smiles)

    def test_invalid_smiles_propagates_error(self, synthetic_smiles):
        with pytest.raises(ValueError):
            scaffold_split(synthetic_smiles + ["garbage!!!"], frac_train=0.8, seed=0)
