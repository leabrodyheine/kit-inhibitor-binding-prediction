"""Unit tests for src/featurization.py (Phase 3: ECFP + ChemBERTa)."""

import numpy as np
import pytest

from featurization import compute_chemberta_embeddings, compute_ecfp_fingerprints

TOLUENE = "Cc1ccccc1"
BENZYL_ALCOHOL = "OCc1ccccc1"
NAPHTHALENE = "c1ccc2ccccc2c1"


class TestComputeEcfpFingerprints:
    def test_shape_and_dtype(self):
        fps = compute_ecfp_fingerprints([TOLUENE, NAPHTHALENE], radius=2, n_bits=2048)
        assert fps.shape == (2, 2048)
        assert fps.dtype == np.uint8

    def test_values_are_binary(self):
        fps = compute_ecfp_fingerprints([TOLUENE, NAPHTHALENE, BENZYL_ALCOHOL])
        assert set(np.unique(fps)).issubset({0, 1})

    def test_deterministic(self):
        fps1 = compute_ecfp_fingerprints([TOLUENE])
        fps2 = compute_ecfp_fingerprints([TOLUENE])
        assert np.array_equal(fps1, fps2)

    def test_different_molecules_give_different_fingerprints(self):
        fps = compute_ecfp_fingerprints([TOLUENE, NAPHTHALENE])
        assert not np.array_equal(fps[0], fps[1])

    def test_same_molecule_different_smiles_string_gives_same_fingerprint(self):
        # Toluene written two different (both valid, non-canonical) ways --
        # the fingerprint is a function of the molecular graph, not the
        # input string, so these must match exactly.
        fp_a = compute_ecfp_fingerprints(["Cc1ccccc1"])[0]
        fp_b = compute_ecfp_fingerprints(["c1ccc(C)cc1"])[0]
        assert np.array_equal(fp_a, fp_b)

    def test_custom_radius_and_n_bits(self):
        fps = compute_ecfp_fingerprints([TOLUENE], radius=3, n_bits=512)
        assert fps.shape == (1, 512)

    def test_invalid_smiles_raises_value_error_naming_the_string(self):
        with pytest.raises(ValueError, match="garbage_smiles"):
            compute_ecfp_fingerprints([TOLUENE, "garbage_smiles!!!"])


class TestComputeChembertaEmbeddings:
    @classmethod
    @pytest.fixture(scope="class")
    def sample_smiles(cls):
        return [TOLUENE, BENZYL_ALCOHOL, NAPHTHALENE]

    @classmethod
    @pytest.fixture(scope="class")
    def embeddings(cls, sample_smiles):
        return compute_chemberta_embeddings(sample_smiles, batch_size=32)

    def test_shape_and_dtype(self, embeddings, sample_smiles):
        assert embeddings.shape == (len(sample_smiles), 768)
        assert embeddings.dtype == np.float32

    def test_no_nans_or_infs(self, embeddings):
        assert np.isfinite(embeddings).all()

    def test_different_molecules_give_different_embeddings(self, embeddings):
        assert not np.allclose(embeddings[0], embeddings[2])

    def test_deterministic(self, sample_smiles, embeddings):
        again = compute_chemberta_embeddings(sample_smiles, batch_size=32)
        assert np.allclose(embeddings, again)

    def test_batch_size_does_not_change_per_molecule_output(self, sample_smiles, embeddings):
        # Padding/attention masking must isolate each example from its
        # batch-mates -- a molecule's embedding shouldn't depend on what
        # else happened to be in its batch.
        one_at_a_time = compute_chemberta_embeddings(sample_smiles, batch_size=1)
        assert np.allclose(embeddings, one_at_a_time, atol=1e-5)
