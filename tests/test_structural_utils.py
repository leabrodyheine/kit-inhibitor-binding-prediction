"""Unit tests for src/structural_utils.py (Phase 6): the pure, offline
geometry/parsing logic. download_pdb() and fetch_ligand_smiles() are
network-dependent and are instead exercised by actually running
notebooks/06_structural_context.ipynb (same pattern as Phase 1's ChEMBL
API pull)."""

import numpy as np
import pytest
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from structural_utils import (
    get_backbone_trace,
    get_hetero_ligand_codes,
    get_residue_coord,
    ligand_atom_coords,
    min_distance_to_point,
    plot_binding_site_context,
)


def _make_atom(name, coord, serial=1, element="C"):
    return Atom(name, np.array(coord, dtype=float), 0, 1, " ", name, serial, element=element)


@pytest.fixture
def toy_model():
    # A minimal KIT-like model: chain A has three protein residues (814-816,
    # so there's a small backbone to trace), a water, and a ligand. Residue
    # 816 is ASP -- the wild-type residue that mutates to Val in D816V.
    structure = Structure("toy")
    model = Model(0)
    structure.add(model)
    chain = Chain("A")
    model.add(chain)

    res814 = Residue((" ", 814, " "), "GLY", " ")
    res814.add(_make_atom("CA", [-2.0, 0.0, 0.0]))
    chain.add(res814)

    res815 = Residue((" ", 815, " "), "PHE", " ")
    res815.add(_make_atom("CA", [-1.0, 0.0, 0.0]))
    chain.add(res815)

    res816 = Residue((" ", 816, " "), "ASP", " ")
    res816.add(_make_atom("CA", [0.0, 0.0, 0.0]))
    chain.add(res816)

    water = Residue(("W", 1, " "), "HOH", " ")
    water.add(_make_atom("O", [100.0, 100.0, 100.0], element="O"))
    chain.add(water)

    ligand = Residue(("H_LIG", 900, " "), "LIG", " ")
    ligand.add(_make_atom("C1", [5.0, 0.0, 0.0]))
    ligand.add(_make_atom("C2", [3.0, 0.0, 0.0]))
    chain.add(ligand)

    return model


class TestGetHeteroLigandCodes:
    def test_excludes_water_by_default(self, toy_model):
        codes = get_hetero_ligand_codes(toy_model)
        assert codes == {"LIG"}

    def test_does_not_include_protein_residues(self, toy_model):
        codes = get_hetero_ligand_codes(toy_model)
        assert "ASP" not in codes

    def test_custom_exclude_set(self, toy_model):
        codes = get_hetero_ligand_codes(toy_model, exclude=frozenset({"HOH", "LIG"}))
        assert codes == set()


class TestGetResidueCoord:
    def test_returns_coord_and_resname(self, toy_model):
        coord, resname = get_residue_coord(toy_model, "A", 816, atom_name="CA")
        assert resname == "ASP"
        assert np.array_equal(coord, np.array([0.0, 0.0, 0.0]))

    def test_missing_residue_raises_key_error(self, toy_model):
        with pytest.raises(KeyError):
            get_residue_coord(toy_model, "A", 999)


class TestLigandAtomCoords:
    def test_returns_all_ligand_atom_coords(self, toy_model):
        coords = ligand_atom_coords(toy_model, "A", "LIG")
        assert coords.shape == (2, 3)
        assert np.array_equal(sorted(coords.tolist()), sorted([[5.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))

    def test_missing_ligand_raises_key_error(self, toy_model):
        with pytest.raises(KeyError):
            ligand_atom_coords(toy_model, "A", "NOPE")


class TestMinDistanceToPoint:
    def test_finds_the_closest_atom(self):
        coords = np.array([[10.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        point = np.array([0.0, 0.0, 0.0])
        assert min_distance_to_point(coords, point) == pytest.approx(3.0)

    def test_zero_distance_when_point_matches_an_atom(self):
        coords = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        point = np.array([1.0, 1.0, 1.0])
        assert min_distance_to_point(coords, point) == pytest.approx(0.0)

    def test_end_to_end_with_toy_model(self, toy_model):
        # The ligand's closest atom to residue 816 is at x=3.0, and residue
        # 816's CA is at the origin -- distance should be exactly 3.0.
        residue_coord, _ = get_residue_coord(toy_model, "A", 816)
        ligand_coords = ligand_atom_coords(toy_model, "A", "LIG")
        assert min_distance_to_point(ligand_coords, residue_coord) == pytest.approx(3.0)


class TestGetBackboneTrace:
    def test_returns_one_coord_per_protein_residue(self, toy_model):
        backbone = get_backbone_trace(toy_model, "A")
        assert backbone.shape == (3, 3)  # residues 814, 815, 816

    def test_excludes_water_and_ligand(self, toy_model):
        backbone = get_backbone_trace(toy_model, "A")
        # Water is at [100, 100, 100] and ligand atoms at x=3/5 -- neither
        # should appear in the protein-only backbone trace.
        assert not np.any(np.all(backbone == [100.0, 100.0, 100.0], axis=1))
        assert backbone.shape[0] == 3

    def test_coords_in_residue_order(self, toy_model):
        backbone = get_backbone_trace(toy_model, "A")
        expected = np.array([[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert np.array_equal(backbone, expected)


class TestPlotBindingSiteContext:
    def test_returns_figure(self, toy_model):
        fig = plot_binding_site_context(toy_model, "A", "LIG")
        try:
            assert fig is not None
            assert len(fig.axes) == 1
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_saves_a_nonempty_file(self, toy_model, tmp_path):
        save_path = tmp_path / "binding_site.png"
        fig = plot_binding_site_context(toy_model, "A", "LIG", save_path=save_path)
        try:
            assert save_path.exists()
            assert save_path.stat().st_size > 0
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_missing_ligand_raises_key_error(self, toy_model):
        with pytest.raises(KeyError):
            plot_binding_site_context(toy_model, "A", "NOPE")
