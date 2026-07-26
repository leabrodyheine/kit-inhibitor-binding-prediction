# Implementation Plan: KIT Inhibitor Binding & Mutant-Selectivity Prediction

Step-by-step build plan derived from [Design Doc.md](Design%20Doc.md) and [README.md](README.md). Organized as sequential phases mapped to the repo structure in Design Doc §6. Check items off as completed; fill in `Results` sections with real numbers as they land (no placeholders left at the end, per Design Doc §8 Day 2).

## Phase 0 — Environment & Scaffolding

- [x] Create repo structure per Design Doc: `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `results/figures/`.
- [x] Write `environment.yml` (Python 3.10+, RDKit, `chembl_webresource_client`, `transformers`, scikit-learn, XGBoost, pandas, NumPy, matplotlib/seaborn).
- [x] Create and activate the conda/venv environment; verify RDKit imports and a HuggingFace model loads.
- [ ] Initialize `src/__init__.py` and stub modules: `data_utils.py`, `featurization.py`, `models.py`, `evaluation.py`.

## Phase 1 — Data Collection (`notebooks/01_data_collection.ipynb`)

- [x] Query ChEMBL for the human KIT target (UniProt P10721) via `chembl_webresource_client`. — resolved to `CHEMBL1936` (Mast/stem cell growth factor receptor Kit, Homo sapiens), confirmed cross-reference to UniProt P10721.
- [ ] Pull all bioactivity records: SMILES, assay type (IC50/Ki/Kd/EC50), reported value + units, assay/document metadata.
- [ ] Save raw, unmodified pull to `data/raw/`.
- [ ] Sanity-check record count and spot-check a few known compounds (e.g., imatinib, dasatinib) are present.
- [ ] Note in README/notebook: actual record count retrieved vs. the "several hundred to a few thousand" estimate in Design Doc §4.1.

## Phase 2 — Data Cleaning & Standardization (`notebooks/02_data_cleaning.ipynb`, `src/data_utils.py`)

Implements Design Doc §5.1 and the quality issues in §4.4.

- [ ] Convert potency values to pIC50/pKi (-log10 molar concentration) on a common scale.
- [ ] Explicit policy for mixed assay types (IC50 vs Ki vs Kd) — document what's combined and why.
- [ ] Deduplicate/aggregate repeated measurements per compound (e.g., median of log-transformed values).
- [ ] Handle censored values (">10000 nM" etc.) — decide drop vs. cap vs. flag, and implement consistently.
- [ ] Canonicalize SMILES via RDKit to collapse representation duplicates.
- [ ] Flag/segment records by wild-type vs. D816V-mutant assay target where annotated, in preparation for Phase 5.
- [ ] Save cleaned dataset to `data/processed/`.
- [ ] Document every filtering decision inline (row counts before/after each step) — this is called out explicitly in Design Doc §8 Day 1.

## Phase 3 — Featurization (`notebooks/03_featurization.ipynb`, `src/featurization.py`)

Implements Design Doc §5.2.

- [ ] Baseline: compute ECFP (Morgan) fingerprints via RDKit for all compounds.
- [ ] Embedding-based: generate ChemBERTa embeddings (HuggingFace, frozen) for all compounds.
- [ ] Cache both feature sets to `data/processed/` (avoid recomputation in later notebooks).
- [ ] Confirm feature matrices align row-for-row with the cleaned dataset (no silent SMILES drops/reordering).

## Phase 4 — Modeling & Evaluation (`notebooks/04_model_training.ipynb`, `src/models.py`, `src/evaluation.py`)

Implements Design Doc §5.3–§5.5.

- [ ] Implement **scaffold split** (not random split) for train/test — group by Bemis-Murcko scaffold before splitting.
- [ ] Train baseline model: XGBoost on ECFP fingerprints.
- [ ] Train comparison model: small MLP head on frozen ChemBERTa embeddings.
- [ ] Evaluate both on held-out set: RMSE, R², Spearman rank correlation.
- [ ] Compare baseline vs. embedding model directly — do not assume the more sophisticated model wins (per Design Doc §5.3 caution, echoing the dissertation's GAN-vs-augmentation finding).
- [ ] Save trained models and evaluation metrics; write comparison table to `results/`.
- [ ] Generate diagnostic plots (predicted vs. actual, residuals) to `results/figures/`.

**Results (fill in after running):**

| Model | RMSE | R² | Spearman ρ |
| --- | --- | --- | --- |
| XGBoost + ECFP | | | |
| MLP + ChemBERTa | | | |

## Phase 5 — Selectivity Analysis (`notebooks/05_selectivity_analysis.ipynb`)

Implements Design Doc §5.6, using data isolated in Phase 2 and anchors from §4.3.

- [ ] Identify compounds with both wild-type and D816V bioactivity records; compute selectivity ratio (WT potency / mutant potency).
- [ ] Apply the Phase 4 potency model separately to wild-type- and mutant-labeled subsets.
- [ ] Check the model reproduces the known direction of the two anchor points:
  - Dasatinib: ~37 nM (D816V) vs. ~79 nM (WT) — comparable potency both ways.
  - Imatinib: known to lose efficacy against D816V relative to WT.
- [ ] Explicitly frame this as a validation exercise on a handful of anchor compounds, **not** a standalone trained classifier — state this caveat directly in the notebook and README, per Design Doc §5.6 and §9.

## Phase 6 — Structural Sanity Check (`notebooks/06_structural_context.ipynb`)

Implements Design Doc §5.7, using PDB structures from §4.2.

- [ ] Pull/inspect the listed PDB structures: 1T45, 1PKG, 4HVS, 6XV9, 6XVA, 6GQJ, 8PQA, 8PQB, 8PQD.
- [ ] For top predicted mutant-selective and wild-type-potent compounds (or closest structural analogues present in the PDB set), visually inspect binding pose relative to the D816V activation-loop mutation site.
- [ ] Confirm predictions are at least structurally plausible — no docking simulation required, qualitative check only.
- [ ] Save annotated structure images/figures to `results/figures/`.

## Phase 7 — Write-Up & Finalization

- [ ] Replace all "to be filled in" placeholders in README.md `## Results` with actual numbers/findings.
- [ ] Update Design Doc / README limitations sections if anything surprising surfaced during implementation (e.g., data volume lower than expected, class imbalance worse than expected).
- [ ] Final pass: confirm every data-handling decision from Phase 2 is documented, and the selectivity analysis caveat (§5.6/§9) is stated prominently, not glossed over.
- [ ] Optional (time permitting, per Design Doc §10 Future Work — do not scope into this pass): note follow-ups (ESM-2 for antibody agents, docking, PDGFRA extension) as explicit "not attempted" items rather than implying completeness.
