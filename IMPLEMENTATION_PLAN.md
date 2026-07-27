# Implementation Plan: KIT Inhibitor Binding & Mutant-Selectivity Prediction

Step-by-step build plan derived from [Design Doc.md](Design%20Doc.md) and [README.md](README.md). Organized as sequential phases mapped to the repo structure in Design Doc §6. Check items off as completed; fill in `Results` sections with real numbers as they land (no placeholders left at the end, per Design Doc §8 Day 2).

## Phase 0 — Environment & Scaffolding

- [x] Create repo structure per Design Doc: `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `results/figures/`.
- [x] Write `environment.yml` (Python 3.10+, RDKit, `chembl_webresource_client`, `transformers`, scikit-learn, XGBoost, pandas, NumPy, matplotlib/seaborn).
- [x] Create and activate the conda/venv environment; verify RDKit imports and a HuggingFace model loads.
- [x] Initialize `src/__init__.py` and stub modules: `data_utils.py`, `featurization.py`, `models.py`, `evaluation.py`. — `featurization.py` filled in with real implementations as part of Phase 3 (below); `data_utils.py`/`models.py`/`evaluation.py` left as stubs for Phase 4.

## Phase 1 — Data Collection (`notebooks/01_data_collection.ipynb`)

- [x] Query ChEMBL for the human KIT target (UniProt P10721) via `chembl_webresource_client`. — resolved to `CHEMBL1936` (Mast/stem cell growth factor receptor Kit, Homo sapiens), confirmed cross-reference to UniProt P10721.
- [x] Pull all bioactivity records: SMILES, assay type (IC50/Ki/Kd/EC50), reported value + units, assay/document metadata. — 8,703 records pulled via `notebooks/01_data_collection.ipynb`.
- [x] Save raw, unmodified pull to `data/raw/`. — saved as `chembl_kit_bioactivity_raw.csv` (gitignored as reproducible; regenerate by re-running the notebook).
- [x] Sanity-check record count and spot-check a few known compounds (e.g., imatinib, dasatinib) are present. — imatinib: 69 records, dasatinib: 36, avapritinib: 7.
- [x] Note in README/notebook: actual record count retrieved vs. the "several hundred to a few thousand" estimate in Design Doc §4.1. — 8,703 records, upper end of estimate; also found 2,291 records (~26%) mention D816V in assay descriptions, more mutant-KIT data than §4.4 anticipated.

## Phase 2 — Data Cleaning & Standardization (`notebooks/02_data_cleaning.ipynb`, `src/data_utils.py`)

Implements Design Doc §5.1 and the quality issues in §4.4.

- [x] Convert potency values to pIC50/pKi (-log10 molar concentration) on a common scale. — computed for 8,547/8,703 rows in `notebooks/02_data_cleaning.ipynb`; cross-checked against ChEMBL's own `pchembl_value` (max deviation 0.0055).
- [x] Explicit policy for mixed assay types (IC50 vs Ki vs Kd) — document what's combined and why. — pooled IC50+Ki+Kd (8,497 rows), dropped EC50 (206 rows, ~2%); cross-type divergence for shared compounds is 0.45-0.80 log units median |diff|, judged comparable to paper-to-paper noise rather than a distinct correction problem. `standard_type` retained as a column.
- [x] Deduplicate/aggregate repeated measurements per compound (e.g., median of log-transformed values). — aggregated by `(molecule_chembl_id, kit_variant)`, not compound alone (see next item); 6,127 exact measurements → 4,093 aggregated rows (median p_value, 1,215 groups had >1 measurement).
- [x] Handle censored values (">10000 nM" etc.) — decide drop vs. cap vs. flag, and implement consistently. — capped rather than dropped: 1,556/5,649 groups (28%) have no exact measurement; 1,476 (95% of those) capped at their tightest one-directional bound and flagged `censored=True`/`censored_direction`, only 80 (5%, conflicting or no usable relation at all) dropped. Avoids worsening ChEMBL's known bias against inactive compounds (Design Doc §9).
- [x] Canonicalize SMILES via RDKit to collapse representation duplicates. — combined exact-median (4,093) + capped-censored (1,476) into one `df_clean` table (5,569 rows), canonicalized via RDKit; 4 rows dropped for missing/unparseable SMILES, 0 discrepancies vs. ChEMBL's own canonical_smiles, 0 duplicate structures within a variant. Final: 5,565 rows.
- [x] Flag/segment records by wild-type vs. D816V-mutant assay target where annotated, in preparation for Phase 5. — done ahead of schedule, as part of the dedup step above: naively deduping ignoring variant would have averaged WT and D816V together, destroying signal — 746 compounds have both, and 226 (30%) differ by >1 log unit between them. WT: 6,206 records, D816V: 2,291.
- [x] Save cleaned dataset to `data/processed/`. — saved as `kit_bioactivity_clean.csv` (5,565 rows; gitignored as reproducible, regenerate by re-running the notebook).
- [x] Document every filtering decision inline (row counts before/after each step) — this is called out explicitly in Design Doc §8 Day 1. — every section has inline before/after counts; §7 adds a consolidated funnel table (8,703 raw → 5,565 clean) for a single-glance summary.

## Phase 3 — Featurization (`notebooks/03_featurization.ipynb`, `src/featurization.py`)

Implements Design Doc §5.2.

- [x] Baseline: compute ECFP (Morgan) fingerprints via RDKit for all compounds. — radius 2, 2048 bits, via `rdFingerprintGenerator`; computed in `notebooks/03_featurization.ipynb` using `src/featurization.py`.
- [x] Embedding-based: generate ChemBERTa embeddings (HuggingFace, frozen) for all compounds. — `seyonec/ChemBERTa-zinc-base-v1`, frozen, mean-pooled over non-padding tokens (768-dim).
- [x] Cache both feature sets to `data/processed/` (avoid recomputation in later notebooks). — saved as `ecfp_fingerprints.npy` (5,565 × 2,048, uint8) and `chemberta_embeddings.npy` (5,565 × 768, float32); gitignored as reproducible like other processed artifacts.
- [x] Confirm feature matrices align row-for-row with the cleaned dataset (no silent SMILES drops/reordering). — featurized 4,640 unique compounds once (5,565 rows include duplicate SMILES across WT/D816V variant pairs), then expanded via SMILES→index lookup back to the full 5,565-row order; shape asserted against `len(df)` and 5 random rows spot-checked by direct recomputation.

## Phase 4 — Modeling & Evaluation (`notebooks/04_model_training.ipynb`, `src/models.py`, `src/evaluation.py`)

Implements Design Doc §5.3–§5.5.

- [x] Implement **scaffold split** (not random split) for train/test — group by Bemis-Murcko scaffold before splitting. — `scaffold_split()`/`bemis_murcko_scaffold()` added to `src/data_utils.py`; groups whole scaffolds (largest-first) into train until an 80% target, remainder to test. Result: 4,452 train rows (814 scaffolds) / 1,113 test rows (1,113 scaffolds), zero scaffold overlap confirmed directly; all 925 compounds with paired WT/D816V measurements landed in train under this seed (relevant for Phase 5). Deterministic given a seed.
- [x] Train baseline model: XGBoost on ECFP fingerprints. — `train_xgboost_ecfp()` added to `src/models.py`, trained on the 4,452-row scaffold-split train set. Sanity-checked (not the formal Phase 4 step 4 evaluation): training R² = 0.809, held-out predictions non-degenerate and positively correlated with truth (Pearson r = 0.725 on the 1,113-row test set) — no hyperparameter tuning done yet.
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
