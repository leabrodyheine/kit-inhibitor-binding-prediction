# KIT Inhibitor Binding & Mutant-Selectivity Prediction

Predicting small-molecule binding affinity against KIT — the kinase driving systemic mastocytosis — using chemical language model embeddings and public bioactivity data, with a focus on **mutant (D816V) vs. wild-type selectivity**.

## Motivation

Mast cell diseases sit on a spectrum, and the KIT receptor tyrosine kinase is central to nearly all of them — but not in the same way for every patient:

- **Systemic mastocytosis** is driven by an activating KIT mutation (most commonly D816V). Treatment requires **mutant-selective** inhibitors that spare wild-type KIT, since normal mast cells and blood cells still need functional wild-type KIT to survive.
- **Non-clonal mast cell diseases** — chronic urticaria, and mechanistically related conditions like idiopathic mast cell activation syndrome (i-MCAS) — involve inappropriately active mast cells *without* a KIT mutation. An emerging therapeutic strategy here is the opposite: deliberately targeting **wild-type** KIT to deplete mast cells altogether (e.g. a small-molecule wild-type KIT inhibitor or an anti-KIT antibody).

This project asks: **given a small molecule's structure, can we predict not just its binding affinity for KIT, but whether it behaves as mutant-selective or wild-type-potent?** That distinction is exactly what determines which patient population a compound is suited for.

This work extends the mast cell disease research from my MSc dissertation (*Machine Learning for Pathology in Mast Cell Diseases*, University of St Andrews), moving from clinical/microbiome pattern discovery into the underlying drug-target mechanism.

## Scope

- **Modality**: small molecules only. Antibody-based approaches (e.g., barzolvolimab) work through a different mechanism and would require protein-level embedding methods (e.g., ESM-2) rather than the chemical-structure models used here — noted as future work, not attempted in this repo.
- **Target**: KIT kinase domain, wild-type and D816V mutant.
- **Task**: binding affinity / potency prediction, with a secondary analysis of mutant-vs-wild-type selectivity where paired data is available.

## Data Sources

- **ChEMBL**: compound structures and bioactivity data (IC50/Ki) for KIT.
- **PDB**: KIT kinase domain structures (wild-type and mutant), several in complex with known inhibitors (e.g., avapritinib, dasatinib, imatinib) — used for structural context, not full docking.
- Published wild-type vs. D816V potency comparisons from the literature, used as a small validation set for the selectivity analysis rather than as training data.

## Approach

1. **Data collection**: pull KIT bioactivity records from ChEMBL via its API; assemble compound structures (SMILES) and associated potency values.
2. **Molecular representation**: generate embeddings for each compound using RDKit fingerprints and/or a pretrained chemical language model (ChemBERTa).
3. **Binding affinity model**: train a regression/classification model (e.g., gradient-boosted trees or a small MLP on top of the embeddings) to predict potency against KIT.
4. **Selectivity analysis**: using the subset of compounds with both wild-type and D816V activity data, evaluate whether the model — or a simple derived feature — captures the mutant-vs-wild-type potency shift, anchored against known examples (e.g., dasatinib retaining potency against D816V where imatinib does not).
5. **Structural sanity check**: cross-reference top predicted mutant-selective vs. wild-type-potent compounds against binding pocket location in available PDB structures.

## Results

### 1. Data collection & cleaning

8,703 raw KIT bioactivity records pulled from ChEMBL (`CHEMBL1936`), cleaned and standardized down to **5,565 rows** (see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) Phase 2 for the full funnel and every filtering decision, documented inline with before/after counts). 2,291 records (~26%) referenced the D816V mutation — more mutant-KIT data than initially expected.

### 2. Binding affinity model

Scaffold-split (not random — grouped by Bemis-Murcko scaffold to avoid structurally near-identical compounds leaking across train/test): 4,452 train / 1,113 test rows.

| Model | RMSE | R² | Spearman ρ |
| --- | --- | --- | --- |
| XGBoost + ECFP fingerprints (tuned) | **0.830** | **0.541** | **0.714** |
| MLP + frozen ChemBERTa embeddings | 1.125 | 0.156 | 0.551 |

Naive mean-baseline RMSE on the same held-out set: 1.233 — both models clearly beat it. XGBoost on plain structural fingerprints beat the ChemBERTa-based model on every metric, in 100% of 1,000 bootstrap resamples of the test set (not a fluke of one fixed split) — a direct example of the "don't assume the fancier representation wins" caution this project set out to test. Most likely explanation: ChemBERTa is used frozen (not fine-tuned) here, and ~4,452 training compounds favors a sample-efficient tree ensemble over a neural net trained from scratch on generic embeddings.

The XGBoost number above is *tuned*: hyperparameters were selected via `RandomizedSearchCV` (80 candidates × 5 folds) using scaffold-grouped cross-validation (not plain `KFold`, which would repeat the exact leakage problem the train/test scaffold split exists to prevent) on the training set only, with the test set touched exactly once at the end. Tuning improved RMSE from 0.849 (untuned defaults) to 0.830 — a real but modest ~2% gain. That ceiling is expected: Phase 2 found ChEMBL's own repeated measurements of the *same* compound disagree by 0.45–0.8 log units across assay types, a noise floor baked into the labels themselves. No amount of tuning predicts past noise in the target — an RMSE near that range is close to what this data can legitimately support, and claims of near-perfect accuracy on this kind of task would be a red flag (leakage or a bug), not a genuine result.

Diagnostic plots (predicted vs. actual, residuals) are in `results/figures/model_diagnostics.png`; full trained models and metrics are in `results/models/` and `results/model_comparison.csv`.

**Addendum — variant-aware model:** the models above predict from structure alone, so a compound's wild-type and D816V measurements — identical structural features either way — are indistinguishable to them. Adding a binary variant flag and retraining (`results/models/xgb_ecfp_variant_aware.joblib`, same tuned hyperparameters) fixed this without hurting held-out performance (tuned RMSE = 0.824, R² = 0.548, Spearman = 0.719), and is the model used for the selectivity analysis below.

**Addendum — ensembling:** XGBoost and the MLP use different representations (structural fingerprints vs. a pretrained chemical language model), so their errors could plausibly be partly uncorrelated — worth checking before treating a single model as the ceiling. A weighted average (`alpha × XGBoost + (1 − alpha) × MLP`) was tried, with `alpha` selected via 5-fold scaffold-grouped out-of-fold cross-validation on the training set only (never the test set, for the same leakage reason as the tuning step above). The best weight found, alpha = 0.87, gives the best result in this project — RMSE = 0.820, R² = 0.552, Spearman = 0.721 — and the gain is bootstrap-robust (beat XGBoost alone in 100% of 1,000 RMSE/R² resamples, 98.3% of Spearman resamples), not a fluke of one fixed test set. It's a small, real improvement, not a game-changer: the two models weren't different enough for ensembling to do much better than the stronger one alone. This isn't a saved model file — reproducing it means running both models and combining outputs with this weight, a modest added serving cost.

### 3. Selectivity analysis

925 compounds have both a WT and a D816V bioactivity record. Selectivity is broadly distributed, not concentrated near "no difference": excluding 65 compounds where both sides are censored at the same bound (an uninformative artifact, not real equipotency), 40.3% are >2-fold more potent against WT, 27.3% >2-fold more potent against D816V, and 32.3% comparable.

Anchor check against the two literature-known compounds (Design Doc §4.3), using the tuned variant-aware model:

- **Imatinib** (known to lose efficacy against D816V): model predicts WT p=7.08 > D816V p=6.48 — correct direction, 3.93× more potent against WT (notably close to the measured 9.07× for this compound in the raw data).
- **Dasatinib** (known to retain comparable potency against both): model predicts a much smaller shift, 1.37× — 4.4× smaller than imatinib's, correctly the *smaller* shift, even though this dataset has no measured D816V value for dasatinib at all (didn't survive cleaning; the model still produces a prediction from structure + the variant flag).

Both directional checks pass, verified with explicit assertions in the notebook, not eyeballed. **This is an in-sample validation exercise on 925 pairs plus two named anchors, not a held-out generalization claim or a standalone selectivity classifier** — see Limitations below.

### 4. Structural sanity check

All 9 PDB structures (Design Doc §4.2) parsed successfully; residue 816 confirmed as **ASP** (wild-type numbering) in every one. 7/9 have a drug-like co-crystallized ligand (1T45 is apo/autoinhibited, 1PKG has only ADP/Mg/phosphotyrosine — matching Design Doc's own description of both).

The top 5 predicted mutant-selective and top 5 predicted WT-potent compounds (per the tuned model) were matched to their closest structural analogue among the 7 PDB ligands (Tanimoto similarity on ECFP, honestly modest at 0.18–0.33 — these are ChEMBL screening compounds, not the exact drug candidates crystallized). For the best match in each category, the ligand sits 18.3 Å (WT-potent case) and 11.5 Å (mutant-selective case) from residue 816's Cα — both within the geometrically expected range for an ATP-competitive kinase inhibitor near the activation loop. No docking was run; this is a plausibility check, not a binding-pose or affinity claim. Figures: `results/figures/structural_context_*.png`.

## Limitations

- This is a hypothesis-generation and portfolio project, not a validated drug discovery tool — results would need wet-lab confirmation before any biological interpretation is trusted.
- Training data for the wild-type-vs-mutant selectivity task specifically is limited (most ChEMBL KIT data doesn't report both values for the same compound), so that analysis is treated as a smaller validation exercise layered on top of the main binding-affinity model, not a fully supervised task in its own right. No new model is trained for selectivity — the same potency model is queried twice per compound (once per variant).
- That validation exercise is explicitly **in-sample, not held-out**: every one of the 925 compounds with both WT and D816V measurements — including both named anchors, imatinib and dasatinib — landed in the training set under the current scaffold split, since there's currently no paired compound in the test set to check against. The in-sample correlation between the model's predicted and measured selectivity shift (Spearman ρ = 0.84 across those 925 pairs) should be read as evidence the model *fit* the signal, not that it generalizes to unseen compounds. Dasatinib additionally has no D816V measurement in this dataset at all (didn't survive cleaning), so its known ~37nM D816V / ~79nM WT potency remains an external literature reference, not something independently reproduced here.
- Held-out potency prediction has a real ceiling on this data: Phase 2 found ChEMBL's own repeated measurements of the *same* compound disagree by 0.45–0.8 log units across assay types (a noise floor in the labels themselves). RMSE around 0.82–0.85 on this held-out set — including after tuning and ensembling — is near that ceiling — expect this, not near-perfect accuracy, from any model trained on this data honestly.
- i-MCAS is included here for disease-spectrum context, not as a directly modeled indication — there is no confirmed KIT-targeting mutation in i-MCAS, and no confirmed clinical trial data (as of this writing) testing wild-type-KIT-targeting agents specifically in i-MCAS patients.

## Tech Stack

Python · RDKit · ChemBERTa (HuggingFace) · scikit-learn / XGBoost · ChEMBL API · PDB structural data
