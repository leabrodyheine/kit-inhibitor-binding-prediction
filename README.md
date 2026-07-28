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

*(To be filled in as the project progresses.)*

## Limitations

- This is a hypothesis-generation and portfolio project, not a validated drug discovery tool — results would need wet-lab confirmation before any biological interpretation is trusted.
- Training data for the wild-type-vs-mutant selectivity task specifically is limited (most ChEMBL KIT data doesn't report both values for the same compound), so that analysis is treated as a smaller validation exercise layered on top of the main binding-affinity model, not a fully supervised task in its own right. No new model is trained for selectivity — the same potency model is queried twice per compound (once per variant).
- That validation exercise is explicitly **in-sample, not held-out**: every one of the 925 compounds with both WT and D816V measurements — including both named anchors, imatinib and dasatinib — landed in the training set under the current scaffold split, since there's currently no paired compound in the test set to check against. The in-sample correlation between the model's predicted and measured selectivity shift (Spearman ρ = 0.86 across those 925 pairs) should be read as evidence the model *fit* the signal, not that it generalizes to unseen compounds. Dasatinib additionally has no D816V measurement in this dataset at all (didn't survive cleaning), so its known ~37nM D816V / ~79nM WT potency remains an external literature reference, not something independently reproduced here.
- i-MCAS is included here for disease-spectrum context, not as a directly modeled indication — there is no confirmed KIT-targeting mutation in i-MCAS, and no confirmed clinical trial data (as of this writing) testing wild-type-KIT-targeting agents specifically in i-MCAS patients.

## Tech Stack

Python · RDKit · ChemBERTa (HuggingFace) · scikit-learn / XGBoost · ChEMBL API · PDB structural data
