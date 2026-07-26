# Design Document: KIT Inhibitor Binding & Mutant-Selectivity Prediction

## 1. Summary

This project builds a machine learning pipeline to predict small-molecule binding affinity/potency against the KIT receptor tyrosine kinase, with a specific secondary goal of distinguishing **mutant-selective** compounds (relevant to systemic mastocytosis) from **wild-type-potent** compounds (relevant to non-clonal mast cell diseases). It uses public bioactivity data (ChEMBL), structural data (PDB), and pretrained chemical representations (RDKit fingerprints, ChemBERTa) rather than any proprietary or novel wet-lab data.

## 2. Problem Statement & Motivation

### 2.1 The biological problem

KIT is a receptor tyrosine kinase required for mast cell development, proliferation, and survival, activated by its ligand stem cell factor (SCF). Its role differs across the mast cell disease spectrum:

- **Systemic mastocytosis** is driven by a somatic activating mutation in KIT — most commonly D816V — which causes ligand-independent, constitutive kinase activity and uncontrolled mast cell proliferation. Standard-of-care treatment (e.g., avapritinib, midostaurin) works by inhibiting this mutant form.
- **Non-clonal mast cell diseases** — chronic urticaria and, mechanistically, conditions like idiopathic mast cell activation syndrome (i-MCAS) — involve mast cells that are inappropriately active or numerous but carry no KIT mutation. Because mast cells require KIT/SCF signaling simply to survive, an emerging therapeutic strategy for this group is to deliberately inhibit **wild-type** KIT, depleting mast cells regardless of mutation status (e.g., BLU-808, a small-molecule wild-type-selective inhibitor; barzolvolimab, an anti-KIT antibody with the most mature clinical data of any KIT-depleting agent to date, primarily studied in chronic urticaria).

These two strategies are not interchangeable: a mutant-selective drug given to a wild-type-disease patient would do little, and a pan-KIT or wild-type-potent drug given to a mastocytosis patient risks depleting the patient's normal, healthy mast cells and blood-lineage cells alongside the diseased ones. **Which category a compound falls into is therefore a clinically meaningful, patient-population-defining property** — not just a potency number.

### 2.2 The ML problem

Given a compound's structure, can we:

1. Predict its binding affinity/potency against KIT (a standard cheminformatics regression task), and
2. Estimate whether it behaves as mutant-selective or wild-type-potent, using the limited paired data available in the literature as a validation anchor?

This is a realistic, scoped version of the modeling problem drug discovery teams actually face early in a KIT-focused program: triaging compound series by target-population fit, not just raw potency.

### 2.3 Why this project, for this portfolio

This extends the mast cell disease research from my MSc dissertation (*Machine Learning for Pathology in Mast Cell Diseases*), moving from clinical/microbiome pattern discovery into the molecular mechanism and drug-target layer — while introducing a skill (molecular representation learning) not otherwise demonstrated in that work.

## 3. Goals & Non-Goals

**Goals:**

- Build a working, reproducible pipeline: data retrieval → featurization → model training → evaluation.
- Produce a defensible potency-prediction model on a public benchmark-quality dataset.
- Produce an honest, appropriately caveated analysis of the mutant-vs-wild-type selectivity question.
- Document biological reasoning alongside the ML, so the project reads as domain-informed, not just a generic modeling exercise.

**Non-goals:**

- No claim of clinical or drug-development validity — this is a portfolio/hypothesis-generation project.
- No docking or physics-based binding simulation (structural data is used qualitatively, not to compute binding poses).
- No antibody modeling (barzolvolimab and similar biologics require a different representation approach — noted as future work).
- No modeling of i-MCAS as a directly-treated indication — it is discussed only as biological context, since no KIT mutation exists to target and no confirmed trial data links wild-type-KIT-targeting agents to i-MCAS specifically.

## 4. Data

### 4.1 Primary source: ChEMBL

- Query ChEMBL for the KIT target (UniProt P10721 / ChEMBL target ID for human KIT).
- Pull all associated bioactivity records: compound structures (SMILES), assay type (IC50, Ki, Kd, EC50), reported value, units, and assay/document metadata.
- Expect several hundred to a few thousand data points, with substantial heterogeneity in assay type and quality — this is typical for a well-studied kinase target and needs explicit handling (see 5.1).

### 4.2 Structural context: PDB

Used qualitatively, not for docking. Representative structures worth pulling directly (found via targeted search, confirmed available):

- **1T45** — inactive (autoinhibited) wild-type KIT cytoplasmic domain.
- **1PKG** — active-state wild-type KIT.
- **4HVS** — KIT kinase domain with small-molecule inhibitor PLX647 (dual FMS/KIT inhibitor).
- **6XV9 / 6XVA** — KIT kinase domain with covalent-type inhibitors.
- **6GQJ** — KIT kinase domain with an AZD3229-analogue, a pan-mutant-KIT inhibitor developed against resistance mutations.
- **8PQA / 8PQB / 8PQD** — KIT (and PDGFRA) kinase domain in complex with avapritinib derivatives — directly relevant to the mutant-selective inhibitor class.

These structures let you visually/qualitatively check whether a predicted mutant-selective or wild-type-potent compound occupies a binding site consistent with known resistance-mutation geometry, without requiring docking software.

### 4.3 Literature-derived selectivity anchors

A small number of published, directly comparable wild-type-vs-D816V potency measurements exist and should be used as **validation anchors**, not training data, given how few such paired points exist:

- Dasatinib: reported IC50 of approximately 37 nM against D816V-mutant KIT vs. approximately 79 nM against wild-type KIT — i.e., comparable potency against both, unlike most KIT inhibitors.
- Imatinib: well-established clinically to lose efficacy against D816V-mutant KIT relative to wild-type, which is the reason mastocytosis patients do not respond to imatinib the way GIST patients (who typically have different, imatinib-sensitive KIT mutations) do.

These two data points are your ground truth for sanity-checking whether the model's selectivity signal is biologically plausible, not a training set.

### 4.4 Data quality issues to expect and handle explicitly

- Mixed assay types (IC50 vs Ki vs Kd) that are not directly comparable without care — document which are combined and how.
- Multiple measurements for the same compound across different papers/assays — needs a deduplication/aggregation strategy (e.g., median of log-transformed values).
- Censored values (e.g., ">10000 nM") — decide explicitly whether to drop, cap, or flag these rather than silently mishandling them.
- Class imbalance between wild-type and D816V-specific assay records (far more compounds tested against wild-type only).

## 5. Methodology

### 5.1 Data pipeline

1. Retrieve raw bioactivity records via the ChEMBL API (`chembl_webresource_client` or direct REST calls).
2. Standardize potency values to a common scale (pIC50/pKi = -log10(molar concentration)) so regression targets are comparable and roughly normally distributed.
3. Deduplicate and aggregate repeated measurements per compound.
4. Filter out or clearly flag censored/qualified values.
5. Canonicalize SMILES (via RDKit) to avoid representation duplicates of the same molecule.

### 5.2 Featurization

Two representations, compared against each other:

- **Baseline**: ECFP (Morgan) fingerprints via RDKit — fast, interpretable-ish, standard cheminformatics baseline.
- **Pretrained chemical language model**: ChemBERTa embeddings (via HuggingFace) — tests whether a pretrained representation improves over hand-crafted fingerprints on this specific, fairly small dataset.

### 5.3 Modeling

- **Baseline model**: gradient-boosted trees (XGBoost) on fingerprint features — given the likely small-to-medium dataset size, tree-based models are a realistic, strong baseline and should not be skipped in favor of jumping straight to deep learning.
- **Embedding-based model**: a small MLP head on top of frozen ChemBERTa embeddings.
- Compare both directly rather than assuming the more sophisticated approach automatically wins — consistent with the dissertation's own finding that a simpler method (traditional augmentation) outperformed a more sophisticated one (GANs) on a small dataset. The same caution applies here.

### 5.4 Train/test splitting

- Use a **scaffold split** (grouping compounds by core molecular scaffold before splitting), not a random split. Random splits on cheminformatics data systematically overestimate generalization because structurally near-identical analogues end up in both train and test sets. This is a well-known pitfall worth explicitly avoiding and documenting.

### 5.5 Evaluation metrics

- RMSE and R² on held-out pIC50/pKi values.
- Spearman rank correlation as a secondary metric — for potency prediction, getting the *relative ranking* of compounds right often matters more in practice than exact value prediction.

### 5.6 Selectivity analysis (secondary task)

- For any compound with both wild-type and D816V bioactivity records, compute a selectivity ratio (wild-type potency / mutant potency, or vice versa).
- Check whether the primary potency model, applied separately to wild-type and mutant-labeled subsets, reproduces the known direction of the dasatinib and imatinib anchor points.
- Treat this as a **validation exercise on a handful of anchor compounds**, not a fully trained supervised classifier — the paired data volume is too small to support a standalone selectivity classifier with any real confidence, and the README/design doc should say so directly rather than overstate it.

### 5.7 Structural sanity check

- For the top few predicted mutant-selective and wild-type-potent compounds (or their closest structural analogues in the PDB set above), visually inspect binding pose relative to the D816V mutation site (activation loop region) using existing structures — confirming the prediction is at least structurally plausible, without running new docking simulations.

## 6. Repository Structure

kit-inhibitor-binding-prediction/
├── README.md
├── DESIGN.md
├── data/
│ ├── raw/ # unmodified ChEMBL pulls
│ └── processed/ # cleaned, standardized datasets
├── notebooks/
│ ├── 01_data_collection.ipynb
│ ├── 02_data_cleaning.ipynb
│ ├── 03_featurization.ipynb
│ ├── 04_model_training.ipynb
│ ├── 05_selectivity_analysis.ipynb
│ └── 06_structural_context.ipynb
├── src/
│ ├── data_utils.py
│ ├── featurization.py
│ ├── models.py
│ └── evaluation.py
├── results/
│ └── figures/
└── environment.yml

## 7. Tools & Environment

- Python 3.10+
- RDKit (molecular handling, fingerprints, canonicalization)
- `chembl_webresource_client` (data retrieval)
- HuggingFace `transformers` (ChemBERTa)
- scikit-learn, XGBoost
- pandas, NumPy
- matplotlib/seaborn for figures

## 8. Milestones (Weekend Scope)

**Day 1**

- Set up environment, pull and inspect raw ChEMBL data.
- Clean and standardize the dataset; document all filtering decisions.
- Build and evaluate the fingerprint + XGBoost baseline.

**Day 2**

- Generate ChemBERTa embeddings and train the comparison model.
- Run the scaffold-split evaluation and compare baseline vs. embedding model.
- Run the selectivity anchor-point analysis and the structural sanity check.
- Write up results and finalize README/design doc with actual numbers in place of placeholders.

## 9. Limitations & Ethical Considerations

- This is a hypothesis-generation portfolio project; no result here should be read as validated drug discovery insight without substantial further wet-lab and clinical work.
- The selectivity analysis is anchored on very few known data points and should be presented with that caveat prominently, not glossed over.
- i-MCAS is discussed only as disease-spectrum context, not as a directly modeled or treated condition in this project.
- Public bioactivity databases like ChEMBL have known biases (e.g., over-representation of compounds from active drug discovery programs, under-representation of negative/inactive results in some cases) that could skew what the model learns as "typical" KIT-binding chemistry.

## 10. Future Work

- Extend to antibody-based agents (e.g., barzolvolimab-like mechanisms) using protein-level language models (ESM-2) rather than small-molecule chemical models — a genuinely different modeling problem, not a simple extension.
- Add physics-based docking for a subset of top candidates to complement the qualitative structural check.
- Extend the same selectivity-prediction framing to other resistance-mutation-prone kinases (e.g., PDGFRA in GIST, which shares structural biology with KIT and appears in some of the same PDB entries).

## 11. References

- Hamilton et al. (2021), *Distinct small intestine mast cell histologic changes in patients with hereditary alpha-tryptasemia and mast cell activation syndrome* — dissertation's primary clinical data source, included here for continuity.
- Furitsu et al. (1993) — original identification of KIT D816V in mast cell leukemia.
- Published pharmacology comparing dasatinib and imatinib potency against wild-type vs. D816V-mutant KIT.
- Blueprint Medicines preclinical data on BLU-808 (wild-type-selective KIT inhibitor).
- Clinical trial literature on barzolvolimab in chronic urticaria (included for mechanistic context on wild-type KIT depletion as a therapeutic strategy).
