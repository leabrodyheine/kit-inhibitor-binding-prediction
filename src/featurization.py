"""Molecular featurization: ECFP (Morgan) fingerprints and frozen ChemBERTa embeddings (Phase 3)."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


def compute_ecfp_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Compute ECFP (Morgan) fingerprints for a list of SMILES strings.

    Returns an (n_compounds, n_bits) uint8 array. Raises ValueError naming the
    offending SMILES if any fail to parse — inputs are expected to already be
    RDKit-canonicalized (Phase 2), so a parse failure indicates a real problem
    rather than an expected edge case.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES at index {i}: {smiles!r}")
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr
    return fps


def compute_chemberta_embeddings(
    smiles_list,
    model_name=CHEMBERTA_MODEL_NAME,
    batch_size=32,
    device=None,
):
    """Compute frozen ChemBERTa embeddings for a list of SMILES strings.

    Each compound's embedding is the mean-pooled last hidden state over
    non-padding tokens. The model runs in eval mode with gradients disabled
    (Design Doc §5.2/§5.3: ChemBERTa is used frozen, only an MLP head is
    trained on top in Phase 4).

    Returns an (n_compounds, hidden_size) float32 array.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            output = model(**encoded)
            last_hidden = output.last_hidden_state  # (batch, seq_len, hidden)
            mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            embeddings.append(mean_pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)
