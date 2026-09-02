"""ESM-2 protein language model embeddings.

Uses HuggingFace ``transformers`` to load a pretrained ESM-2 checkpoint
(default: ``facebook/esm2_t6_8M_UR50D``, the smallest 8M-parameter model,
chosen for speed) and extracts the CLS-token embedding of the final hidden
layer as a fixed-length sequence representation.

GPU is used automatically when available (``torch.cuda.is_available()``);
otherwise the model runs on CPU.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"


def prepare_esm(
    model_name: str = DEFAULT_ESM_MODEL,
    device: torch.device | str | None = None,
) -> Tuple[AutoTokenizer, AutoModel]:
    """Load an ESM-2 tokenizer and model in evaluation mode.

    Parameters
    ----------
    model_name:
        HuggingFace Hub identifier of the ESM-2 checkpoint to load.
    device:
        Torch device to move the model to. If ``None``, uses CUDA when
        available, else CPU.

    Returns
    -------
    (tokenizer, model)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def get_esm_embeddings(
    sequences: Iterable[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract CLS-token ESM-2 embeddings for a list of peptide sequences.

    Parameters
    ----------
    sequences:
        Iterable of peptide sequence strings.
    tokenizer, model:
        As returned by :func:`prepare_esm`.
    batch_size:
        Number of sequences per forward pass. Lower this if you hit
        out-of-memory errors on long sequences or a small GPU.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, hidden_size)``, dtype ``float32``
        (``float32`` after moving off the accelerator). ``hidden_size`` is
        320 for ``esm2_t6_8M_UR50D``.
    """
    sequences = list(sequences)
    device = next(model.parameters()).device
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            # ESM-2 tokenizers expect whitespace-separated residues.
            toks = [" ".join(s.upper()) for s in batch]
            inputs = tokenizer(toks, return_tensors="pt", padding=True, truncation=True).to(device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls)
    return (
        np.vstack(embeddings)
        if embeddings
        else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    )


def esm_feature_names(n_features: int) -> List[str]:
    """Generate placeholder names ``esm_0 .. esm_{n-1}`` for the embedding matrix.

    Parameters
    ----------
    n_features:
        Number of embedding dimensions (i.e. ``X_esm.shape[1]``).

    Returns
    -------
    list of str
    """
    return [f"esm_{i}" for i in range(n_features)]
