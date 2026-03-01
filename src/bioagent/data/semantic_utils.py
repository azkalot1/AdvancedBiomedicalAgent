#!/usr/bin/env python3
"""
Shared semantic embedding helpers.

This module centralizes embedding model constants and lazy loading so
search/ingestion code can use one implementation pattern.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import contextlib

EMBEDDING_MODEL_NAME = os.getenv("BIOAGENT_EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384
SMILES_EMBEDDING_MODEL_NAME = os.getenv("BIOAGENT_SMILES_MODEL_NAME", "DeepChem/ChemBERTa-5M-MLM")
SMILES_EMBEDDING_DIMENSION = 384
PROTEIN_EMBEDDING_MODEL_NAME = os.getenv("BIOAGENT_PROTEIN_MODEL_NAME", "facebook/esm2_t12_35M_UR50D")
PROTEIN_EMBEDDING_DIMENSION = 480

_MODEL: SentenceTransformer | None = None
_SMILES_TOKENIZER = None
_SMILES_MODEL = None
_PROTEIN_TOKENIZER = None
_PROTEIN_MODEL = None
_TORCH_DEVICE: torch.device | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_torch_device() -> torch.device:
    """
    Resolve runtime device for embedding models.

    Priority:
      1) BIOAGENT_DEVICE env var (cuda/cpu/cuda:0)
      2) CUDA if available
      3) CPU fallback
    """
    global _TORCH_DEVICE
    if _TORCH_DEVICE is not None:
        return _TORCH_DEVICE

    requested = (os.getenv("BIOAGENT_DEVICE") or "").strip().lower()
    if requested:
        try:
            device = torch.device(requested)
            if device.type == "cuda" and not torch.cuda.is_available():
                _TORCH_DEVICE = torch.device("cpu")
            else:
                _TORCH_DEVICE = device
            return _TORCH_DEVICE
        except Exception:
            pass

    _TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _TORCH_DEVICE


def normalize_semantic_text(value: str | None) -> str:
    """Normalize text for embedding lookups."""
    if not value:
        return ""
    return " ".join(value.lower().strip().split())


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model."""
    global _MODEL
    if _MODEL is None:
        device = get_torch_device()
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=str(device))
    return _MODEL


def get_smiles_embedding_model():
    """Lazily load and cache the ChemBERTa tokenizer/model."""
    global _SMILES_TOKENIZER, _SMILES_MODEL
    if _SMILES_TOKENIZER is None or _SMILES_MODEL is None:
        model_id = os.getenv("BIOAGENT_SMILES_MODEL_PATH") or SMILES_EMBEDDING_MODEL_NAME
        local_only = _env_bool("BIOAGENT_HF_LOCAL_ONLY", default=False)
        try:
            _SMILES_TOKENIZER = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=False,
                local_files_only=local_only,
            )
            _SMILES_MODEL = AutoModel.from_pretrained(
                model_id,
                local_files_only=local_only,
            )
            _SMILES_MODEL.to(get_torch_device())
            _SMILES_MODEL.eval()

            if torch.cuda.is_available():
                dummy = _SMILES_TOKENIZER(
                    ["CCO"],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                dummy = {k: v.cuda() for k, v in dummy.items()}
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = _SMILES_MODEL(**dummy)
                torch.cuda.synchronize()
                print("  ✅ SMILES model GPU warm-up complete")
        except Exception as e:
            raise RuntimeError(
                "Failed to load SMILES embedding model. "
                "Set BIOAGENT_SMILES_MODEL_PATH to a local model directory "
                "or pre-download DeepChem/ChemBERTa-77M-MLM. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    return _SMILES_TOKENIZER, _SMILES_MODEL


def _normalize_protein_sequence_esm(value: str | None) -> str:
    # ESM expects a plain AA string, typically uppercase, no spaces
    seq = re.sub(r"\s+", "", (value or "").upper())
    seq = re.sub(r"[UZOB]", "X", seq)
    seq = re.sub(r"[^A-Z]", "", seq)
    return seq

def get_protein_embedding_model():
    """Lazily load and cache ESM2 tokenizer/model."""
    global _PROTEIN_TOKENIZER, _PROTEIN_MODEL
    if _PROTEIN_TOKENIZER is None or _PROTEIN_MODEL is None:
        model_id = os.getenv("BIOAGENT_PROTEIN_MODEL_PATH") or PROTEIN_EMBEDDING_MODEL_NAME
        local_only = _env_bool("BIOAGENT_HF_LOCAL_ONLY", default=False)
        try:
            _PROTEIN_TOKENIZER = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                local_files_only=local_only,
            )
            _PROTEIN_MODEL = AutoModel.from_pretrained(
                model_id,
                local_files_only=local_only,
            )
            _PROTEIN_MODEL.to(get_torch_device())
            _PROTEIN_MODEL.eval()

            # Warm up CUDA kernel cache so first real batch
            # doesn't pay JIT compilation cost.
            device = get_torch_device()
            use_cuda = getattr(device, "type", str(device)).startswith("cuda")
            if use_cuda:
                dummy_seq = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"],
                dummy_tokens = _PROTEIN_TOKENIZER(
                    dummy_seq,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                dummy_tokens = {k: v.to(device) for k, v in dummy_tokens.items()}
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = _PROTEIN_MODEL(**dummy_tokens)
                torch.cuda.synchronize()
                print("  ✅ Protein model GPU warm-up complete")

        except Exception as e:
            raise RuntimeError(
                "Failed to load protein embedding model/tokenizer. "
                "Try installing tokenizers/sentencepiece if tokenizer conversion fails. "
                "Set BIOAGENT_PROTEIN_MODEL_PATH to a local directory and "
                "BIOAGENT_HF_LOCAL_ONLY=true for offline mode. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    return _PROTEIN_TOKENIZER, _PROTEIN_MODEL


def encode_texts(values: Iterable[str]) -> list[list[float]]:
    """Encode text values into vector embeddings."""
    model = get_embedding_model()
    encoded = model.encode(list(values), show_progress_bar=False)
    return [row.tolist() for row in encoded]


def encode_query_vector(value: str) -> list[float]:
    """Encode a single query string as an embedding vector."""
    model = get_embedding_model()
    return model.encode(normalize_semantic_text(value), show_progress_bar=False).tolist()


def encode_smiles_texts(
    values: Iterable[str],
    batch_size: int = 64,
    normalize: bool = True,
    tokenizer=None,
    model=None,
) -> list[list[float]]:
    """
    Encode SMILES strings with a ChemBERTa-style encoder.

    Returns mean-pooled last_hidden_state vectors (mask-aware).
    If normalize=True, returns L2-normalized vectors (good for cosine similarity search).

    GPU optimizations applied:
      - max_length=128 cap (SMILES rarely exceed this; avoids 512-token padding waste)
      - Length-sorted batching to minimize intra-batch padding
      - torch.inference_mode() instead of no_grad()
      - torch.cuda.amp.autocast() for fp16 inference on GPU
      - Non-blocking GPU transfers with pinned memory
      - Original order restored after sorted inference
    """
    if tokenizer is None or model is None:
        tokenizer = _SMILES_TOKENIZER
        model = _SMILES_MODEL
    if tokenizer is None or model is None:
        raise RuntimeError(
            "SMILES embedding model is not initialized. "
            "Call get_smiles_embedding_model() once before encode_smiles_texts()."
        )

    smiles_values = [str(v or "").strip() for v in values]
    if not smiles_values:
        return []

    device = get_torch_device()
    use_cuda = device.type == "cuda" if hasattr(device, "type") else str(device).startswith("cuda")

    # Sort by token-length proxy (string length) to minimize padding waste per batch.
    # We restore original order at the end.
    indexed = sorted(enumerate(smiles_values), key=lambda x: len(x[1]))
    sorted_indices, sorted_smiles = zip(*indexed)
    sorted_smiles = list(sorted_smiles)

    embeddings_sorted: list[list[float]] = []

    # Use autocast for fp16 on GPU — roughly 2x throughput on modern GPUs.
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if use_cuda
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        for i in range(0, len(sorted_smiles), batch_size):
            batch = sorted_smiles[i : i + batch_size]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,         # ChemBERTa default is 512; SMILES are short.
                return_tensors="pt",    # Avoids 4x wasted compute on padding tokens.
            )

            # Non-blocking transfer to GPU with pinned memory awareness.
            tokens = {
                k: v.to(device, non_blocking=True)
                for k, v in tokens.items()
            }

            outputs = model(**tokens)
            hidden = outputs.last_hidden_state          # [B, T, H]

            # Mask-aware mean pooling.
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)                             # [B, H]
            counts = mask.sum(dim=1).clamp(min=1)                           # [B, 1]
            pooled = summed / counts                                         # [B, H]

            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            # Cast back to float32 before moving to CPU (avoids numpy/psycopg2 issues
            # with float16 values when inserting into postgres).
            embeddings_sorted.extend(pooled.float().cpu().tolist())

    # Restore original insertion order.
    result: list[list[float]] = [None] * len(smiles_values)  # type: ignore[list-item]
    for original_idx, emb in zip(sorted_indices, embeddings_sorted):
        result[original_idx] = emb

    return result


def encode_smiles_query_vector(smiles: str, normalize: bool = True) -> list[float]:
    """Encode one SMILES string (same settings as batch encoder)."""
    tokenizer, model = get_smiles_embedding_model()
    vectors = encode_smiles_texts(
        [smiles],
        batch_size=1,
        normalize=normalize,
        tokenizer=tokenizer,
        model=model,
    )
    return vectors[0] if vectors else []


def encode_protein_sequences(
    values: Iterable[str],
    batch_size: int = 16,
    normalize: bool = True,
    max_length: int | None = None,
) -> list[list[float]]:
    """
    Encode protein sequences using the currently configured protein embedding model.

    GPU optimizations applied:
      - Length-sorted batching to minimize intra-batch padding (critical for proteins
        since sequence lengths vary from ~10 to 1000+ AAs)
      - torch.inference_mode() instead of no_grad()
      - torch.cuda.amp.autocast(fp16) for ~2x throughput on Tensor Core GPUs
      - non_blocking=True GPU transfers
      - .float() cast before CPU transfer to avoid psycopg2 fp16 serialization errors
      - Original order restored after sorted inference
      - Zero-vector placeholders preserved for empty/invalid sequences
    """
    tokenizer, model = get_protein_embedding_model()

    normalizer = globals().get(
        "_normalize_protein_sequence_esm", _normalize_protein_sequence_esm
    )

    if max_length is None:
        max_length = 1024

    original_seqs = [normalizer(v) for v in values]

    # Separate valid sequences from empty ones; we'll fill zeros for empties at the end.
    valid_positions = [idx for idx, s in enumerate(original_seqs) if s]
    if not valid_positions:
        return [[0.0] * PROTEIN_EMBEDDING_DIMENSION for _ in original_seqs]

    seqs = [original_seqs[idx] for idx in valid_positions]

    device = get_torch_device()
    use_cuda = getattr(device, "type", str(device)).startswith("cuda")

    # Sort valid sequences by length to minimize padding waste per batch.
    # Protein sequences vary enormously in length (10 -> 1000+ tokens),
    # so this has a much bigger impact than for SMILES.
    sort_order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    sorted_seqs = [seqs[i] for i in sort_order]

    # Map sorted position -> original valid_positions index for order restoration.
    unsort_order = [0] * len(sort_order)
    for sorted_idx, original_idx in enumerate(sort_order):
        unsort_order[original_idx] = sorted_idx

    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if use_cuda
        else contextlib.nullcontext()
    )

    embeddings_sorted: list[list[float]] = []

    with torch.inference_mode(), autocast_ctx:
        for i in range(0, len(sorted_seqs), batch_size):
            batch = sorted_seqs[i : i + batch_size]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Non-blocking H2D transfer — overlaps with CPU work on the CUDA stream.
            tokens = {
                k: v.to(device, non_blocking=True)
                for k, v in tokens.items()
            }

            outputs = model(**tokens)
            hidden = outputs.last_hidden_state  # [B, T, H]

            mask = tokens.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
                summed = (hidden * mask).sum(dim=1)          # [B, H]
                counts = mask.sum(dim=1).clamp(min=1)        # [B, 1]
                pooled = summed / counts                     # [B, H]

            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            # Cast to float32 before CPU transfer — psycopg2 cannot serialize fp16.
            embeddings_sorted.extend(pooled.float().cpu().tolist())

    # Unsort: restore embeddings to the order of valid_positions.
    embeddings_valid: list[list[float]] = [None] * len(seqs)  # type: ignore[list-item]
    for sorted_idx, emb in enumerate(embeddings_sorted):
        original_valid_idx = sort_order[sorted_idx]
        embeddings_valid[original_valid_idx] = emb

    # Re-insert zero-vector placeholders for empty/invalid sequences,
    # preserving 1:1 cardinality with the original input list.
    full_embeddings: list[list[float]] = [
        [0.0] * PROTEIN_EMBEDDING_DIMENSION for _ in original_seqs
    ]
    for emb_idx, original_idx in enumerate(valid_positions):
        full_embeddings[original_idx] = embeddings_valid[emb_idx]

    return full_embeddings

def encode_protein_query_vector(sequence: str, normalize: bool = True) -> list[float]:
    """Encode one protein sequence (same settings as batch encoder)."""
    vectors = encode_protein_sequences([sequence], batch_size=1, normalize=normalize)
    return vectors[0] if vectors else []