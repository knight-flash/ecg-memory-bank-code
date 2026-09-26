"""Frozen ECG embedding extraction adapter.

Purpose
-------
Convert already preprocessed ECG arrays into aligned, L2-normalized record
embeddings.

Method
------
Load the full ECGCLIP checkpoint, strip optional DDP prefixes, run inference
in batches, and write an embedding tensor with a case-ID manifest.

Inputs
------
External waveform NumPy array, metadata CSV with unique record IDs,
checkpoint, model YAML, and MedCPT model directory.

Outputs
-------
External ecg_embeddings.npy, ecg_embeddings.pt, and embedding_record_ids.csv;
no generated data is stored in source.

Paper parameters
----------------
batch-size controls memory usage; device selects CPU/GPU. Preprocessing must
match the dataset-specific extraction protocol before this adapter.

Shapes and meaning
------------------
Waveforms are [N,12,T], model input batches [B,12,T], and output features
[N,256]. Manifest row i identifies embedding row i.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import utils_builder
from ecg_memory_bank.protocol import (
    external_output_dir,
    unwrap_ecgclip_checkpoint,
    validate_embedding_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="External full ECGCLIP checkpoint")
    parser.add_argument("--medcpt-model", type=Path, required=True)
    parser.add_argument("--waveforms-npy", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--record-id-col", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.config, args.checkpoint, args.waveforms_npy, args.metadata_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = external_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if args.record_id_col not in (reader.fieldnames or []):
            raise KeyError(f"Missing record ID column: {args.record_id_col}")
        ids = [row[args.record_id_col] for row in reader]
    waveforms = np.load(args.waveforms_npy, mmap_mode="r", allow_pickle=False)
    if waveforms.ndim != 3 or waveforms.shape[1] != 12:
        raise ValueError("Waveforms must have shape [records, 12, samples]")
    if len(waveforms) == 0:
        raise ValueError("Waveform array is empty")
    validate_embedding_alignment(ids, len(waveforms))
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not args.medcpt_model.is_dir():
        raise FileNotFoundError(args.medcpt_model)
    config["network"]["medcpt_model"] = str(args.medcpt_model)
    model = utils_builder.ECGCLIP(config["network"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a full ECGCLIP state dictionary or contain model_state_dict")
    model.load_state_dict(unwrap_ecgclip_checkpoint(checkpoint), strict=True)
    model = model.to(args.device).eval()
    mapped = None
    with torch.inference_mode():
        for start in range(0, len(waveforms), args.batch_size):
            batch = torch.from_numpy(np.asarray(waveforms[start : start + args.batch_size]).copy()).float().to(args.device)
            features = model.ext_ecg_emb(batch)
            if not torch.isfinite(features).all() or (torch.linalg.vector_norm(features, dim=1) == 0).any():
                raise ValueError(f"Invalid embedding values in rows {start}:{start + len(batch)}")
            normalized = F.normalize(features, p=2, dim=1).cpu().numpy().astype(np.float32, copy=False)
            if mapped is None:
                mapped = np.lib.format.open_memmap(
                    output_dir / "ecg_embeddings.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(waveforms), normalized.shape[1]),
                )
            mapped[start : start + len(normalized)] = normalized
    assert mapped is not None
    mapped.flush()
    validate_embedding_alignment(ids, len(mapped))
    torch.save(torch.from_numpy(mapped), output_dir / "ecg_embeddings.pt")
    with (output_dir / "embedding_record_ids.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([args.record_id_col])
        writer.writerows([[record_id] for record_id in ids])


if __name__ == "__main__":
    main()
