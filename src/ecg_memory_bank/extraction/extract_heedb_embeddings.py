#!/usr/bin/env python3
"""Distributed HEEDB extraction.

Purpose
-------
Reproduce the original multi-GPU generation of HEEDB ECG and text embeddings.

Method
------
Load the configured full ECGCLIP checkpoint, distribute indexed records,
checkpoint batches, then merge per-rank vectors and metadata.

Inputs
------
External HEEDB indexed CSV with paths, reports, labels and IDs; external model
YAML, checkpoint, and MedCPT files.

Outputs
-------
External HEEDB ECG/text .pt tensors, metadata CSV, progress records, and
extraction statistics.

Paper parameters
----------------
The historical script defaults to six GPU workers and batch size 256; the
wrapper supplies required paths and keeps output outside this repository.

Shapes and meaning
------------------
ECG batches are [B,12,5000], output ECG/text embeddings [N,256], and metadata
row i is intended to describe vector row i.
"""


import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import sys
import argparse
import yaml
from collections import defaultdict
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import gc
import json
import shutil
from datetime import datetime

os.environ["NCCL_TIMEOUT"] = "1800"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pretraining')))


try:
    from dataset_sources import HEEDBDataset_train
    import utils_builder
except ImportError as e:
    print(f"Error: could not import custom modules: {e}")
    print("Ensure that the required modules are available in the Python environment.")
    sys.exit(1)

def custom_collate_for_test(batch):
    """Collate ECG tensors and variable-length metadata fields."""
    ecgs = [item['ecg'] for item in batch]
    txts = [item['txt'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    historical_labels_list = [item['historical_labels'] for item in batch]
    hash_filenames = [item['hash_filename'] for item in batch]

    ecgs_collated = torch.stack(ecgs, 0)

    return {
        'ecg': ecgs_collated,
        'txt': txts,
        'labels': labels_list,
        'historical_labels': historical_labels_list,
        'hash_filenames': hash_filenames
    }

def load_pretrained_model(config_path, local_rank):
    """Load the configured ECGCLIP checkpoint for distributed extraction."""
    print(f"[Rank {local_rank}] Loading the pretrained model...")

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    ckpt_path = config['network']['ckpt_path']

    model = utils_builder.ECGCLIP(config['network'])

    print(f"[Rank {local_rank}] Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=f'cuda:{local_rank}')

    if config['network']['model_name'] == 'ECGCLIP':
        checkpoint = {key.replace('module.', ''): value for key, value in checkpoint['model_state_dict'].items()}
    elif config['network']['model_name'] == 'MERLCLIP':
        checkpoint = {
            k: v for k, v in checkpoint.items()
        }

    model.load_state_dict(checkpoint, strict=True)
    model = model.to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    print(f"[Rank {local_rank}] Pretrained model loaded successfully.")
    return model, config

class ProgressManager:
    """Manage per-worker extraction progress and checkpoints."""

    def __init__(self, local_rank, output_dir):
        self.local_rank = local_rank
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, f'progress_rank_{local_rank}.json')
        self.checkpoint_dir = os.path.join(output_dir, f'checkpoints_rank_{local_rank}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_progress(self, processed_batches, total_samples, current_time):
        """Save the extraction progress marker."""
        progress_data = {
            'local_rank': self.local_rank,
            'processed_batches': processed_batches,
            'total_samples': total_samples,
            'last_update': current_time,
            'status': 'processing'
        }

        print(f"[Rank {self.local_rank}] Saving progress:")
        print(f"[Rank {self.local_rank}]    - Processed batches: {processed_batches}")
        print(f"[Rank {self.local_rank}]    - Total samples: {total_samples}")
        print(f"[Rank {self.local_rank}]    - Progress file: {self.progress_file}")

        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            print(f"[Rank {self.local_rank}] Progress saved successfully.")
        except Exception as e:
            print(f"[Rank {self.local_rank}] Failed to save progress: {e}")

    def load_progress(self):
        """Restore the extraction progress marker."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def save_checkpoint(self, batch_idx, ecg_embs, text_embs, texts, labels, historical_labels, hash_filenames):
        """Save a batch of extracted embeddings."""
        checkpoint_data = {
            'batch_idx': batch_idx,
            'ecg_embeddings': ecg_embs,
            'text_embeddings': text_embs,
            'texts': texts,
            'labels': labels,
            'historical_labels': historical_labels,
            'hash_filenames': hash_filenames,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_batch_{batch_idx:06d}.pt')

        print(f"[Rank {self.local_rank}] Saving checkpoint:")
        print(f"[Rank {self.local_rank}]    - Batch index: {batch_idx}")
        print(f"[Rank {self.local_rank}]    - ECG embedding shape: {ecg_embs.shape}")
        print(f"[Rank {self.local_rank}]    - Text embedding shape: {text_embs.shape}")
        print(f"[Rank {self.local_rank}]    - Text count: {len(texts)}")
        print(f"[Rank {self.local_rank}]    - Label count: {len(labels)}")
        print(f"[Rank {self.local_rank}]    - Historical-label count: {len(historical_labels)}")
        print(f"[Rank {self.local_rank}]    - Filename count: {len(hash_filenames)}")
        print(f"[Rank {self.local_rank}]    - Save path: {checkpoint_file}")

        start_time = time.time()
        torch.save(checkpoint_data, checkpoint_file)
        save_time = time.time() - start_time

        if os.path.exists(checkpoint_file):
            file_size = os.path.getsize(checkpoint_file) / (1024 * 1024)  # MB
            print(f"[Rank {self.local_rank}] Checkpoint saved successfully.")
            print(f"[Rank {self.local_rank}]    - Save time: {save_time:.2f}s")
            print(f"[Rank {self.local_rank}]    - File size: {file_size:.2f} MB")
        else:
            print(f"[Rank {self.local_rank}] Failed to save checkpoint.")

        return checkpoint_file

    def mark_completed(self):
        """Mark one worker's extraction as complete."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            progress_data['status'] = 'completed'
            progress_data['completion_time'] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)

def sync_and_check(local_rank, world_size, stage_name=""):
    """Synchronize workers before merging extraction results."""
    try:
        dist.barrier()
        print(f"[Rank {local_rank}] {stage_name}: synchronization complete.")
        return True
    except Exception as e:
        print(f"[Rank {local_rank}] {stage_name}: synchronization failed: {e}")
        return False

def main_distributed(args):
    """Extract HEEDB embeddings across distributed workers."""
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=1800))

    print(f"[Rank {local_rank}] Starting HEEDB embedding extraction with the pretrained model.")
    print(f"[Rank {local_rank}] Device: {device}")
    print(f"[Rank {local_rank}] Total processes: {dist.get_world_size()}")

    world_size = dist.get_world_size()

    model, config = load_pretrained_model(args.config_path, local_rank)

    print(f"[Rank {local_rank}] Loading dataset from {args.main_csv_path}...")
    dataset = HEEDBDataset_train(indexed_csv_path=args.main_csv_path)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_for_test,
        sampler=sampler
    )

    print(f"[Rank {local_rank}] Dataset loaded: {len(data_loader.dataset)} local records.")
    total_batches = len(data_loader)

    progress_manager = ProgressManager(local_rank, args.output_dir)

    previous_progress = progress_manager.load_progress()
    start_batch = 0

    if previous_progress and previous_progress.get('status') == 'processing':
        start_batch = previous_progress.get('processed_batches', 0)
        print(f"[Rank {local_rank}] Resuming an incomplete run from batch {start_batch}.")
    elif previous_progress and previous_progress.get('status') == 'completed':
        print(f"[Rank {local_rank}] Extraction already complete; skipping processing.")
        return True

    print(f"[Rank {local_rank}] --- Extracting embeddings from batch {start_batch} ---")

    processed_samples = 0
    batch_buffer_ecg = []
    batch_buffer_text = []
    batch_buffer_texts = []
    batch_buffer_labels = []
    batch_buffer_historical_labels = []
    batch_buffer_hash_filenames = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"[Rank {local_rank}] Extraction progress", initial=start_batch)):
            if batch_idx < start_batch:
                continue

            if batch_idx % 10 == 0 or batch_idx >= start_batch:
                print(f"[Rank {local_rank}] Processing batch {batch_idx}; buffer size: {len(batch_buffer_ecg)}")

            try:
                ecg = batch['ecg'].to(torch.float32).to(device).contiguous()
                text = batch['txt']
                labels = batch['labels']
                historical_labels = batch['historical_labels']
                hash_filenames = batch['hash_filenames']

                report_tokenize_output = model.module._tokenize(text)
                input_ids = report_tokenize_output.input_ids.to(device).contiguous()
                attention_mask = report_tokenize_output.attention_mask.to(device).contiguous()

                ecg_embeddings = model.module.ext_ecg_emb(ecg)
                text_embeddings = model.module.get_text_emb(input_ids, attention_mask)

                ecg_embeddings = torch.nn.functional.normalize(ecg_embeddings, p=2, dim=-1)
                text_embeddings = model.module.proj_t(text_embeddings)
                text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)

                batch_buffer_ecg.append(ecg_embeddings.cpu())
                batch_buffer_text.append(text_embeddings.cpu())
                batch_buffer_texts.extend(text)
                batch_buffer_labels.extend(labels)
                batch_buffer_historical_labels.extend(historical_labels)
                batch_buffer_hash_filenames.extend(hash_filenames)
                processed_samples += len(text)

                save_condition = (batch_idx + 1) % args.save_interval == 0 or batch_idx == total_batches - 1
                if save_condition:
                    print(f"[Rank {local_rank}] Save condition reached:")
                    print(f"[Rank {local_rank}]    - Current batch: {batch_idx}")
                    print(f"[Rank {local_rank}]    - Next save batch: {((batch_idx // args.save_interval) + 1) * args.save_interval - 1}")
                    print(f"[Rank {local_rank}]    - (batch_idx + 1) % save_interval = {(batch_idx + 1) % args.save_interval}")
                    print(f"[Rank {local_rank}]    - Save interval: {args.save_interval}")
                    print(f"[Rank {local_rank}]    - Final batch: {batch_idx == total_batches - 1}")
                    print(f"[Rank {local_rank}]    - Buffered batches: {len(batch_buffer_ecg)}")
                    print(f"[Rank {local_rank}]    - Buffered samples: {len(batch_buffer_texts)}")

                    if batch_buffer_ecg:
                        print(f"[Rank {local_rank}] Merging buffered data...")
                        ecg_embs_combined = torch.cat(batch_buffer_ecg, dim=0)
                        text_embs_combined = torch.cat(batch_buffer_text, dim=0)
                        print(f"[Rank {local_rank}]    - Merged ECG shape: {ecg_embs_combined.shape}")
                        print(f"[Rank {local_rank}]    - Merged text shape: {text_embs_combined.shape}")

                        progress_manager.save_checkpoint(
                            batch_idx,
                            ecg_embs_combined,
                            text_embs_combined,
                            batch_buffer_texts,
                            batch_buffer_labels,
                            batch_buffer_historical_labels,
                            batch_buffer_hash_filenames
                        )

                        progress_manager.save_progress(batch_idx + 1, processed_samples, datetime.now().isoformat())

                        print(f"[Rank {local_rank}] Clearing buffers and releasing memory...")
                        batch_buffer_ecg = []
                        batch_buffer_text = []
                        batch_buffer_texts = []
                        batch_buffer_labels = []
                        batch_buffer_historical_labels = []
                        batch_buffer_hash_filenames = []

                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"[Rank {local_rank}] Buffers cleared and memory released.")
                    else:
                        print(f"[Rank {local_rank}] Buffer empty; skipping save.")

            except Exception as e:
                print(f"[Rank {local_rank}] Error processing batch {batch_idx}: {e}")
                progress_manager.save_progress(batch_idx, processed_samples, datetime.now().isoformat())
                continue

    if batch_buffer_ecg:
        print(f"[Rank {local_rank}] Saving the remaining buffered data:")
        print(f"[Rank {local_rank}]    - Remaining batches: {len(batch_buffer_ecg)}")
        print(f"[Rank {local_rank}]    - Remaining samples: {len(batch_buffer_texts)}")

        ecg_embs_combined = torch.cat(batch_buffer_ecg, dim=0)
        text_embs_combined = torch.cat(batch_buffer_text, dim=0)
        print(f"[Rank {local_rank}]    - Final ECG shape: {ecg_embs_combined.shape}")
        print(f"[Rank {local_rank}]    - Final text shape: {text_embs_combined.shape}")

        progress_manager.save_checkpoint(
            total_batches - 1,
            ecg_embs_combined,
            text_embs_combined,
            batch_buffer_texts,
            batch_buffer_labels,
            batch_buffer_historical_labels,
            batch_buffer_hash_filenames
        )
        print(f"[Rank {local_rank}] Final buffer saved.")
    else:
        print(f"[Rank {local_rank}] No buffered data remains to be saved.")

    progress_manager.mark_completed()
    print(f"[Rank {local_rank}] Local processing complete: {processed_samples} samples.")

    if not sync_and_check(local_rank, world_size, "Data processing complete"):
        return False

    if local_rank == 0:
        print(f"[Rank {local_rank}] --- Merging results from all ranks ---")

        merge_dir = os.path.join(args.output_dir, 'merged_results')
        os.makedirs(merge_dir, exist_ok=True)

        all_ecg_embeddings = []
        all_text_embeddings = []
        all_texts = []
        all_labels = []
        all_historical_labels = []
        all_hash_filenames = []

        for rank in range(world_size):
            rank_checkpoint_dir = os.path.join(args.output_dir, f'checkpoints_rank_{rank}')
            if os.path.exists(rank_checkpoint_dir):
                print(f"[Rank {local_rank}] Merging data from rank {rank}...")

                checkpoint_files = sorted([f for f in os.listdir(rank_checkpoint_dir) if f.endswith('.pt')])

                for checkpoint_file in checkpoint_files:
                    checkpoint_path = os.path.join(rank_checkpoint_dir, checkpoint_file)
                    try:
                        checkpoint_data = torch.load(checkpoint_path)

                        all_ecg_embeddings.append(checkpoint_data['ecg_embeddings'])
                        all_text_embeddings.append(checkpoint_data['text_embeddings'])
                        all_texts.extend(checkpoint_data['texts'])
                        all_labels.extend(checkpoint_data['labels'])
                        all_historical_labels.extend(checkpoint_data['historical_labels'])
                        all_hash_filenames.extend(checkpoint_data['hash_filenames'])

                    except Exception as e:
                        print(f"[Rank {local_rank}] Failed to load checkpoint {checkpoint_file}: {e}")
                        continue

        if all_ecg_embeddings:
            print(f"[Rank {local_rank}] Merging all embeddings...")
            all_ecg_embeddings = torch.cat(all_ecg_embeddings, dim=0)
            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

            print(f"[Rank {local_rank}] Merge complete - ECG: {all_ecg_embeddings.shape}, text: {all_text_embeddings.shape}")

            print(f"[Rank {local_rank}] --- Saving final results ---")

            try:
                ecg_output_path = os.path.join(args.output_dir, 'heedb_ecg_embeddings.pt')
                torch.save(all_ecg_embeddings, ecg_output_path)
                print(f"[Rank {local_rank}] ECG embeddings saved to: {ecg_output_path}")

                text_output_path = os.path.join(args.output_dir, 'heedb_text_embeddings.pt')
                torch.save(all_text_embeddings, text_output_path)
                print(f"[Rank {local_rank}] Text embeddings saved to: {text_output_path}")

                text_df = pd.DataFrame({
                    'HashFileName': all_hash_filenames,
                    'text': all_texts,
                    'labels': ['|'.join(labels) for labels in all_labels],
                    'historical_labels': ['|'.join(labels) for labels in all_historical_labels]
                })
                text_info_path = os.path.join(args.output_dir, 'heedb_texts.csv')
                text_df.to_csv(text_info_path, index=False)
                print(f"[Rank {local_rank}] Text information saved to: {text_info_path}")

                stats = {
                    'total_samples': len(all_texts),
                    'ecg_embedding_dim': all_ecg_embeddings.shape[1],
                    'text_embedding_dim': all_text_embeddings.shape[1],
                    'num_gpus': world_size,
                    'batch_size': args.batch_size,
                    'save_interval': args.save_interval,
                    'model_config': args.config_path,
                    'completion_time': datetime.now().isoformat()
                }

                stats_path = os.path.join(args.output_dir, 'extraction_stats.txt')
                with open(stats_path, 'w', encoding='utf-8') as f:
                    f.write("HEEDB Embedding Extraction Statistics (Robust Version)\n")
                    f.write("=" * 50 + "\n")
                    for key, value in stats.items():
                        f.write(f"{key}: {value}\n")
                print(f"[Rank {local_rank}] Statistics saved to: {stats_path}")

                if args.cleanup_checkpoints:
                    print(f"[Rank {local_rank}] Removing checkpoint files...")
                    for rank in range(world_size):
                        rank_checkpoint_dir = os.path.join(args.output_dir, f'checkpoints_rank_{rank}')
                        if os.path.exists(rank_checkpoint_dir):
                            shutil.rmtree(rank_checkpoint_dir)
                        progress_file = os.path.join(args.output_dir, f'progress_rank_{rank}.json')
                        if os.path.exists(progress_file):
                            os.remove(progress_file)

            except Exception as e:
                print(f"[Rank {local_rank}] Error saving data: {e}")
                return False

    if not sync_and_check(local_rank, world_size, "Task complete"):
        return False

    dist.destroy_process_group()
    print(f"[Rank {local_rank}] --- HEEDB embedding extraction complete ---")

    return True

def main(args):
    """Launch distributed HEEDB embedding extraction."""
    return main_distributed(args)

if __name__ == '__main__':
    from datetime import timedelta

    parser = argparse.ArgumentParser(description="Extract HEEDB ECG and text embeddings with the pretrained model")

    parser.add_argument('--config_path', type=str,
                        default='__ECG_CONFIG_REQUIRED__',
                        help='Path to the model configuration file')
    parser.add_argument('--main_csv_path', type=str,
                        default='__ECG_DATA_ROOT_REQUIRED__/heedb_embedding/heedb_with_multilabels_historical_updated.csv',
                        help='Path to the HEEDB indexed CSV with updated waveform paths')
    parser.add_argument('--output_dir', type=str,
                        default='__ECG_DATA_ROOT_REQUIRED__/heedb_embedding/robust',
                        help='Output directory for ECG embeddings and metadata')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of data-loader workers')
    parser.add_argument('--save_interval', type=int, default=6,
                        help='Number of batches between checkpoints')
    parser.add_argument('--cleanup_checkpoints', action='store_true',
                        help='Remove checkpoint files after completion')

    args = parser.parse_args()

    if not os.path.exists(args.main_csv_path):
        print(f"Error: input CSV does not exist: {args.main_csv_path}")
        sys.exit(1)

    if not os.path.exists(args.config_path):
        print(f"Error: configuration file does not exist: {args.config_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    success = main(args)

    if success:
        print("\nEmbedding extraction completed successfully.")
    else:
        print("\nEmbedding extraction failed.")
        sys.exit(1)
