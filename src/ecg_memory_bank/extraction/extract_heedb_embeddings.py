#!/usr/bin/env python3
"""Extract HEEDB ECGCLIP embeddings with the original distributed pipeline."""


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
    print(f"错误：无法导入自定义模块: {e}")
    print("请确保您的Python环境可以找到相关模块。")
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
    print(f"[Rank {local_rank}] 正在加载预训练模型...")

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    ckpt_path = config['network']['ckpt_path']

    model = utils_builder.ECGCLIP(config['network'])

    print(f"[Rank {local_rank}] 正在从 {ckpt_path} 加载权重...")
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

    print(f"[Rank {local_rank}] 预训练模型加载成功！")
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

        print(f"[Rank {self.local_rank}] 📝 保存进度信息:")
        print(f"[Rank {self.local_rank}]    - 已处理批次: {processed_batches}")
        print(f"[Rank {self.local_rank}]    - 总样本数: {total_samples}")
        print(f"[Rank {self.local_rank}]    - 进度文件: {self.progress_file}")

        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            print(f"[Rank {self.local_rank}] ✅ 进度保存成功")
        except Exception as e:
            print(f"[Rank {self.local_rank}] ❌ 进度保存失败: {e}")

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

        print(f"[Rank {self.local_rank}] 🔄 开始保存检查点:")
        print(f"[Rank {self.local_rank}]    - 批次索引: {batch_idx}")
        print(f"[Rank {self.local_rank}]    - ECG嵌入形状: {ecg_embs.shape}")
        print(f"[Rank {self.local_rank}]    - Text嵌入形状: {text_embs.shape}")
        print(f"[Rank {self.local_rank}]    - 文本数量: {len(texts)}")
        print(f"[Rank {self.local_rank}]    - 标签数量: {len(labels)}")
        print(f"[Rank {self.local_rank}]    - 历史标签数量: {len(historical_labels)}")
        print(f"[Rank {self.local_rank}]    - 文件名数量: {len(hash_filenames)}")
        print(f"[Rank {self.local_rank}]    - 保存路径: {checkpoint_file}")

        start_time = time.time()
        torch.save(checkpoint_data, checkpoint_file)
        save_time = time.time() - start_time

        if os.path.exists(checkpoint_file):
            file_size = os.path.getsize(checkpoint_file) / (1024 * 1024)  # MB
            print(f"[Rank {self.local_rank}] ✅ 检查点保存成功!")
            print(f"[Rank {self.local_rank}]    - 保存耗时: {save_time:.2f}秒")
            print(f"[Rank {self.local_rank}]    - 文件大小: {file_size:.2f}MB")
        else:
            print(f"[Rank {self.local_rank}] ❌ 检查点保存失败!")

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
        print(f"[Rank {local_rank}] {stage_name} 同步完成")
        return True
    except Exception as e:
        print(f"[Rank {local_rank}] {stage_name} 同步失败: {e}")
        return False

def main_distributed(args):
    """Extract HEEDB embeddings across distributed workers."""
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=1800))

    print(f"[Rank {local_rank}] 开始使用预训练模型执行 HEEDD 嵌入向量提取任务")
    print(f"[Rank {local_rank}] 使用设备: {device}")
    print(f"[Rank {local_rank}] 总进程数: {dist.get_world_size()}")

    world_size = dist.get_world_size()

    model, config = load_pretrained_model(args.config_path, local_rank)

    print(f"[Rank {local_rank}] 正在从 {args.main_csv_path} 加载数据集...")
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

    print(f"[Rank {local_rank}] 数据集加载完成，本地数据量: {len(data_loader.dataset)} 条记录。")
    total_batches = len(data_loader)

    progress_manager = ProgressManager(local_rank, args.output_dir)

    previous_progress = progress_manager.load_progress()
    start_batch = 0

    if previous_progress and previous_progress.get('status') == 'processing':
        start_batch = previous_progress.get('processed_batches', 0)
        print(f"[Rank {local_rank}] 检测到未完成的任务，从批次 {start_batch} 继续处理")
    elif previous_progress and previous_progress.get('status') == 'completed':
        print(f"[Rank {local_rank}] 任务已完成，跳过处理")
        return True

    print(f"[Rank {local_rank}] --- 开始分批提取嵌入向量 (从批次 {start_batch})... ---")

    processed_samples = 0
    batch_buffer_ecg = []
    batch_buffer_text = []
    batch_buffer_texts = []
    batch_buffer_labels = []
    batch_buffer_historical_labels = []
    batch_buffer_hash_filenames = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"[Rank {local_rank}] 提取进度", initial=start_batch)):
            if batch_idx < start_batch:
                continue

            if batch_idx % 10 == 0 or batch_idx >= start_batch:
                print(f"[Rank {local_rank}] 🔍 处理批次 {batch_idx}, 缓冲区大小: {len(batch_buffer_ecg)}")

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
                    print(f"[Rank {local_rank}] 🚀 触发保存条件:")
                    print(f"[Rank {local_rank}]    - 当前批次: {batch_idx}")
                    print(f"[Rank {local_rank}]    - 下一个保存批次: {((batch_idx // args.save_interval) + 1) * args.save_interval - 1}")
                    print(f"[Rank {local_rank}]    - (batch_idx + 1) % save_interval = {(batch_idx + 1) % args.save_interval}")
                    print(f"[Rank {local_rank}]    - 保存间隔: {args.save_interval}")
                    print(f"[Rank {local_rank}]    - 是否最后批次: {batch_idx == total_batches - 1}")
                    print(f"[Rank {local_rank}]    - 缓冲区批次数量: {len(batch_buffer_ecg)}")
                    print(f"[Rank {local_rank}]    - 缓冲区样本数量: {len(batch_buffer_texts)}")

                    if batch_buffer_ecg:
                        print(f"[Rank {local_rank}] 📦 开始合并缓冲区数据...")
                        ecg_embs_combined = torch.cat(batch_buffer_ecg, dim=0)
                        text_embs_combined = torch.cat(batch_buffer_text, dim=0)
                        print(f"[Rank {local_rank}]    - 合并后ECG形状: {ecg_embs_combined.shape}")
                        print(f"[Rank {local_rank}]    - 合并后Text形状: {text_embs_combined.shape}")

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

                        print(f"[Rank {local_rank}] 🧹 清空缓冲区和清理内存...")
                        batch_buffer_ecg = []
                        batch_buffer_text = []
                        batch_buffer_texts = []
                        batch_buffer_labels = []
                        batch_buffer_historical_labels = []
                        batch_buffer_hash_filenames = []

                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"[Rank {local_rank}] ✅ 缓冲区已清空，内存已清理")
                    else:
                        print(f"[Rank {local_rank}] ⚠️ 缓冲区为空，跳过保存")

            except Exception as e:
                print(f"[Rank {local_rank}] 处理批次 {batch_idx} 时出错: {e}")
                progress_manager.save_progress(batch_idx, processed_samples, datetime.now().isoformat())
                continue

    if batch_buffer_ecg:
        print(f"[Rank {local_rank}] 🏁 保存最后剩余的缓冲区数据:")
        print(f"[Rank {local_rank}]    - 剩余批次数量: {len(batch_buffer_ecg)}")
        print(f"[Rank {local_rank}]    - 剩余样本数量: {len(batch_buffer_texts)}")

        ecg_embs_combined = torch.cat(batch_buffer_ecg, dim=0)
        text_embs_combined = torch.cat(batch_buffer_text, dim=0)
        print(f"[Rank {local_rank}]    - 最终ECG形状: {ecg_embs_combined.shape}")
        print(f"[Rank {local_rank}]    - 最终Text形状: {text_embs_combined.shape}")

        progress_manager.save_checkpoint(
            total_batches - 1,
            ecg_embs_combined,
            text_embs_combined,
            batch_buffer_texts,
            batch_buffer_labels,
            batch_buffer_historical_labels,
            batch_buffer_hash_filenames
        )
        print(f"[Rank {local_rank}] ✅ 最终缓冲区保存完成")
    else:
        print(f"[Rank {local_rank}] ℹ️ 没有剩余的缓冲区数据需要保存")

    progress_manager.mark_completed()
    print(f"[Rank {local_rank}] 本地处理完成，共处理 {processed_samples} 个样本")

    if not sync_and_check(local_rank, world_size, "数据处理完成"):
        return False

    if local_rank == 0:
        print(f"[Rank {local_rank}] --- 开始合并所有rank的结果... ---")

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
                print(f"[Rank {local_rank}] 正在合并 Rank {rank} 的数据...")

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
                        print(f"[Rank {local_rank}] 加载检查点 {checkpoint_file} 失败: {e}")
                        continue

        if all_ecg_embeddings:
            print(f"[Rank {local_rank}] 正在合并所有嵌入向量...")
            all_ecg_embeddings = torch.cat(all_ecg_embeddings, dim=0)
            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

            print(f"[Rank {local_rank}] 合并完成 - ECG: {all_ecg_embeddings.shape}, Text: {all_text_embeddings.shape}")

            print(f"[Rank {local_rank}] --- 正在保存最终结果... ---")

            try:
                ecg_output_path = os.path.join(args.output_dir, 'heedb_ecg_embeddings.pt')
                torch.save(all_ecg_embeddings, ecg_output_path)
                print(f"[Rank {local_rank}] ECG嵌入已保存至: {ecg_output_path}")

                text_output_path = os.path.join(args.output_dir, 'heedb_text_embeddings.pt')
                torch.save(all_text_embeddings, text_output_path)
                print(f"[Rank {local_rank}] Text嵌入已保存至: {text_output_path}")

                text_df = pd.DataFrame({
                    'HashFileName': all_hash_filenames,
                    'text': all_texts,
                    'labels': ['|'.join(labels) for labels in all_labels],
                    'historical_labels': ['|'.join(labels) for labels in all_historical_labels]
                })
                text_info_path = os.path.join(args.output_dir, 'heedb_texts.csv')
                text_df.to_csv(text_info_path, index=False)
                print(f"[Rank {local_rank}] 文本信息已保存至: {text_info_path}")

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
                print(f"[Rank {local_rank}] 统计信息已保存至: {stats_path}")

                if args.cleanup_checkpoints:
                    print(f"[Rank {local_rank}] 清理检查点文件...")
                    for rank in range(world_size):
                        rank_checkpoint_dir = os.path.join(args.output_dir, f'checkpoints_rank_{rank}')
                        if os.path.exists(rank_checkpoint_dir):
                            shutil.rmtree(rank_checkpoint_dir)
                        progress_file = os.path.join(args.output_dir, f'progress_rank_{rank}.json')
                        if os.path.exists(progress_file):
                            os.remove(progress_file)

            except Exception as e:
                print(f"[Rank {local_rank}] 保存数据时出错: {e}")
                return False

    if not sync_and_check(local_rank, world_size, "任务完成"):
        return False

    dist.destroy_process_group()
    print(f"[Rank {local_rank}] --- HEEDB 嵌入向量提取任务执行完毕 ---")

    return True

def main(args):
    """Launch distributed HEEDB embedding extraction."""
    return main_distributed(args)

if __name__ == '__main__':
    from datetime import timedelta

    parser = argparse.ArgumentParser(description="使用预训练模型提取HEEDB ECG和Text嵌入向量的脚本 (增强版)")

    parser.add_argument('--config_path', type=str,
                        default='__ECG_CONFIG_REQUIRED__',
                        help='模型配置文件路径')
    parser.add_argument('--main_csv_path', type=str,
                        default='__ECG_DATA_ROOT_REQUIRED__/heedb_embedding/heedb_with_multilabels_historical_updated.csv',
                        help='包含更新后路径的HEEDB数据索引CSV文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='__ECG_DATA_ROOT_REQUIRED__/heedb_embedding/robust',
                        help='ECG嵌入和元数据文件的输出目录')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='推理时使用的批次大小')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='数据加载使用的工作线程数')
    parser.add_argument('--save_interval', type=int, default=6,
                        help='保存检查点的间隔批次数')
    parser.add_argument('--cleanup_checkpoints', action='store_true',
                        help='完成后清理检查点文件')

    args = parser.parse_args()

    if not os.path.exists(args.main_csv_path):
        print(f"错误: 输入CSV文件不存在: {args.main_csv_path}")
        sys.exit(1)

    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在: {args.config_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    success = main(args)

    if success:
        print("\n✅ 嵌入向量提取成功完成！")
    else:
        print("\n❌ 嵌入向量提取失败！")
        sys.exit(1)
