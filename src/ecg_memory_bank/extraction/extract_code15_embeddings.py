#!/usr/bin/env python3
"""Extract CODE-15 ECGCLIP embeddings with restartable batches."""


import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import argparse
import time
import gc
import json
import shutil
import yaml
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pretraining')))


try:
    import utils_builder
except ImportError as e:
    print(f"错误：无法导入自定义模块: {e}")
    sys.exit(1)

from torch.utils.data import Dataset
import numpy as np
import wfdb

class CustomCODE15Dataset(Dataset):
    """Read indexed CODE-15 waveforms and diagnostic labels."""
    def __init__(self, indexed_csv_path):
        super().__init__()
        self.df = pd.read_csv(indexed_csv_path)
        self.label_cols = ['chagas', '1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']

        for col in self.label_cols:
            if col not in self.df.columns:
                print(f"警告: 列 '{col}' 不在 CSV 中！")

    def len(self):
        return len(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ecg_path = row['full_ecg_path']
        try:
            if ecg_path.endswith('.hea') or ecg_path.endswith('.dat'):
                ecg_path = ecg_path[:-4]

            try:
                record = wfdb.rdrecord(ecg_path)
            except Exception as e_first:
                if '/data/' in ecg_path:
                    fallback_ecg_path = ecg_path.replace('/data/', '/holdout_data/')
                    try:
                        record = wfdb.rdrecord(fallback_ecg_path)
                    except Exception as e_second:
                        raise FileNotFoundError(f"'{ecg_path}' 和 '{fallback_ecg_path}' 均未找到。")
                else:
                    raise e_first

            ecg_data = record.p_signal
            ecg_data = np.nan_to_num(ecg_data)

            ecg_tensor = torch.from_numpy(ecg_data.copy()).float().transpose(0, 1).contiguous()

            target_length = 4096
            current_length = ecg_tensor.shape[1]
            if current_length > target_length:
                ecg_tensor = ecg_tensor[:, :target_length]
            elif current_length < target_length:
                padding = torch.zeros((12, target_length - current_length), dtype=torch.float32)
                ecg_tensor = torch.cat([ecg_tensor, padding], dim=1)
        except Exception as e:
            print(f"读取 ECG 失败 (第 {idx} 行): {e}")
            ecg_tensor = torch.zeros((12, 4096), dtype=torch.float32)

        labels = []
        for col in self.label_cols:
            val = row.get(col, 0)
            if isinstance(val, bool):
                labels.append(1.0 if val else 0.0)
            else:
                try:
                    labels.append(float(val))
                except:
                    labels.append(0.0)

        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return ecg_tensor, labels_tensor

def load_pretrained_model(config_path, device):
    print(f"正在加载预训练提取模型...")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    ckpt_path = config['network']['ckpt_path']

    model = utils_builder.ECGCLIP(config['network'])

    print(f"正在从 {ckpt_path} 加载权重...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if config['network']['model_name'] == 'ECGCLIP':
        checkpoint = {key.replace('module.', ''): value for key, value in checkpoint['model_state_dict'].items()}
    elif config['network']['model_name'] == 'MERLCLIP':
        checkpoint = {k: v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()

    print(f"预训练模型 (ECGCLIP) 加载成功！")
    return model, config

class ProgressManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, 'progress.json')
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_progress(self, processed_batches, total_samples, current_time):
        progress_data = {
            'processed_batches': processed_batches,
            'total_samples': total_samples,
            'last_update': current_time,
            'status': 'processing'
        }
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            print(f"✅ 进度保存成功 (已处理批次: {processed_batches})")
        except Exception as e:
            print(f"❌ 进度保存失败: {e}")

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def save_checkpoint(self, batch_idx, ecg_embs):
        checkpoint_data = {
            'batch_idx': batch_idx,
            'ecg_embeddings': ecg_embs,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_batch_{batch_idx:06d}.pt')
        print(f"🔄 开始保存检查点 (批次 {batch_idx}) ...")
        start_time = time.time()
        torch.save(checkpoint_data, checkpoint_file)
        save_time = time.time() - start_time

        if os.path.exists(checkpoint_file):
            file_size = os.path.getsize(checkpoint_file) / (1024 * 1024)
            print(f"✅ 检查点保存成功! 耗时: {save_time:.2f}s, 大小: {file_size:.2f}MB")
        return checkpoint_file

    def mark_completed(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            progress_data['status'] = 'completed'
            progress_data['completion_time'] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)

def main_single_gpu(args):
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"开始使用 ECGCLIP 执行 CODE-15 嵌入向量提取任务")
    print(f"使用设备: {device}")

    model, config = load_pretrained_model(args.config_path, device)

    print(f"正在从 {args.main_csv_path} 加载 CODE-15 自定义数据集...")
    dataset = CustomCODE15Dataset(indexed_csv_path=args.main_csv_path)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"数据集加载完成，数据量: {len(data_loader)} 个批次，共 {len(dataset)} 条记录。")
    total_batches = len(data_loader)

    progress_manager = ProgressManager(args.output_dir)
    previous_progress = progress_manager.load_progress()
    start_batch = 0

    if previous_progress and previous_progress.get('status') == 'processing':
        start_batch = previous_progress.get('processed_batches', 0)
        print(f"检测到未完成的任务，从批次 {start_batch} 继续处理")
    elif previous_progress and previous_progress.get('status') == 'completed':
        print(f"任务已完成，直接进入合并阶段")

    if start_batch < total_batches and (not previous_progress or previous_progress.get('status') != 'completed'):
        print(f"--- 开始分批提取嵌入向量 (从批次 {start_batch})... ---")

        processed_samples = 0
        batch_buffer_ecg = []
        batch_buffer_labels = []

        with torch.no_grad():
            for batch_idx, (ecg_signals, labels) in enumerate(tqdm(data_loader, desc=f"提取进度", initial=start_batch)):
                if batch_idx < start_batch:
                    continue

                try:
                    ecg = ecg_signals.to(torch.float32).to(device).contiguous()

                    ecg_features = model.ext_ecg_emb(ecg)
                    ecg_features = F.normalize(ecg_features, p=2, dim=-1)

                    batch_buffer_ecg.append(ecg_features.cpu())
                    batch_buffer_labels.append(labels)
                    processed_samples += ecg.size(0)

                    save_condition = (batch_idx + 1) % args.save_interval == 0 or batch_idx == total_batches - 1
                    if save_condition:
                        if batch_buffer_ecg:
                            ecg_embs_combined = torch.cat(batch_buffer_ecg, dim=0)
                            labels_combined = torch.cat(batch_buffer_labels, dim=0)

                            checkpoint_dict = {
                                'ecg_embeddings': ecg_embs_combined,
                                'labels': labels_combined
                            }

                            progress_manager.save_checkpoint(batch_idx, checkpoint_dict)
                            progress_manager.save_progress(batch_idx + 1, processed_samples, datetime.now().isoformat())

                            batch_buffer_ecg = []
                            batch_buffer_labels = []
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                except Exception as e:
                    print(f"处理批次 {batch_idx} 时出错: {e}")
                    progress_manager.save_progress(batch_idx, processed_samples, datetime.now().isoformat())
                    continue

        if batch_buffer_ecg:
            ecg_embs_combined = torch.cat(batch_buffer_ecg, dim=0)
            labels_combined = torch.cat(batch_buffer_labels, dim=0)
            checkpoint_dict = {
                'ecg_embeddings': ecg_embs_combined,
                'labels': labels_combined
            }
            progress_manager.save_checkpoint(total_batches - 1, checkpoint_dict)

        progress_manager.mark_completed()
        print(f"本地处理完成，共计由本进程处理 {processed_samples} 个样本。")

    print(f"--- 开始合并检查点结果... ---")
    all_ecg_embeddings = []
    all_labels = []
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')

    if os.path.exists(checkpoint_dir):
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        print(f"找到 {len(checkpoint_files)} 个检查点缓存块, 开始按序加载提取...")
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_data = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
                if isinstance(checkpoint_data['ecg_embeddings'], dict):
                    all_ecg_embeddings.append(checkpoint_data['ecg_embeddings']['ecg_embeddings'])
                    all_labels.append(checkpoint_data['ecg_embeddings']['labels'])
                else:
                    all_ecg_embeddings.append(checkpoint_data['ecg_embeddings'])
            except Exception as e:
                print(f"加载检查点 {checkpoint_file} 失败: {e}")

    if all_ecg_embeddings:
        all_ecg_embeddings = torch.cat(all_ecg_embeddings, dim=0)
        print(f"合并完成 - Feature Tensor: {all_ecg_embeddings.shape}")

        if all_labels:
            all_labels = torch.cat(all_labels, dim=0)
            print(f"合并完成 - Labels Tensor: {all_labels.shape}")

        try:
            ecg_output_path = os.path.join(args.output_dir, 'code15_ecgclip_embeddings.pt')
            torch.save(all_ecg_embeddings, ecg_output_path)

            if len(all_labels) > 0:
                labels_output_path = os.path.join(args.output_dir, 'code15_labels.pt')
                torch.save(all_labels, labels_output_path)
                print(f"7类目标标签已保存至: {labels_output_path}")

            print(f"特征已保存至: {ecg_output_path}")

            original_df = pd.read_csv(args.main_csv_path)
            if len(original_df) != all_ecg_embeddings.shape[0]:
                print(f"⚠️ 警告: 特征数量 ({all_ecg_embeddings.shape[0]}) 与元数据行数 ({len(original_df)}) 不一致！")
            else:
                print("✅ 特征总数与原始 CSV 行数完全一致。")

            metadata_output_path = os.path.join(args.output_dir, 'metadata_code15_ecgclip.csv')
            original_df.to_csv(metadata_output_path, index=False)

            if args.cleanup_checkpoints:
                shutil.rmtree(checkpoint_dir)
                os.remove(os.path.join(args.output_dir, 'progress.json'))

        except Exception as e:
            print(f"保存最终结果时发生错误: {e}")
            return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='__ECG_CONFIG_REQUIRED__',
                        help='模型配置文件路径')
    parser.add_argument('--main_csv_path', type=str,
                        default='__ECG_DATA_ROOT_REQUIRED__/feature/code15/code15_indexed_en.csv')
    parser.add_argument('--output_dir', type=str,
                        default='__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--cleanup_checkpoints', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=3)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    success = main_single_gpu(args)
    if success:
        print("\n✅ ECGCLIP (CODE-15) 特征提取闭环完成！")
    else:
        sys.exit(1)
