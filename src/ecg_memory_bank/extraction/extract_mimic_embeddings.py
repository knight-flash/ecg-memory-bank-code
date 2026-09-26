"""Extract MIMIC-IV-ECG embeddings with the original waveform pipeline."""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pretraining')))

try:
    from dataset_sources import MIMIC_Feature_Dataset
    import utils_builder
except ImportError:
    print("Unable to import dataset_sources or utils_builder from the code archive.")
    sys.exit(1)


def load_full_model(config, device):
    """Load the complete ECGCLIP model from its checkpoint."""
    model_name = config['network']['model_name']
    model_path = config['network']['ckpt_path']

    print(f"正在从 {model_path} 加载完整的 {model_name} 模型...")

    model = utils_builder.ECGCLIP(config['network'])

    checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    print("模型加载并准备就绪！")
    return model


def main(args):
    print("--- 开始执行嵌入向量提取任务 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = load_full_model(config, device)

    print(f"正在从 {args.main_csv_path} 加载MIMIC数据集...")

    MIMIC_RAW_DATA_DIR = args.mimic_raw_data_dir

    dataset = MIMIC_Feature_Dataset(
        indexed_csv_path=args.main_csv_path,
        mimic_raw_data_dir=MIMIC_RAW_DATA_DIR
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"数据集加载完成，共 {len(dataset)} 条记录。")

    print("\n--- 正在提取嵌入向量... ---")
    all_ecg_embs = []
    all_text_embs = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="提取进度"):
            ecg_signals = batch['ecg'].to(device)
            raw_texts = batch['txt']


            tokenized_output = model._tokenize(raw_texts)
            input_ids = tokenized_output.input_ids.to(device)
            attention_mask = tokenized_output.attention_mask.to(device)

            outputs = model(ecg_signals, input_ids, attention_mask, labels=None)

            proj_ecg_emb = outputs['proj_ecg_emb'][0]
            proj_text_emb = outputs['proj_text_emb'][0]

            all_ecg_embs.append(proj_ecg_emb.cpu())
            all_text_embs.append(proj_text_emb.cpu())


    print("\n--- 正在整合并保存提取结果... ---")

    os.makedirs(args.output_dir, exist_ok=True)

    ecg_embeddings_tensor = torch.cat(all_ecg_embs, dim=0)
    text_embeddings_tensor = torch.cat(all_text_embs, dim=0)

    ecg_output_path = os.path.join(args.output_dir, 'ecg_embeddings.pt')
    text_output_path = os.path.join(args.output_dir, 'text_embeddings.pt')

    torch.save(ecg_embeddings_tensor, ecg_output_path)
    torch.save(text_embeddings_tensor, text_output_path)

    print(f"ECG嵌入已保存至: {ecg_output_path} (形状: {ecg_embeddings_tensor.shape})")
    print(f"文本嵌入已保存至: {text_output_path} (形状: {text_embeddings_tensor.shape})")

    metadata_output_path = os.path.join(args.output_dir, 'metadata.csv')
    original_df = pd.read_csv(args.main_csv_path)
    original_df.to_csv(metadata_output_path, index=False)
    print(f"元数据已保存至: {metadata_output_path}")

    print("\n--- 任务执行完毕 ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ECG和文本嵌入向量提取脚本")

    parser.add_argument('--config_path', type=str, default='__ECG_CONFIG_REQUIRED__',
                        help='主配置文件路径')
    parser.add_argument('--main_csv_path', type=str, default='__ECG_DATA_ROOT_REQUIRED__/mimic/mimic_final_with_labels.csv',
                        help='包含所有数据的索引CSV文件路径')
    parser.add_argument('--mimic_raw_data_dir', type=str, required=True,
                        help='外部 MIMIC-IV-ECG 原始波形根目录')
    parser.add_argument('--output_dir', type=str, default='__ECG_DATA_ROOT_REQUIRED__/feature/mimic',
                        help='嵌入和元数据文件的输出目录')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='推理时使用的批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载使用的工作线程数')

    args = parser.parse_args()
    main(args)
