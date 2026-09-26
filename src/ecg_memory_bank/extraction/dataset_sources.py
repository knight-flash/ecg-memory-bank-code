"""Original HEEDB and MIMIC dataset readers retained for extraction parity."""

from __future__ import annotations

import ast
import os
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.interpolate import interp1d
from mne.filter import filter_data, notch_filter

class HEEDBDataset_train(Dataset):
    """Read indexed HEEDB waveforms, reports, and labels."""
    def __init__(self, indexed_csv_path, debug_limit=None):
        """Load indexed waveform and label metadata."""
        self.window_size = 5000
        self.fs = 500.0
        self.leads = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']

        print(f"正在从 {indexed_csv_path} 加载数据...")
        try:
            self.data = pd.read_csv(indexed_csv_path)
            self.data = self.data.dropna(subset=['deid_t_diagnosis_original', 'full_ecg_path', 'labels_current'])
        except FileNotFoundError:
            print(f"[错误] 索引文件未找到: {indexed_csv_path}")
            print("请确保已运行数据预处理流水线脚本以生成该文件。")
            raise

        if debug_limit is not None:
            print(f"[DEBUG 模式] 数据集被限制为前 {debug_limit} 个样本。")
            self.data = self.data.head(debug_limit)

    def __len__(self):
        return len(self.data)

    def normalization(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0: return ts
        t = len(ts) / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
        if fs_out == fs_in: return ts
        x_old = np.linspace(0, 1, num=len(ts), endpoint=True)
        x_new = np.linspace(0, 1, num=int(t * fs_out), endpoint=True)
        f = interp1d(x_old, ts, kind='linear')
        return f(x_new)

    def preprocess(self, arr, sample_rate):
        out = []
        for tmp in arr:
            if sample_rate != self.fs:
                tmp = self.resample_unequal(tmp, sample_rate, self.fs)
            tmp = notch_filter(tmp, self.fs, 60, verbose='ERROR')
            tmp = filter_data(tmp, self.fs, 0.5, 50, verbose='ERROR')
            out.append(tmp)
        out = np.array(out)
        n_length = out.shape[1]
        if n_length > self.window_size:
            i_start = (n_length - self.window_size) // 2
            i_end = i_start + self.window_size
            out = out[:, i_start:i_end]
        elif n_length < self.window_size:
            pad_len = np.zeros((len(self.leads), self.window_size - n_length))
            out = np.concatenate([out, pad_len], axis=1)
        return out

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        original_text = row['deid_t_diagnosis_original']

        try:
            current_labels = ast.literal_eval(row['labels_current']) if pd.notna(row['labels_current']) else []
        except (ValueError, SyntaxError):
            current_labels = []

        try:
            historical_labels = ast.literal_eval(row['labels_historical']) if pd.notna(row['labels_historical']) else []
        except (ValueError, SyntaxError):
            historical_labels = []

        mat_path = row['full_ecg_path']

        try:
            mat_data = loadmat(mat_path)
            ecg_data = mat_data['val']

            hea_path = mat_path.replace('.mat', '.hea')
            with open(hea_path, 'r') as hea_file:
                first_line = hea_file.readlines()[0].strip()
                sample_rate = int(first_line.split()[2])

            ecg_data = np.array(ecg_data, dtype=float)
            ecg_data = self.normalization(ecg_data)
            ecg_data = self.preprocess(ecg_data, sample_rate)
            ecg_data = torch.tensor(ecg_data, dtype=torch.float)

            return {
                'ecg': ecg_data,
                'txt': original_text,
                'labels': current_labels,
                'historical_labels': historical_labels,
                'hash_filename': row['HashFileName']
            }

        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))


class MIMIC_Feature_Dataset(Dataset):
    """Read MIMIC waveforms with HEEDB-compatible preprocessing."""
    def __init__(self, indexed_csv_path, mimic_raw_data_dir, debug_limit=None):
        self.mimic_raw_data_dir = mimic_raw_data_dir
        self.fs = 500.0
        self.window_size = 5000
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        print(f"正在从索引文件加载元数据: {indexed_csv_path}")
        try:
            self.data = pd.read_csv(indexed_csv_path)
            self.data.dropna(subset=['path', 'final_report'], inplace=True)
            print(f"元数据加载完毕，共找到 {len(self.data)} 条有效记录。")
        except FileNotFoundError:
            print(f"[错误] 索引文件未找到: {indexed_csv_path}")
            raise

        if debug_limit is not None:
            print(f"[调试模式] 数据集被限制为前 {debug_limit} 条记录。")
            self.data = self.data.head(debug_limit)

    def __len__(self):
        return len(self.data)

    def normalization(self, signal):
        """Apply global z-score normalization."""
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def resample_unequal(self, ts, fs_in, fs_out):
        """Resample a waveform to the model sampling rate."""
        t = ts.shape[1] / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
        if fs_out == fs_in:
            return ts
        x_old = np.linspace(0, 1, num=ts.shape[1])
        x_new = np.linspace(0, 1, num=int(t * fs_out))
        f = interp1d(x_old, ts, kind='linear', axis=1, fill_value="extrapolate")
        return f(x_new)

    def preprocess(self, arr, sample_rate):
        """Filter and align a 12-lead waveform to the model input length."""
        if sample_rate != self.fs:
            arr = self.resample_unequal(arr, sample_rate, self.fs)

        arr = arr.astype(np.float64)

        filtered_data = []
        for i in range(arr.shape[0]):
            tmp = notch_filter(arr[i, :], self.fs, 60, verbose='ERROR')
            tmp = filter_data(tmp, self.fs, 0.5, 50, verbose='ERROR')
            filtered_data.append(tmp)
        out = np.array(filtered_data)

        n_length = out.shape[1]
        if n_length > self.window_size:
            i_start = (n_length - self.window_size) // 2
            out = out[:, i_start:i_start + self.window_size]
        elif n_length < self.window_size:
            pad_len = np.zeros((out.shape[0], self.window_size - n_length))
            out = np.concatenate([out, pad_len], axis=1)

        return out.astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:
            row = self.data.iloc[idx]
            report_text = row['final_report']
            relative_path = row['path']

            full_ecg_path = os.path.join(self.mimic_raw_data_dir, relative_path)
            ecg_data_raw, meta = wfdb.rdsamp(full_ecg_path)

            ecg_data_raw = ecg_data_raw.T
            original_fs = meta['fs']

            ordered_ecg = np.zeros((len(self.leads), ecg_data_raw.shape[1]))
            for i, lead_name in enumerate(self.leads):
                try:
                    j = meta['sig_name'].index(lead_name)
                    ordered_ecg[i, :] = ecg_data_raw[j, :]
                except ValueError:
                    continue

            normalized_ecg = self.normalization(ordered_ecg)

            processed_ecg = self.preprocess(normalized_ecg, original_fs)

            ecg_tensor = torch.tensor(processed_ecg, dtype=torch.float)

            return {'ecg': ecg_tensor, 'txt': report_text}

        except Exception as e:
            print(f"警告: 处理索引 {idx} (文件: {self.data.iloc[idx]['path']}) 时出错: {e}。将尝试下一个样本。")
            return self.__getitem__((idx + 1) % len(self))
