"""HEEDB waveform preprocessing used for ECG–text pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mne.filter import filter_data, notch_filter
from scipy.interpolate import interp1d
from scipy.io import loadmat
from torch.utils.data import Dataset


class HEEDBDataset(Dataset):
    """Read indexed MAT/HEA cases and their original report text.

    Invalid cases raise an error. Substituting another row would corrupt the
    record-to-embedding correspondence used by the paper.
    """

    window_size = 5000
    sampling_rate = 500.0
    leads = ("I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVF", "aVL", "aVR")

    def __init__(self, indexed_csv_path: str | Path) -> None:
        self.data = pd.read_csv(indexed_csv_path)
        required = {"deid_t_diagnosis_original", "full_ecg_path"}
        if not required.issubset(self.data):
            raise KeyError(f"Index CSV requires {sorted(required)}")
        if self.data[list(required)].isna().any().any():
            raise ValueError("Index CSV contains missing report text or waveform paths")
        if self.data.full_ecg_path.duplicated().any():
            raise ValueError("Index CSV contains duplicate waveform paths")

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _resample(signal: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
        if fs_in == fs_out:
            return signal
        old = np.linspace(0, 1, num=len(signal), endpoint=True)
        new = np.linspace(0, 1, num=int(len(signal) * fs_out / fs_in), endpoint=True)
        return interp1d(old, signal, kind="linear")(new)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.data.iloc[index]
        mat_path = Path(row.full_ecg_path)
        hea_path = mat_path.with_suffix(".hea")
        waveform = np.asarray(loadmat(mat_path)["val"], dtype=np.float64)
        sampling_rate = float(hea_path.read_text(encoding="utf-8").splitlines()[0].split()[2])
        if waveform.ndim != 2 or waveform.shape[0] != len(self.leads) or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid 12-lead waveform: {mat_path}")
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        processed = []
        for lead in waveform:
            lead = self._resample(lead, sampling_rate, self.sampling_rate)
            lead = notch_filter(lead, self.sampling_rate, 60, verbose="ERROR")
            lead = filter_data(lead, self.sampling_rate, 0.5, 50, verbose="ERROR")
            processed.append(lead)
        waveform = np.asarray(processed)
        if waveform.shape[1] > self.window_size:
            start = (waveform.shape[1] - self.window_size) // 2
            waveform = waveform[:, start : start + self.window_size]
        elif waveform.shape[1] < self.window_size:
            waveform = np.pad(waveform, ((0, 0), (0, self.window_size - waveform.shape[1])))
        return {"ecg": torch.as_tensor(waveform, dtype=torch.float32), "txt": row.deid_t_diagnosis_original}
