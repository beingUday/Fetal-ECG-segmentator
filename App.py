import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import (
    find_peaks,
    butter,
    filtfilt,
    medfilt,
    detrend,
    iirnotch,
    savgol_filter,
)
import pywt
import os

# ==========================================
# 1. CONFIGURATION (UPDATED FOR TS MODEL)
# ==========================================


class PipelineConfig:
    FS = 1000
    POWERLINE_FREQ = 50
    # TS Model specific settings
    MODEL_PATH = "ts.pth"
    # Standard TS parameters from your notebook
    CLEANING_PARAMS = {"win": 60, "thresh": 1.5}


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================


class ImprovedJudge(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            128,
            128,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout,
        )
        self.dense = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.up = nn.Upsample(scale_factor=4, mode="linear", align_corners=False)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dense(x)
        x = x.permute(0, 2, 1)
        x = self.up(x)
        return x.squeeze(1)


# ==========================================
# 3. CLEANING UTILS (TS ONLY)
# ==========================================


class SignalCleaner:
    @staticmethod
    def _fit_template(beat, template):
        num = np.dot(beat, template)
        den = np.dot(template, template)
        return num / (den + 1e-6)

    @staticmethod
    def apply_ts(data, win=60, thresh=1.5):
        cleaned = []
        fs = PipelineConfig.FS
        # Convert window from ms to samples
        win_samples = int((win / 1000) * fs)
        half_win = win_samples // 2

        for segment in data:
            seg_clean = np.zeros_like(segment)
            for ch in range(segment.shape[0]):
                sig = segment[ch]
                # Detect Mother's peaks (high amplitude)
                peak_thresh = np.percentile(np.abs(sig), 75) * thresh
                peaks, _ = find_peaks(sig, height=peak_thresh, distance=int(fs * 0.4))

                if len(peaks) > 1:
                    # Extract beats
                    beats = [
                        sig[max(0, p - half_win) : min(len(sig), p + half_win)]
                        for p in peaks
                    ]
                    beats = [b for b in beats if len(b) == win_samples]

                    if len(beats) >= 2:
                        # Create Template (Median Beat)
                        template = np.median(beats, axis=0)
                        clean_sig = sig.copy()

                        # Subtract Template
                        for p in peaks:
                            s, e = p - half_win, p + half_win
                            if 0 <= s and e <= len(sig):
                                curr = sig[s:e]
                                if len(curr) == len(template):
                                    scale = np.clip(
                                        SignalCleaner._fit_template(curr, template),
                                        0.5,
                                        2.0,
                                    )
                                    clean_sig[s:e] -= scale * template
                        seg_clean[ch] = clean_sig
                    else:
                        seg_clean[ch] = sig
                else:
                    seg_clean[ch] = sig
            cleaned.append(seg_clean)
        return np.array(cleaned, dtype=np.float32)


# ==========================================
# 4. PREPROCESSING
# ==========================================


def complete_wavelet_filter(signal):
    signal = np.nan_to_num(signal)
    b, a = butter(4, [0.5 / (0.5 * 1000), 100 / (0.5 * 1000)], btype="band")
    b_notch, a_notch = iirnotch(50, 30.0, 1000)

    # Filter
    for i in range(4):
        signal[:, i] = filtfilt(b_notch, a_notch, signal[:, i])
        signal[:, i] = filtfilt(b, a, signal[:, i])

    # Wavelet Denoising
    wavelet_result = np.zeros_like(signal)
    for i in range(4):
        coeffs = pywt.wavedec(signal[:, i], "sym10", level=6)
        coeffs[0] = np.zeros_like(coeffs[0])
        threshold = (np.median(np.abs(coeffs[-1])) / 0.6745) * np.sqrt(
            2 * np.log(len(signal[:, i]))
        )
        for j in range(1, len(coeffs)):
            coeffs[j] = pywt.threshold(coeffs[j], threshold, mode="soft")
        rec = pywt.waverec(coeffs, "sym10")
        wavelet_result[:, i] = (
            rec[: len(signal)]
            if len(rec) >= len(signal)
            else np.pad(rec, (0, len(signal) - len(rec)))
        )

    # Median Detrend
    baseline = np.zeros_like(wavelet_result)
    k_size = int(0.2 * 1000) | 1
    for i in range(4):
        baseline[:, i] = medfilt(wavelet_result[:, i], kernel_size=k_size)
    return wavelet_result - baseline


# ==========================================
# 5. STREAMLIT APP LOGIC
# ==========================================

st.title("Fetal ECG QRS Detection (TS Model) 🩺")
st.write("Upload a 4-channel AECG CSV file.")


# 1. Load TS Model
@st.cache_resource
def load_ts_model():
    if not os.path.exists(PipelineConfig.MODEL_PATH):
        return None
    model = ImprovedJudge()
    model.load_state_dict(
        torch.load(PipelineConfig.MODEL_PATH, map_location=torch.device("cpu"))
    )
    model.eval()
    return model


model = load_ts_model()

if model is None:
    st.error(f"❌ Model file '{PipelineConfig.MODEL_PATH}' not found.")
    st.info("Please download 'best_model_ts.pth' and place it in the same folder.")
else:
    st.success(f"✅ TS Model Loaded Successfully!")

# 2. File Upload
uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])

if uploaded_file is not None and model:
    try:
        df = pd.read_csv(uploaded_file)
        if not {"AECG1", "AECG2", "AECG3", "AECG4"}.issubset(df.columns):
            st.error("CSV must contain columns: AECG1, AECG2, AECG3, AECG4")
        else:
            raw_signal = df[["AECG1", "AECG2", "AECG3", "AECG4"]].values

            with st.spinner("Filtering & Extracting Mother's Signal..."):
                # 1. Filter
                filtered = complete_wavelet_filter(raw_signal)

                # 2. Segment & Normalize
                X_list = []
                norm_sig = filtered.copy()
                for ch in range(4):
                    d = norm_sig[:, ch]
                    norm_sig[:, ch] = (d - np.mean(d)) / (np.std(d) + 1e-6)

                for i in range(0, len(norm_sig) - 1000, 1000):
                    X_list.append(norm_sig[i : i + 1000].T)

                X_segments = np.array(X_list, dtype=np.float32)

                # 3. Apply TS Cleaning (Key Change)
                p = PipelineConfig.CLEANING_PARAMS
                X_cleaned = SignalCleaner.apply_ts(
                    X_segments, win=p["win"], thresh=p["thresh"]
                )

                # 4. Predict
                preds_list = []
                with torch.no_grad():
                    batch_tensor = torch.from_numpy(X_cleaned)
                    logits = model(batch_tensor)
                    probs = torch.sigmoid(logits).numpy()
                    preds_list = probs.flatten()

                # 5. Detect Peaks
                final_sig = preds_list
                final_sig = savgol_filter(final_sig, 21, 3)
                peaks, _ = find_peaks(final_sig, height=0.5, distance=300)

            # --- Visuals ---
            st.subheader("Analysis Results (TS Method)")
            st.metric("Fetal QRS Complexes Detected", len(peaks))

            # Display Graph
            display_len = min(4000, len(final_sig))
            chart_df = pd.DataFrame(
                {
                    "Filtered Signal (Ch1)": norm_sig[:display_len, 0],
                    "Fetal Probability": final_sig[:display_len],
                }
            )
            st.line_chart(chart_df)

            # Download
            st.download_button(
                "Download Peaks CSV",
                pd.DataFrame({"Peak_Index": peaks}).to_csv(index=False).encode("utf-8"),
                "ts_peaks.csv",
                "text/csv",
            )

    except Exception as e:
        st.error(f"Error: {e}")
