import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks, butter, filtfilt, medfilt, detrend, iirnotch, savgol_filter
from sklearn.decomposition import FastICA
import pywt
import os

# ==========================================
# 1. CONFIGURATION & CLASSES FROM NOTEBOOK
# ==========================================

class PipelineConfig:
    FS = 1000
    POWERLINE_FREQ = 50
    SEQ_LEN = 1000
    STRIDE = 1000 # Using non-overlapping stride for inference visualization simplicity
    DEVICE = torch.device("cpu") # Force CPU for the web app
    
    MODELS = {
        'ts':    {'path': 'best_model_ts.pth', 'params': {'win': 60, 'thresh': 1.5}, 'method': 'ts'},
        'ica':   {'path': 'best_model_ica.pth', 'params': {'comps': 4}, 'method': 'ica'},
        'combo': {'path': 'best_model_combo.pth', 'params': {'win': 60, 'thresh': 1.8, 'comps': 3}, 'method': 'combo'}
    }

class ImprovedJudge(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(4, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout))
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        self.dense = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.up = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)

    def forward(self, x):
        x = self.enc1(x); x = self.pool1(x); x = self.enc2(x); x = self.pool2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dense(x)
        x = x.permute(0, 2, 1); x = self.up(x)
        return x.squeeze(1)

class SignalCleaner:
    @staticmethod
    def _fit_template(beat: np.ndarray, template: np.ndarray) -> float:
        num = np.dot(beat, template)
        den = np.dot(template, template)
        return num / (den + 1e-6)

    @staticmethod
    def apply_ts(data: np.ndarray, win: int = 60, thresh: float = 1.5) -> np.ndarray:
        cleaned = []; fs = PipelineConfig.FS; win_samples = int((win/1000) * fs); half_win = win_samples // 2
        for segment in data:
            seg_clean = np.zeros_like(segment)
            for ch in range(segment.shape[0]):
                sig = segment[ch]; peak_thresh = np.percentile(np.abs(sig), 75) * thresh
                peaks, _ = find_peaks(sig, height=peak_thresh, distance=int(fs*0.4))
                if len(peaks) > 1:
                    beats = [sig[max(0, p-half_win):min(len(sig), p+half_win)] for p in peaks]
                    beats = [b for b in beats if len(b) == win_samples]
                    if len(beats) >= 2:
                        template = np.median(beats, axis=0); clean_sig = sig.copy()
                        for p in peaks:
                            s, e = p - half_win, p + half_win
                            if 0 <= s and e <= len(sig):
                                curr = sig[s:e]
                                if len(curr) == len(template):
                                    scale = np.clip(SignalCleaner._fit_template(curr, template), 0.5, 2.0)
                                    clean_sig[s:e] -= (scale * template)
                        seg_clean[ch] = clean_sig
                    else: seg_clean[ch] = sig
                else: seg_clean[ch] = sig
            cleaned.append(seg_clean)
        return np.array(cleaned, dtype=np.float32)

    @staticmethod
    def apply_ica(data: np.ndarray, comps: int = 4) -> np.ndarray:
        cleaned = []
        for segment in data:
            try:
                X = segment.T; 
                safe_comps = min(comps, X.shape[1])
                if np.linalg.matrix_rank(X) < safe_comps or np.any(np.std(X, axis=0) < 1e-6): 
                    cleaned.append(segment); continue
                ica = FastICA(n_components=safe_comps, random_state=42, whiten='unit-variance')
                S = ica.fit_transform(X)
                scores = [abs(np.corrcoef(S[:, i], X[:, 0])[0, 1]) for i in range(safe_comps)]
                S[:, np.argmax(scores)] = 0 
                clean = ica.inverse_transform(S)
                cleaned.append(clean.T)
            except Exception:
                cleaned.append(segment)
        return np.array(cleaned, dtype=np.float32)

    @classmethod
    def clean(cls, data: np.ndarray, method: str, params: dict) -> np.ndarray:
        if method == 'ts': return cls.apply_ts(data, **params)
        elif method == 'ica': return cls.apply_ica(data, **params)
        elif method == 'combo':
            temp = cls.apply_ts(data, win=params['win'], thresh=params['thresh'])
            return cls.apply_ica(temp, comps=params['comps'])
        return data

# ==========================================
# 2. HELPER FUNCTIONS (Filtering & Utils)
# ==========================================

def notch_filter(signal, Q=30.0):
    b_notch, a_notch = iirnotch(PipelineConfig.POWERLINE_FREQ, Q, PipelineConfig.FS)
    filtered = signal.copy()
    for i in range(filtered.shape[1]):
        filtered[:, i] = filtfilt(b_notch, a_notch, filtered[:, i])
    return filtered

def bandpass_filter(signal, lowcut=0.5, highcut=100, order=4):
    nyquist = 0.5 * PipelineConfig.FS
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)
    b, a = butter(order, [low, high], btype='band')
    bandpassed = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        bandpassed[:, i] = filtfilt(b, a, signal[:, i])
    return bandpassed

def median_detrend(signal, kernel_size_ms=200):
    kernel_size = int(kernel_size_ms * 1000 / PipelineConfig.FS)
    if kernel_size % 2 == 0: kernel_size += 1
    baseline = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        baseline[:, i] = medfilt(signal[:, i], kernel_size=max(kernel_size, 3))
    return signal - baseline

def complete_wavelet_filter(signal, wavelet='sym10', level=6):
    signal = np.nan_to_num(signal)
    filtered_notch = notch_filter(signal)
    bandpassed = bandpass_filter(filtered_notch)
    
    wavelet_result = np.zeros_like(bandpassed)
    for i in range(bandpassed.shape[1]):
        coeffs = pywt.wavedec(bandpassed[:, i], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0]) # Remove DC
        
        detail_coeffs = coeffs[-1]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(bandpassed[:, i])))
        for j in range(1, len(coeffs)):
            coeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
            
        reconstructed = pywt.waverec(coeffs, wavelet)
        # Fix length mismatch
        if len(reconstructed) > len(bandpassed): reconstructed = reconstructed[:len(bandpassed)]
        elif len(reconstructed) < len(bandpassed): 
            reconstructed = np.pad(reconstructed, (0, len(bandpassed)-len(reconstructed)), mode='edge')
        wavelet_result[:, i] = reconstructed

    baseline_corrected = median_detrend(wavelet_result)
    detrended = detrend(baseline_corrected, axis=0, type='linear')
    return detrended - np.mean(detrended, axis=0, keepdims=True)

# ==========================================
# 3. STREAMLIT APP LOGIC
# ==========================================

st.title("Fetal ECG QRS Detection System 🩺")
st.write("Upload a 4-channel AECG CSV file (Columns: AECG1, AECG2, AECG3, AECG4)")

# 1. Load Models
@st.cache_resource
def load_models():
    models = {}
    missing_files = []
    for name, conf in PipelineConfig.MODELS.items():
        if not os.path.exists(conf['path']):
            missing_files.append(conf['path'])
            continue
        model = ImprovedJudge()
        # Load weights on CPU
        model.load_state_dict(torch.load(conf['path'], map_location=torch.device('cpu')))
        model.eval()
        models[name] = model
    return models, missing_files

models, missing = load_models()

if missing:
    st.error(f"❌ Missing model files: {missing}")
    st.info("Please download them from your notebook output and place them in the same folder as this app.")
else:
    st.success("✅ All models loaded successfully!")

# 2. File Upload
uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])

if uploaded_file is not None and len(models) == 3:
    try:
        # Read Data
        df = pd.read_csv(uploaded_file)
        required_cols = ['AECG1', 'AECG2', 'AECG3', 'AECG4']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            raw_signal = df[required_cols].values
            st.subheader("1. Raw Signal Preview")
            st.line_chart(raw_signal[:2000]) # Plot first 2 seconds

            with st.spinner("Applying Wavelet Filtering & Preprocessing..."):
                filtered_signal = complete_wavelet_filter(raw_signal)
            
            # Segment Data
            X_list = []
            SEQ_LEN = PipelineConfig.SEQ_LEN
            STRIDE = PipelineConfig.SEQ_LEN # Non-overlapping for simple stitching
            
            # Normalize and Segment
            processed_for_model = filtered_signal.copy()
            for ch in range(4):
                ch_data = processed_for_model[:, ch]
                if np.std(ch_data) > 1e-6:
                    processed_for_model[:, ch] = (ch_data - np.mean(ch_data)) / np.std(ch_data)

            num_segments = (len(processed_for_model) - SEQ_LEN) // STRIDE + 1
            for i in range(0, num_segments * STRIDE, STRIDE):
                seg = processed_for_model[i:i+SEQ_LEN]
                X_list.append(seg.T) # Transpose to (4, 1000)
            
            X_segments = np.array(X_list, dtype=np.float32)
            
            # Prediction Loop
            st.subheader("2. Running Ensemble Prediction")
            progress_bar = st.progress(0)
            
            final_preds = []
            
            # We predict batch by batch or segment by segment
            # For simplicity in UI, just loop or create a simple loader
            tensor_X = torch.from_numpy(X_segments)
            
            ensemble_preds = np.zeros((len(X_segments), SEQ_LEN))
            
            for i, (name, model) in enumerate(models.items()):
                conf = PipelineConfig.MODELS[name]
                st.write(f"Running {name.upper()} model cleaning and inference...")
                
                # Clean specific to model
                X_cleaned = SignalCleaner.clean(X_segments, conf['method'], conf['params'])
                
                # Predict
                model_preds = []
                with torch.no_grad():
                    # Batch processing to prevent freeze
                    batch_size = 32
                    for b in range(0, len(X_cleaned), batch_size):
                        batch = torch.from_numpy(X_cleaned[b:b+batch_size])
                        logits = model(batch)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        model_preds.append(probs)
                
                ensemble_preds += np.concatenate(model_preds)
                progress_bar.progress((i + 1) / 3)

            # Average Ensemble
            ensemble_preds /= 3.0
            
            # Reconstruct continuous signal
            reconstructed = ensemble_preds.flatten()
            
            # Peak Detection
            smoothed = savgol_filter(reconstructed, window_length=51, polyorder=3)
            peaks, _ = find_peaks(smoothed, height=0.4, distance=int(PipelineConfig.FS * 0.35))
            
            st.subheader("3. Results")
            st.metric("Total QRS Peaks Detected", len(peaks))
            
            # Visualization of results
            # Create a dataframe for the chart: Signal vs Probability
            display_len = min(5000, len(reconstructed)) # Show first 5 seconds
            
            chart_data = pd.DataFrame({
                "Filtered AECG (Ch1)": processed_for_model[:display_len, 0],
                "QRS Probability": reconstructed[:display_len]
            })
            
            st.line_chart(chart_data)
            
            # Show Peak Locations
            st.write("Detected Peak Indices (First 20):")
            st.write(peaks[:20])
            
            # Download results
            res_df = pd.DataFrame({"Sample_Index": peaks})
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Peak Locations CSV", csv, "qrs_peaks.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")