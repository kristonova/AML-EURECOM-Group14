# %% [markdown]
# ## Introduction
# This notebook constitutes a reproducible baseline for the Slide‑Rail Acoustic‑Anomaly Detection task of the AML 2025 course.
# It chronicles the complete experimental workflow, spanning data acquisition, signal processing, model construction, and result submission.
# The structure adheres to scholarly conventions to facilitate critical inspection and future extension.

# %%
import os
from pathlib import Path
import zipfile
import subprocess
import sys
import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import pandas as pd

# %% [markdown]
# ## Environment and Data Acquisition
# This section ensures that the required libraries are installed and retrieves the official DCASE Slide‑Rail dataset via the Kaggle API.
# Users who already possess a local copy may deactivate the download block by setting the `DOWNLOAD_DATA` flag to `False`.
# All paths are resolved relative to `DATA_DIR`, promoting portability across heterogeneous computing environments.

# %%
DOWNLOAD_DATA = False  # set to True on first run
DATA_DIR = Path("./data/slide_rail")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

if DOWNLOAD_DATA:
    kaggle_url = "dcase-task2-slidRail-2023.zip"  # placeholder; update with actual Kaggle file name
    if not (RAW_DIR / kaggle_url).exists():
        subprocess.run(["kaggle", "datasets", "download", "-d", "dcase-repo/dcase2023-task2", "-p", str(RAW_DIR)])
    with zipfile.ZipFile(RAW_DIR / kaggle_url, "r") as zf:
        zf.extractall(RAW_DIR)

# %% [markdown]
# ## Acoustic Feature Extraction
# Log‑Mel spectrograms are employed owing to their proven efficacy in representing perceptual frequency content for machine‑condition sounds.
# The helper function `extract_logmels` converts each waveform into a time–frequency representation that serves as input to both deep and classical models.
# Parameters follow the DCASE baseline configuration to enable fair comparison with published results.

# %%
def extract_logmels(wav_path, sr=16000, n_fft=1024, hop_length=512, n_mels=64):
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    logmel = librosa.power_to_db(mel).astype(np.float32)
    return logmel.T  # shape (time, n_mels)

# %% [markdown]
# ## Convolutional Auto‑Encoder Baseline
# The convolutional auto‑encoder (CAE) is adopted as a reconstruction‑based anomaly detector that minimizes the mean‑squared error on normal signals.
# During inference, samples exhibiting elevated reconstruction loss are hypothesised as anomalous, yielding a scalar anomaly score.
# The architecture remains intentionally lightweight to accommodate limited GPU resources typically available in academic settings.

# %%
class CAE(nn.Module):
    def __init__(self, n_mels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# Dummy training loop placeholder (user may implement full routine)

def train_cae(model, dataloader, epochs=20, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for ep in range(epochs):
        epoch_loss = 0.0
        for xb in dataloader:
            xb = xb.to(next(model.parameters()).device)
            recon = model(xb)
            loss = crit(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {ep+1}/{epochs} - loss: {epoch_loss/len(dataloader):.4f}")

# %% [markdown]
# ## Gaussian Mixture Model Baseline
# As a classical point of reference, a Gaussian Mixture Model (GMM) estimates the distribution of normal MFCC vectors by maximum‑likelihood via the Expectation‑Maximization algorithm.
# Samples with low log‑likelihood under the trained density are deemed anomalous, enabling a direct comparison with reconstruction‑based deep models.
# Although simplistic, the GMM continues to serve as a competitive non‑neural baseline in the DCASE evaluations.

# %%

def extract_mfcc(wav_path, sr=16000, n_mfcc=20, hop_length=512):
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc.T  # shape (time, n_mfcc)

class GMMDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        return extract_mfcc(self.file_paths[idx])

# Placeholder fitting

def fit_gmm(train_files, n_components=16):
    feats = np.vstack([extract_mfcc(f) for f in train_files])
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=100)
    gmm.fit(feats)
    return gmm

# %% [markdown]
# ## Evaluation Metrics
# The Area Under the Receiver Operating Characteristic Curve (AUC‑ROC) is adopted as the principal metric, consistent with prior DCASE tasks.
# It offers a threshold‑independent assessment of ranking quality, thereby accommodating heterogeneous operating scenarios.
# Confidence intervals may be approximated via bootstrap resampling to gauge statistical significance among competing models.

# %%

def evaluate(scores, labels):
    auc = metrics.roc_auc_score(labels, scores)
    print(f"AUC‑ROC: {auc:.3f}")
    return auc

# %% [markdown]
# ## Submission File Generation
# The final cell assembles the anomaly scores for the evaluation subset into a comma‑separated values (CSV) file, conforming to the challenge submission format.
# The CSV comprises two columns—`filename` and `score`—without a header, aligning with the official evaluation script.
# Ensure deterministic ordering of filenames to preclude inadvertent misalignment between scores and audio files.

# %%

def write_submission(filenames, scores, out_path="submission.csv"):
    df = pd.DataFrame({"filename": filenames, "score": scores})
    df.to_csv(out_path, index=False, header=False)
    print(f"Submission saved to {out_path}")
