# Gradio web app for AMP prediction with the saved C ⊕ D ensemble
# - Mutually exclusive input modes: CSV | FASTA | Manual
# - Rebuilds raw features in the SAME order used in training:
#     C: [kmer(2,3,4), ESM2 t6_8M_UR50D, modlAMP]
#     D: [physchem(core), ESM2 t6_8M_UR50D, modlAMP]

import re
import io
import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModel
from modlamp.descriptors import GlobalDescriptor
from collections import Counter
from itertools import product
from functools import lru_cache

# ----------------------
# Config
# ----------------------
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_VALID = set(AA_ALPHABET)
# For Spaces, keep the model file in repo root or /models and use a relative path:
JOBLIB_PATH = os.getenv("JOBLIB_PATH", "best_ensemble_cd.joblib")

# ----------------------
# PeptideEnsembleCD (must match class used when saving joblib)
# ----------------------
class PeptideEnsembleCD:
    def __init__(self,
                 modelC, scalerC_full, scalerC_keep, maskC, nzv_maskC,
                 modelD, scalerD_full, scalerD_keep, maskD, nzv_maskD,
                 alpha=0.65, thr=0.36):
        self.modelC = modelC
        self.scalerC_full = scalerC_full
        self.scalerC_keep = scalerC_keep
        self.maskC = maskC
        self.nzv_maskC = nzv_maskC

        self.modelD = modelD
        self.scalerD_full = scalerD_full
        self.scalerD_keep = scalerD_keep
        self.maskD = maskD
        self.nzv_maskD = nzv_maskD

        self.alpha = alpha
        self.thr = thr

    def _prep_block(self, X_raw, nzv_mask, scaler_full, keep_mask, scaler_keep):
        if nzv_mask is not None:
            X_raw = X_raw[:, nzv_mask]
        X = scaler_full.transform(X_raw)
        X = X[:, keep_mask]
        X = scaler_keep.transform(X)
        return X

    def predict_proba(self, X_C_raw, X_D_raw):
        Xc = self._prep_block(X_C_raw, getattr(self, "nzv_maskC", None),
                              self.scalerC_full, self.maskC, self.scalerC_keep)
        Xd = self._prep_block(X_D_raw, getattr(self, "nzv_maskD", None),
                              self.scalerD_full, self.maskD, self.scalerD_keep)
        pC = self.modelC.predict_proba(Xc)[:, 1]
        pD = self.modelD.predict_proba(Xd)[:, 1]
        return self.alpha * pC + (1 - self.alpha) * pD

    def predict(self, X_C_raw, X_D_raw):
        probs = self.predict_proba(X_C_raw, X_D_raw)
        return (probs > self.thr).astype(int)

# ----------------------
# Cached loaders (use lru_cache so Spaces won't re-download each run)
# ----------------------
@lru_cache(maxsize=1)
def load_ensemble(path=JOBLIB_PATH):
    return joblib.load(path)

@lru_cache(maxsize=1)
def load_esm(model_name=ESM_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

# ----------------------
# Parsing helpers
# ----------------------
FASTA_HDR = re.compile(r"^>.*$")

def parse_fasta(text: str):
    seqs, cur = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if FASTA_HDR.match(line):
            if cur:
                seqs.append(''.join(cur)); cur = []
        else:
            cur.append(re.sub(r"[^A-Za-z]", "", line))
    if cur: seqs.append(''.join(cur))
    return seqs

def parse_textbox(text: str):
    if not text: return []
    if ">" in text:  # FASTA-like
        return parse_fasta(text)
    return [re.sub(r"[^A-Za-z]", "", s.strip())
            for s in text.splitlines() if s.strip()]

def normalize_seq(s: str):
    s = s.upper()
    return ''.join([c for c in s if c in AA_VALID])

# ----------------------
# Feature builders (mirror training)
# ----------------------
def esm_embeddings(seqs, tokenizer, model, device, batch_size=16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            toks = [" ".join(s) for s in batch]
            inputs = tokenizer(toks, return_tensors="pt",
                               padding=True, truncation=True).to(device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embs.append(cls)
    return np.vstack(embs) if embs else np.zeros((0, model.config.hidden_size), dtype=np.float32)

def kmer_freqs(seqs, ks=[2,3,4]):
    all_feats = []
    for k in ks:
        vocab = [''.join(p) for p in product(AA_ALPHABET, repeat=k)]
        vidx = {kmer:i for i,kmer in enumerate(vocab)}
        mat = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
        for i, s in enumerate(seqs):
            kmers = [s[j:j+k] for j in range(len(s)-k+1)]
            kmers = [kmer for kmer in kmers if all(ch in AA_VALID for ch in kmer)]
            c = Counter(kmers)
            total = float(sum(c.values()))
            if total > 0:
                for kmer, cnt in c.items():
                    mat[i, vidx[kmer]] = cnt/total
        all_feats.append(mat)
    return np.concatenate(all_feats, axis=1) if all_feats else np.zeros((len(seqs),0), dtype=np.float32)

hydro_scale = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
    'W': -0.9, 'Y': -1.3
}
pKa_basic = {'K': 10.5, 'R': 12.5, 'H': 6.0}
pKa_N = 9.69
helix_pref = set("AEHKLMQR")
sheet_pref = set("VIYFWTC")

def hydrophobic_moment(seq, radians_per_res):
    if not seq: return 0.0
    angles = np.arange(len(seq)) * radians_per_res
    h = np.array([hydro_scale.get(a, 0.0) for a in seq], dtype=float)
    x = np.sum(h * np.cos(angles)); y = np.sum(h * np.sin(angles))
    return float(np.sqrt(x*x + y*y) / max(len(seq),1))

def positive_charge_at_pH(seq, pH=7.0, include_Nterm=True):
    chg = 0.0
    for aa, pKa in pKa_basic.items():
        n = seq.count(aa)
        chg += n * (1.0 / (1.0 + 10.0**(pH - pKa)))
    if include_Nterm and seq:
        chg += 1.0 / (1.0 + 10.0**(pH - pKa_N))
    return float(chg)

def cleavage_density(seq, set_chars):
    if not seq: return 0.0
    L, sites = len(seq), 0
    for i in range(L-1):
        if seq[i] in set_chars and seq[i+1] != 'P':
            sites += 1
    if seq[-1] in set_chars:
        sites += 1
    return sites / L

def physchem_core(seqs, pH=7.0):
    feats = []
    for s in seqs:
        L = len(s); Ls = max(L,1)
        KD = [hydro_scale.get(a, 0.0) for a in s]
        KD_mean = float(np.mean(KD)) if KD else 0.0
        muH_helix = hydrophobic_moment(s, np.deg2rad(100.0))
        muH_sheet = hydrophobic_moment(s, np.deg2rad(180.0))
        pos_charge = positive_charge_at_pH(s, pH=pH, include_Nterm=True)
        pos_charge_density = pos_charge / Ls
        f_helix = sum(1 for a in s if a in helix_pref) / Ls
        f_sheet = sum(1 for a in s if a in sheet_pref) / Ls
        dens_trypsin  = cleavage_density(s, set("KR"))
        dens_chymo    = cleavage_density(s, set("FYWL"))
        dens_elastase = cleavage_density(s, set("AVIL"))
        feats.append([float(L), KD_mean, muH_helix, muH_sheet,
                      pos_charge, pos_charge_density, f_helix, f_sheet,
                      dens_trypsin, dens_chymo, dens_elastase])
    return np.array(feats, dtype=np.float32)

def modlamp_features(seqs):
    desc = GlobalDescriptor(seqs)
    desc.calculate_all()
    return np.array(desc.descriptor, dtype=np.float32)

# Build raw matrices for C and D (order matters!)
def build_raw_C_D(seqs, tokenizer, model, device, batch_size=16):
    seqs = [normalize_seq(s) for s in seqs]
    X_kmer = kmer_freqs(seqs, ks=[2,3,4])
    X_phys = physchem_core(seqs, pH=7.0)
    X_modl = modlamp_features(seqs)
    X_esm  = esm_embeddings(seqs, tokenizer, model, device, batch_size=batch_size)
    X_C_raw = np.concatenate([X_kmer, X_esm, X_modl], axis=1)
    X_D_raw = np.concatenate([X_phys, X_esm, X_modl], axis=1)
    return X_C_raw, X_D_raw

# ----------------------
# Core inference
# ----------------------
def run_predict(mode, text, csv_file, fasta_file, batch_size):
    # Build sequence list from the active mode
    seqs = []
    if mode == "Manual":
        seqs = parse_textbox(text)
    elif mode == "CSV" and csv_file is not None:
        try:
            df = pd.read_csv(csv_file.name if hasattr(csv_file, "name") else csv_file)
            if 'peptide_sequence' not in df.columns:
                return None, None, None, "CSV must contain a 'peptide_sequence' column."
            seqs = df['peptide_sequence'].astype(str).tolist()
        except Exception as e:
            return None, None, None, f"Error reading CSV: {e}"
    elif mode == "FASTA" and fasta_file is not None:
        try:
            data = fasta_file.read() if hasattr(fasta_file, "read") else open(fasta_file.name, "rb").read()
            text = data.decode('utf-8', errors='ignore')
            seqs = parse_fasta(text)
        except Exception as e:
            return None, None, None, f"Error reading FASTA: {e}"

    seqs = [s for s in [normalize_seq(s) for s in seqs] if len(s) > 0]
    if not seqs:
        return None, None, None, "No sequences found for the selected input."

    try:
        ensemble = load_ensemble(JOBLIB_PATH)
    except Exception as e:
        return None, None, None, f"Failed to load ensemble joblib: {e}"

    try:
        tokenizer, esm_model, device = load_esm(ESM_MODEL_NAME)
    except Exception as e:
        return None, None, None, f"Failed to load ESM model: {e}"

    try:
        X_C_raw, X_D_raw = build_raw_C_D(seqs, tokenizer, esm_model, device, batch_size=int(batch_size))
    except Exception as e:
        return None, None, None, f"Feature computation failed: {e}"

    try:
        probs = ensemble.predict_proba(X_C_raw, X_D_raw)
        thr = getattr(ensemble, "thr", 0.36)
        preds = (probs > thr).astype(int)
    except Exception as e:
        return None, None, None, f"Prediction failed: {e}"

    df_out = pd.DataFrame({
        "peptide_sequence": seqs,
        "probability": probs,
        "prediction": preds
    })

    # Prepare downloadable CSV
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    csv_path = "predictions_ensemble_cd.csv"
    with open(csv_path, "wb") as f:
        f.write(csv_bytes)

    # Pie chart only for CSV/FASTA
    fig = None
    if mode in ("CSV", "FASTA"):
        counts = df_out["prediction"].value_counts().reindex([0,1], fill_value=0)
        c0, c1 = int(counts.get(0,0)), int(counts.get(1,0))
        fig, ax = plt.subplots()
        ax.pie([c0, c1], labels=["0", "1"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    return df_out, csv_path, fig, f"Loaded {len(seqs)} sequences. Threshold={getattr(ensemble,'thr',0.36)}"

# ----------------------
# UI (Gradio Blocks)
# ----------------------
with gr.Blocks(title="KPhysicoPIP (Gradio)") as demo:
    gr.Markdown("## 🧪 KPhysicoPIP — Pro-Inflammatory Peptide Predictor")

    with gr.Row():
        mode = gr.Radio(choices=["CSV","FASTA","Manual"], value="Manual", label="Choose ONE input mode")
        batch_size = gr.Slider(4, 128, value=16, step=4, label="ESM batch size")

    with gr.Row():
        text = gr.Textbox(label="Manual input (FASTA or one-per-line)",
                          placeholder=">seq1\nKLAKLAK...\n>seq2\nGIGKFLHSAKKFGKAFVGEIMNS...",
                          lines=10)
        with gr.Column():
            csv_upl = gr.File(label="Upload CSV (must have column: peptide_sequence)", file_types=[".csv"])
            fasta_upl = gr.File(label="Upload FASTA", file_types=[".fa",".fasta",".faa",".txt"])

    run_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        df_out = gr.Dataframe(label="Predictions", interactive=False, wrap=True)
    with gr.Row():
        csv_dl = gr.File(label="Download predictions.csv")
        fig_out = gr.Plot(label="Predicted label counts (CSV/FASTA only)")
    status = gr.Markdown()
    
    def clear_conflicts(m):
        """Simple UX helper: when a mode is chosen, ignore the other inputs visually.
        (No need to erase files; we just use the selected mode inside run_predict)."""
        return f"Active input mode: **{m}** (other inputs are ignored)."

    mode.change(fn=clear_conflicts, inputs=mode, outputs=status)
    run_btn.click(fn=run_predict,
                  inputs=[mode, text, csv_upl, fasta_upl, batch_size],
                  outputs=[df_out, csv_dl, fig_out, status])

if __name__ == "__main__":
    demo.launch()
