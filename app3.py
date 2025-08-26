# app_ecg_streamlit.py
import io, requests, os
import streamlit as st
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from math import pi
from PIL import Image
from tensorflow.keras.models import load_model
from urllib.parse import quote

st.set_page_config(page_title="Classification ECG", layout="wide")

# =========================
# Modèles (cache + non bloquant)
# =========================
@st.cache_resource(show_spinner=True)
def load_cls_model(path: str):
    return load_model(path)

def try_load_model(path: str):
    try:
        return load_cls_model(path), None
    except Exception as e:
        return None, e

# =========================
# Utils
# =========================
def _best_numeric_vector_from_mat(mat_dict):
    best = None
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v).squeeze()
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            if best is None or arr.size > best.size:
                best = arr
    return best

# =========================
# Filtrage & FrFT
# =========================
def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=4):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.999999, highcut / nyq)
    if low >= high:
        high = min(0.99, max(low + 1e-3, high))
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def frft(f, a):
    N = len(f)
    shft = np.arange(N)
    shft = np.where(shft > N/2, shft - N, shft)
    alpha = a * pi / 2
    if a == 0:  return f
    if a == 1:  return np.fft.fft(f)
    if a == 2:  return np.flipud(f)
    if a == -1: return np.fft.ifft(f)
    tana2 = np.tan(alpha/2) + 1e-12
    sina  = np.sin(alpha) + 1e-12
    chirp1 = np.exp(-1j * pi * (shft**2) * tana2 / N)
    f2 = f * chirp1
    F = np.fft.fft(f2 * np.exp(-1j * pi * (shft**2) / (N * sina)))
    F = F * np.exp(-1j * pi * (shft**2) * tana2 / N)
    return F

def frft_magnitude_image(signal_1d, a, target_size=(224,224)):
    mag = np.abs(frft(signal_1d, a))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.plot(mag)
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    img_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    from PIL import Image as PILImage
    return PILImage.fromarray(img_array).resize(target_size)

# =========================
# Pan-Tompkins (simplifié)
# =========================
def pan_tompkins_detect(signal, fs):
    filtered = bandpass_filter(signal, 5, 15, fs)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2
    window = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(window)/window, mode='same')
    th = np.mean(integrated) + 0.5*np.std(integrated)
    peaks, _ = find_peaks(integrated, distance=max(1, int(0.3*fs)), height=th)
    return peaks

def extract_beats(signal_1d, r_peaks, fs, pre_s=0.3, post_s=0.3, max_beats=5):
    pre = int(pre_s * fs)
    post = int(post_s * fs)
    beats, centers, windows = [], [], []
    count = min(max_beats, len(r_peaks))
    for i in range(count):
        c = int(r_peaks[i])
        start = max(0, c - pre)
        end = min(len(signal_1d), c + post)
        seg = signal_1d[start:end]
        if seg.size >= 5:
            beats.append(seg)
            centers.append(c)
            windows.append((start, end))
    return beats, centers, windows

# =========================
# Heuristique image ECG
# =========================
def image_gray_std(img: Image.Image) -> float:
    g = img.convert("L")
    arr = np.asarray(g, dtype=np.float32)
    return float(arr.std())

def ecg_likeness_from_std(std_val: float, low_ok=5.0, high_ok=30.0) -> str:
    if std_val < low_ok:
        return "non_ecg_uniforme"
    if std_val > high_ok:
        return "non_ecg_texture"
    return "ecg_plausible"

# =========================
# En-tête : Logo + Titre + Message
# =========================
col1, col2 = st.columns([1, 8])
with col1:
    try:
        logo = Image.open("heart.png")   # Ajoute un 'heart.png' (cœur anatomique)
        st.image(logo, width=60)
    except Exception:
        st.write("❤️")  # fallback universel
with col2:
    st.title("Classification des battements cardiaques")

st.markdown(
    """
    ℹ️ **Conseil d'utilisation :**  
    - Pour un **signal temporel** (`.mat` ou `.csv`) → utilisez le modèle **FrFT**  
    - Pour une **image de battement ECG** (`.png` ou `.jpg`) → utilisez le modèle **Beat**
    """
)

# =========================
# Zone "Samples" (local OU GitHub)
# =========================
GITHUB_USER = "Simohamed7"
GITHUB_REPO = "ECG_classification_v10"
GITHUB_BRANCH = "main"
GITHUB_BASE = f"https://raw.githubusercontent.com/{quote(GITHUB_USER)}/{quote(GITHUB_REPO)}/{quote(GITHUB_BRANCH)}/SAMPLES"

LOCAL_SAMPLES_DIR = "SAMPLES"

@st.cache_data(ttl=300)
def list_local_samples():
    out = {}
    # signals
    loc_sig = os.path.join(LOCAL_SAMPLES_DIR, "signals")
    if os.path.isdir(loc_sig):
        for f in os.listdir(loc_sig):
            if f.lower().endswith((".mat", ".csv")):
                out[f] = os.path.join(loc_sig, f)
    # images
    loc_img = os.path.join(LOCAL_SAMPLES_DIR, "images")
    if os.path.isdir(loc_img):
        for f in os.listdir(loc_img):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                out[f] = os.path.join(loc_img, f)
    return out

@st.cache_data(ttl=300)
def list_github_samples():
    out = {}
    # GitHub API pour lister 2 dossiers
    for sub in ("signals", "images"):
        api = f"https://api.github.com/repos/{quote(GITHUB_USER)}/{quote(GITHUB_REPO)}/contents/SAMPLES/{sub}?ref={quote(GITHUB_BRANCH)}"
        r = requests.get(api, timeout=10)
        if r.status_code == 200:
            for item in r.json():
                if item.get("type") == "file":
                    name = item["name"]
                    if sub == "signals" and name.lower().endswith((".mat", ".csv")):
                        out[name] = f"{GITHUB_BASE}/signals/{name}"
                    if sub == "images" and name.lower().endswith((".png", ".jpg", ".jpeg")):
                        out[name] = f"{GITHUB_BASE}/images/{name}"
    return out

with st.expander("📁 Samples (choisir un fichier dans SAMPLES)"):
    samples_local = list_local_samples()
    samples_gh = list_github_samples() if not samples_local else {}
    source = "Local" if samples_local else ("GitHub" if samples_gh else "—")
    st.caption(f"Source détectée : **{source}**")

    SAMPLES = samples_local if samples_local else samples_gh
    if not SAMPLES:
        st.warning("Aucun sample trouvé (ni en local `SAMPLES/` ni sur GitHub).")
    else:
        sample_name = st.selectbox("Choisir un exemple :", sorted(SAMPLES.keys()))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬇️ Charger l'exemple sélectionné", use_container_width=True):
                try:
                    src = SAMPLES[sample_name]
                    # local file -> open ; url -> download
                    if os.path.exists(src):
                        with open(src, "rb") as f:
                            content = f.read()
                    else:
                        r = requests.get(src, timeout=20)
                        r.raise_for_status()
                        content = r.content
                    bio = io.BytesIO(content)
                    bio.name = sample_name
                    st.session_state["sample_file"] = bio
                    st.success(f"Exemple chargé : {sample_name}")
                except Exception as e:
                    st.error(f"Échec du chargement : {e}")
        with c2:
            if st.button("🔄 Réinitialiser l'exemple", use_container_width=True):
                st.session_state.pop("sample_file", None)
                st.info("Exemple réinitialisé. Importez un fichier ou rechargez un sample.")

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Paramètres")
model_choice = st.sidebar.radio("📌 Choix du modèle :", ["FrFT", "Beat"], index=0)
model_paths = {"FrFT": "FrFT.h5", "Beat": "Beat.h5"}

st.sidebar.write("Sélectionnez un fichier de test (.mat / .csv / .png / .jpg)")
uploaded_signal = st.sidebar.file_uploader(
    "Importer ECG (.mat, .csv, .png, .jpg)",
    type=["mat","csv","png","jpg","jpeg"]
)

# Si un sample est choisi, il remplace l'upload
if "sample_file" in st.session_state and st.session_state["sample_file"] is not None:
    uploaded_signal = st.session_state["sample_file"]
    st.info(f"Fichier en cours : **{uploaded_signal.name}** (depuis SAMPLES)")

# fs & alpha
fs = st.sidebar.number_input("Fréquence d'échantillonnage (Hz)", value=360, step=10, min_value=50, max_value=2000)
alpha_selected = st.sidebar.slider("Ordre FrFT (alpha)", 0.01, 1.0, 0.01, 0.01)

# =========================
# Chargement modèle (non bloquant)
# =========================
model, model_err = try_load_model(model_paths[model_choice])
if model is None:
    st.warning(f"Le modèle '{model_choice}' n'a pas été chargé (classification désactivée). Détail: {model_err}")

# =========================
# Fichier requis
# =========================
if uploaded_signal is None:
    st.info("Chargez un fichier ECG pour commencer (ou utilisez un Sample).")
    st.stop()

signal = None
name_lc = uploaded_signal.name.lower()

# =========================
# Branche image directe (Beat)
# =========================
if name_lc.endswith((".png","jpg","jpeg")):
    raw_img = Image.open(uploaded_signal).convert("RGB")
    sigma = image_gray_std(raw_img)
    verdict = ecg_likeness_from_std(sigma, low_ok=5, high_ok=30)
    if verdict != "ecg_plausible":
        st.warning("⚠️ Ça ne ressemble pas à un battement ECG")
        st.stop()

    img = raw_img.resize((224, 224))
    X = np.expand_dims(np.array(img, dtype=np.float32)/255.0, 0)

    st.image(img, caption="Image 224×224 (entrée)")

    if model is None:
        st.info("Classification désactivée car le modèle n'est pas chargé.")
        st.stop()

    class_names = ["F3", "N0", "Q4", "S1", "V2"]
    class_full_names = {"N0":"NORMAL","S1":"SUPRAVENTRICULAR","V2":"VENTRICULAR","F3":"FUSION","Q4":"UNKNOWN"}

    preds = model.predict(X, verbose=0)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    label_full = class_full_names.get(class_names[pred_idx], f"Classe {pred_idx}")

    st.subheader("✅ Résultat (Beat)")
    st.write("Classe:", f"**{label_full}**")
    st.write("Probabilités (%):", {class_full_names[c]: float(f"{preds[0][j]*100:.1f}") for j, c in enumerate(class_names)})
    st.stop()

# =========================
# Branche .mat / .csv (FrFT)
# =========================
elif name_lc.endswith(".mat"):
    try:
        mat_data = sio.loadmat(uploaded_signal)
        signal = _best_numeric_vector_from_mat(mat_data)
        if signal is None:
            st.error("Aucun vecteur numérique 1D valide trouvé dans le .mat.")
            st.stop()
    except Exception as e:
        st.error(f"Erreur lecture .mat : {e}")
        st.stop()

elif name_lc.endswith(".csv"):
    try:
        df = pd.read_csv(uploaded_signal)
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not num_cols:
            st.error("CSV sans colonne numérique.")
            st.stop()
        col = st.sidebar.selectbox("Colonne CSV à utiliser", num_cols, index=0)
        signal = df[col].astype(float).values
    except Exception as e:
        st.error(f"Erreur lecture .csv : {e}")
        st.stop()

# =========================
# 1) Signal original (temps en s)
# =========================
signal = np.asarray(signal, dtype=np.float32).squeeze()
if signal.ndim != 1:
    signal = signal.ravel()

t = np.arange(len(signal)) / fs

st.subheader("1) Signal original")
fig1, ax1 = plt.subplots()
ax1.plot(t, signal, linewidth=1)
ax1.set_xlabel("Temps (s)")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1); plt.close(fig1)

# =========================
# 2) Filtrage (temps en s)
# =========================
filtered = bandpass_filter(signal, 0.5, 50, fs)

st.subheader("2) Signal filtré")
fig2, ax2 = plt.subplots()
ax2.plot(t, filtered, linewidth=1)
ax2.set_xlabel("Temps (s)")
ax2.set_ylabel("Amplitude (filtrée)")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2); plt.close(fig2)

# =========================
# 3) R-peaks + BPM + Segmentation
# =========================
r_peaks = pan_tompkins_detect(filtered, fs)
if len(r_peaks) < 1:
    st.warning("Aucun R-peak détecté.")
    st.stop()

# BPM estimé
if len(r_peaks) > 1:
    rr_intervals = np.diff(r_peaks) / fs  # secondes
    bpm = 60.0 / np.mean(rr_intervals)
    st.info(f"🫀 Fréquence cardiaque estimée : **{bpm:.1f} BPM**")

st.subheader("3) R-peaks détectés + fenêtres des battements")
beats, centers, windows = extract_beats(filtered, r_peaks, fs, pre_s=0.30, post_s=0.30, max_beats=5)

fig3, ax3 = plt.subplots()
ax3.plot(t, filtered, linewidth=1, label="Filtré")
ax3.scatter(r_peaks / fs, filtered[r_peaks], s=20, c="tab:red", label="R-peaks", zorder=3)
for (start, end) in windows:
    ax3.axvspan(start / fs, end / fs, color="tab:green", alpha=0.15)
ax3.set_xlabel("Temps (s)")
ax3.set_ylabel("Amplitude")
ax3.grid(True, alpha=0.3)
ax3.legend()
st.pyplot(fig3); plt.close(fig3)

if len(beats) == 0:
    st.warning("Impossible d'extraire des battements.")
    st.stop()

# =========================
# 4) Battements (temps) + FrFT + prédictions
# =========================
st.subheader("4) Battements segmentés (temps) + FrFT + prédictions")

cols_time = st.columns(len(beats))
for i, seg in enumerate(beats):
    with cols_time[i]:
        t_seg = np.arange(len(seg)) / fs
        fig_seg, ax_seg = plt.subplots()
        ax_seg.plot(t_seg, seg, linewidth=1)
        ax_seg.set_title(f"Battement {i+1} (temps)")
        ax_seg.set_xlabel("Temps (s)")
        ax_seg.grid(True, alpha=0.3)
        st.pyplot(fig_seg); plt.close(fig_seg)

# Images FrFT (α contrôlé par le slider 0.01→1.0)
imgs_pil = [frft_magnitude_image(b, alpha_selected, (224,224)) for b in beats]
X = np.stack([np.array(img, dtype=np.float32)/255.0 for img in imgs_pil], axis=0)

# Si pas de modèle, on montre juste les images FrFT
if model is None:
    st.info("Images FrFT affichées, mais classification désactivée (modèle non chargé).")
    cols_img = st.columns(len(imgs_pil))
    for i, img_pil in enumerate(imgs_pil):
        with cols_img[i]:
            st.image(img_pil, caption=f"Battement {i+1} — FrFT (α={alpha_selected:.2f})", use_container_width=True)
    st.stop()

# Prédictions (FrFT)
class_names = ["F3", "N0", "Q4", "S1", "V2"]
class_full_names = {"N0":"NORMAL","S1":"SUPRAVENTRICULAR","V2":"VENTRICULAR","F3":"FUSION","Q4":"UNKNOWN"}
preds_all = model.predict(X, verbose=0)

cols_img = st.columns(len(imgs_pil))
for i, (img_pil, preds) in enumerate(zip(imgs_pil, preds_all)):
    pred_idx = int(np.argmax(preds))
    label_full = class_full_names.get(class_names[pred_idx], f"Classe {pred_idx}")
    with cols_img[i]:
        st.image(img_pil, caption=f"Battement {i+1} — {label_full} (α={alpha_selected:.2f})", use_container_width=True)
        st.caption("Probabilité (%)")
        for j, c in enumerate(class_names):
            pct = float(preds[j]) * 100.0
            line = f"{class_full_names[c]}: {pct:.1f}%"
            if j == pred_idx:
                st.markdown(f"**{line}**")
            else:
                st.write(line)

# =========================
# 5) Résultat final
# =========================
avg_probs = np.mean(preds_all, axis=0)
final_idx = int(np.argmax(avg_probs))
final_label = class_full_names[class_names[final_idx]]

st.subheader("5) ✅ Résultat final (moyenne des battements)")
st.write(f"Classe prédite: **{final_label}**")
for j, c in enumerate(class_names):
    st.write(f"{class_full_names[c]}: {avg_probs[j]*100:.1f}%")
