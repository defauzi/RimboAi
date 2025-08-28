# ==========================================================
# Streamlit — Plant & Disease (NasNet) + Multi-Layer Eigen-CAM
# ==========================================================
import os, io, glob, time
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image, ImageFilter
import cv2
import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# ── Keras 3 / TF 2.15+: bikin tracing lebih "jinak"
tf.config.run_functions_eagerly(True)

# ---------- KONFIG ----------
BASE_DIR    = r"C:\Users\.CPG\Documents\RimboAi"
MODEL_DIR   = os.path.join(BASE_DIR, "Model")
ENCODER_DIR = os.path.join(BASE_DIR, "Encoder")   # opsional
IMG_SIZE    = (224, 224)

# nama head (kalau beda, otomatis dinormalisasi)
PLANT_HEAD   = "plant_output"
DISEASE_HEAD = "disease_output"

# Eigen-CAM (diselaraskan dengan Colab)
EIGEN_LAYERS_K = 3            # ambil 3 feature map terakhir
FUSION_METHOD  = "weighted"   # "mean" | "max" | "weighted"
PROBE_CHUNK    = 60
MIN_SIZE_LIST  = [14, 7, 1]   # prefer resolusi spatial lebih besar
PNORM          = 99.5
GAMMA          = 0.85
ALPHA_OVERLAY  = 0.45

st.set_page_config(page_title="RimboAi", layout="wide")
st.title("🌿 RimboAi - Plant & Disease Prediction Using NasnetMobile")
st.caption("Upload gambar daun → klik **Proses**. Model otomatis diambil dari folder **Model/** dan di-cache.")

# ==== Utils normalisasi & masking daun (HSV + Otsu) ====
import matplotlib

def percentile_norm(x: np.ndarray, p: float = 99.5) -> np.ndarray:
    x = x - np.min(x)
    hi = np.percentile(x, p)
    if hi <= 1e-8:
        return np.zeros_like(x)
    x = np.clip(x / hi, 0, 1)
    return x.astype("float32")

def hsv_leaf_mask(img_rgb01: np.ndarray) -> np.ndarray:
    """img_rgb01: [0..1], shape (H,W,3) -> bool mask"""
    hsv = matplotlib.colors.rgb_to_hsv(img_rgb01)  # 0..1
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m1 = (H >= 0.12) & (H <= 0.50) & (S >= 0.18) & (V >= 0.18)               # hijau/kuning
    m2 = (S >= 0.10) & (V >= 0.25) & (img_rgb01[...,1] > img_rgb01[...,2])   # G > B
    return (m1 | m2)

def otsu_mask(gray01: np.ndarray) -> np.ndarray:
    g = (gray01 * 255).astype(np.uint8)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th > 0

def largest_cc(mask_bool: np.ndarray) -> np.ndarray:
    m = mask_bool.astype(np.uint8)
    num, lab = cv2.connectedComponents(m, connectivity=4)
    if num <= 1:
        return mask_bool
    sizes = [(lab == i).sum() for i in range(1, num)]
    k = 1 + int(np.argmax(sizes))
    return (lab == k)

def refine_mask(mask_bool: np.ndarray, iters: int = 2) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    m = (mask_bool * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=iters)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=iters)
    return (m > 0)

def build_leaf_mask(pil_img: Image.Image, iters: int = 2) -> np.ndarray:
    img01 = np.asarray(pil_img, dtype=np.float32) / 255.0
    m = hsv_leaf_mask(img01) | otsu_mask(img01.mean(axis=-1))
    m = refine_mask(m, iters=max(0, iters))
    m = largest_cc(m)
    return m  # bool [H,W]

def apply_mask_to_heatmap(heat01: np.ndarray, mask_bool: np.ndarray,
                          mode: str = "attenuate", atten: float = 0.25) -> np.ndarray:
    if mode == "hard":
        out = heat01 * mask_bool.astype(np.float32)
    else:
        out = heat01 * (mask_bool.astype(np.float32) + (~mask_bool).astype(np.float32) * atten)
    out = out / (out.max() + 1e-8)
    return out

def overlay_heatmap_on_image(img_rgb: np.ndarray, heat: np.ndarray, alpha: float = ALPHA_OVERLAY) -> np.ndarray:
    heat_uint8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heat_color, alpha, img_bgr, 1 - alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def make_gray_heat(
    heat01: np.ndarray,
    ori_pil: Image.Image,
    p: float = 99.5,
    gamma: float = 0.85,
    blur_px: float = 1.5,
    use_mask: bool = True,
    morph_iters: int = 2,
) -> np.ndarray:
    """Kembalikan heatmap grayscale 0..255 dengan latar hitam (termask daun)."""
    h = percentile_norm(heat01, p=p)
    h = np.power(h, gamma)
    h = h / (h.max() + 1e-8)

    if use_mask:
        m = build_leaf_mask(ori_pil, iters=morph_iters)   # bool ukuran ori
        m224 = cv2.resize(m.astype(np.float32), IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        h = h * m224

    if blur_px and blur_px > 0:
        h = cv2.GaussianBlur(h.astype(np.float32), ksize=(0, 0), sigmaX=blur_px, sigmaY=blur_px)

    h = h / (h.max() + 1e-8)
    gray = (h * 255.0).astype("uint8")
    return gray

# ---------- UTIL PATH ----------
def find_latest_model(model_dir: str) -> Optional[str]:
    pats = ["*.keras", "*.h5"]
    cands: List[str] = []
    for p in pats: cands.extend(glob.glob(os.path.join(model_dir, p)))
    if not cands: return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]

# ---------- PRE/POST ----------
def preprocess_pil(img: Image.Image) -> np.ndarray:
    img_resized = img.resize(IMG_SIZE, resample=Image.BILINEAR)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    return arr

def argmax_and_score(probs: np.ndarray) -> Tuple[int, float]:
    idx = int(np.argmax(probs)); score = float(probs[idx])
    return idx, score

# ---------- LABEL ENCODER (opsional) ----------
@st.cache_data
def load_label_encoders():
    import joblib, pickle
    plant_pkl   = os.path.join(ENCODER_DIR, "plant_encoder.pkl")
    disease_pkl = os.path.join(ENCODER_DIR, "disease_encoder.pkl")

    def safe_load(p):
        if not os.path.exists(p): return None, f"(Info) Tidak ditemukan: {os.path.basename(p)}"
        try: return joblib.load(p), None
        except Exception:
            try:
                with open(p, "rb") as f: return pickle.load(f), None
            except Exception as e: return None, f"(Skip) Gagal load {os.path.basename(p)}: {e}"

    plant_enc, w1   = safe_load(plant_pkl)
    disease_enc, w2 = safe_load(disease_pkl)
    warns = [w for w in (w1, w2) if w]
    return plant_enc, disease_enc, warns

def decode_label(idx: int, enc) -> str:
    if enc is None: return str(idx)
    try: return enc.inverse_transform([idx])[0]
    except Exception: return str(idx)

# ---------- MODEL BUNDLE (probe backbone & siapkan extractor) ----------
@st.cache_resource
def get_model_bundle():
    model_path = find_latest_model(MODEL_DIR)
    if model_path is None:
        raise FileNotFoundError(f"Tidak ada file model (.keras/.h5) di {MODEL_DIR}")
    model = load_model(model_path, compile=False)

    # backbone NASNetMobile
    try:
        backbone = model.get_layer("nasnet_mobile")
    except Exception as e:
        raise AssertionError(f"Backbone 'nasnet_mobile' tidak ditemukan: {e}")

    # probe semua layer 4D di backbone (chunked)
    def probe_layers_chunked(bbone, img_size=(224,224), chunk=50):
        layers_with_out = []
        for l in bbone.layers:
            try: _ = l.output; layers_with_out.append(l)
            except Exception: pass
        dummy = tf.zeros((1, img_size[0], img_size[1], 3), dtype=tf.float32)
        results = []
        for i in range(0, len(layers_with_out), chunk):
            sub = layers_with_out[i:i+chunk]
            try:
                probe = Model(bbone.input, [l.output for l in sub])
                vals  = probe(dummy, training=False)
                for l, v in zip(sub, vals):
                    shp = tuple(v.shape.as_list()) if hasattr(v.shape,"as_list") else tuple(v.shape)
                    results.append((l.name, shp))
            except Exception:
                for l in sub:
                    try:
                        probe = Model(bbone.input, l.output)
                        v = probe(dummy, training=False)
                        shp = tuple(v.shape.as_list()) if hasattr(v.shape,"as_list") else tuple(v.shape)
                        results.append((l.name, shp))
                    except Exception:
                        pass
        cands=[]
        for name, shp in results:
            if len(shp)==4 and None not in shp[1:]:
                h,w,c = int(shp[1]), int(shp[2]), int(shp[3])
                cands.append((name,h,w,c))
        return cands

    cands_all = probe_layers_chunked(backbone, img_size=IMG_SIZE, chunk=PROBE_CHUNK)
    if not cands_all:
        raise AssertionError("Tidak menemukan feature map 4D pada backbone.")

    # pilih K layer terakhir dengan preferensi resolusi lebih besar
    def pick_layers(cands, min_size_list, k):
        for minsz in min_size_list:
            pool = [t for t in cands if t[1]>=minsz and t[2]>=minsz]
            if pool: return pool[-min(k, len(pool)):]
        return cands[-min(k, len(cands)):]

    chosen = pick_layers(cands_all, MIN_SIZE_LIST, EIGEN_LAYERS_K)
    extractors = [Model(backbone.input, backbone.get_layer(n).output) for (n,_,_,_) in chosen]
    chans      = [c for (_,_,_,c) in chosen]
    for ex in extractors:
        ex.trainable = False
        try: ex.compile(run_eagerly=True)
        except Exception: pass

    # normalisasi nama head
    keras_outputs = [t.name for t in model.outputs]
    plant_head = PLANT_HEAD if PLANT_HEAD in keras_outputs else keras_outputs[0]
    disease_head = DISEASE_HEAD if DISEASE_HEAD in keras_outputs else (
        keras_outputs[1] if len(keras_outputs) > 1 else keras_outputs[0]
    )

    return {
        "model": model,
        "path": model_path,
        "extractors": extractors,
        "extractor_channels": chans,
        "chosen_layer_names": [n for (n,_,_,_) in chosen],
        "plant_head": plant_head,
        "disease_head": disease_head,
    }

# ---------- INIT ----------
B = get_model_bundle()
MODEL: tf.keras.Model = B["model"]
EXTRACTORS = B["extractors"]
EXTRACTOR_CHANS = B["extractor_channels"]
CHOSEN_NAMES = B["chosen_layer_names"]
PLANT_HEAD_NAME = B["plant_head"]
DISEASE_HEAD_NAME = B["disease_head"]

st.sidebar.success("Model loaded ✅")
st.sidebar.caption(f"Source: {B['path']}")
st.sidebar.caption(f"Eigen layers: {', '.join(CHOSEN_NAMES)}")

PLANT_ENC, DISEASE_ENC, ENC_WARN = load_label_encoders()
for m in ENC_WARN: st.sidebar.info(m)
if PLANT_ENC is None or DISEASE_ENC is None:
    st.sidebar.warning("Encoder tidak aktif → label ditampilkan sebagai indeks kelas.")

# ---------- EIGEN-CAM core ----------
def eigen_map_from_extractor(arr01, extractor, size):
    """arr01: (H,W,3) float[0..1], extractor -> fmap (H',W',C) -> eigen-cam [size]."""
    fmap = extractor(tf.expand_dims(arr01,0), training=False)[0].numpy()  # (H,W,C)
    H, W, C = fmap.shape
    X = fmap.reshape((-1, C))
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        cov = Xc.T @ Xc
        _, evecs = np.linalg.eigh(cov)
        v = evecs[:, -1]
        cam = (X @ v).reshape(H, W)
    except Exception:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        v = Vt[0, :]
        cam = (X @ v).reshape(H, W)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam.astype("float32"), size, interpolation=cv2.INTER_CUBIC)
    return cam

def fuse_cams(cams, method="mean", chans=None):
    cams = np.stack(cams, axis=0)  # [L,H,W]
    if method == "max":
        return cams.max(axis=0)
    if method == "weighted" and chans is not None and len(chans)==cams.shape[0]:
        w = np.asarray(chans, np.float32); w = w / (w.sum() + 1e-8)
        return np.tensordot(w, cams, axes=(0,0))
    return cams.mean(axis=0)

def input_gradient_saliency(pil_img: Image.Image) -> np.ndarray:
    """fallback terakhir bila extractor Eigen-CAM gagal"""
    img = preprocess_pil(pil_img)
    x = tf.convert_to_tensor(img[None, ...], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = MODEL(x, training=False)
        if isinstance(out, (list, tuple)):
            y = out[0][..., tf.argmax(out[0][0])]
        else:
            y = out[..., tf.argmax(out[0])]
    grad = tape.gradient(y, x)[0].numpy()
    g = np.mean(np.abs(grad), axis=-1)
    g = g / (g.max() + 1e-8)
    return g.astype("float32")

def get_cam_for_image(pil_img: Image.Image):
    inp = preprocess_pil(pil_img)  # (224,224,3) float 0..1
    try:
        per_layer = [eigen_map_from_extractor(inp, ex, IMG_SIZE) for ex in EXTRACTORS]
        heat = fuse_cams(
            per_layer,
            method=FUSION_METHOD,
            chans=EXTRACTOR_CHANS if FUSION_METHOD=="weighted" else None
        )
        heat = percentile_norm(heat, p=PNORM)
        heat = np.power(heat, GAMMA)
        heat = heat / (heat.max() + 1e-8)
        mode = "eigen-cam"
    except Exception as e:
        st.warning(f"Eigen-CAM gagal di backend, pakai saliency. Detail: {e}")
        heat = input_gradient_saliency(pil_img)
        mode = "fallback-saliency"

    heat_224 = cv2.resize(heat, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    overlay_224 = overlay_heatmap_on_image((inp * 255).astype("uint8"), heat_224, alpha=ALPHA_OVERLAY)

    # heat grayscale (sesuai contoh Colab)
    heat_gray = make_gray_heat(
        heat_224,
        pil_img.resize(IMG_SIZE),
        p=PNORM, gamma=GAMMA, blur_px=1.5, use_mask=True, morph_iters=2
    )
    return heat_224, overlay_224, heat_gray, mode

# ---------- PREDIKSI ----------
def predict_heads(pil_img: Image.Image):
    inp = preprocess_pil(pil_img)[None, ...]
    outputs = MODEL.predict(inp, verbose=0)
    if isinstance(outputs, (list, tuple)):
        out_map = {t.name: o for t, o in zip(MODEL.outputs, outputs)}
        plant_probs = out_map.get(PLANT_HEAD_NAME, outputs[0])[0]
        disease_probs = out_map.get(DISEASE_HEAD_NAME, outputs[min(1, len(outputs)-1)])[0]
    else:
        plant_probs = disease_probs = outputs[0]
    p_idx, p_score = argmax_and_score(plant_probs)
    d_idx, d_score = argmax_and_score(disease_probs)
    plant_label   = decode_label(p_idx, PLANT_ENC)
    disease_label = decode_label(d_idx, DISEASE_ENC)
    return plant_label, float(p_score), disease_label, float(d_score)

# ---------- UI ----------
files = st.file_uploader(
    "Pilih gambar (JPG/PNG), bisa lebih dari satu:",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)
run = st.button("Proses")

col1, col2, col3 = st.columns(3)
with col1: st.subheader("Uploaded Images")
with col2: st.subheader("Prediction Results")
with col3: st.subheader("EIGEN-CAM (Ori • Heat • Overlay)")

rows = st.container()

def read_image(uploaded) -> Image.Image:
    return Image.open(io.BytesIO(uploaded.getbuffer())).convert("RGB")

if run:
    if not files:
        st.error("Belum ada file yang diupload.")
    else:
        for f in files:
            img = read_image(f)
            c1, c2, c3 = rows.columns(3)

            # Kolom 1 — Ori
            with c1:
                st.image(img, caption=f"Uploaded: {f.name}", use_container_width=True)

            # Kolom 2 — Prediksi
            with c3:
                heat_224, overlay_224, heat_gray, mode = get_cam_for_image(img)
                if mode != "eigen-cam":
                    st.warning("Eigen-CAM gagal di backend, menampilkan *Input-Gradient Saliency* sebagai fallback.")
                ori_224 = np.asarray(img.resize(IMG_SIZE))
                g1, g2, g3 = st.columns(3)
                with g1: st.image(ori_224, caption="Ori", use_container_width=True)
                # grayscale heat (mask + blur), clamp agar tidak auto-rescale brightness
                with g2: st.image(heat_gray, caption="Heat", use_container_width=True, clamp=True)
                with g3: st.image(overlay_224, caption="Overlay", use_container_width=True)

            # Kolom 3 — Eigen-CAM
            t0 = time.time()
            plant_label, p_score, disease_label, d_score = predict_heads(img)
            t_pred = (time.time() - t0) * 1000.0
            with c2:
                st.success(
                    f"**Plant:** {plant_label} ({p_score:.3f})\n\n"
                    f"**Disease:** {disease_label} ({d_score:.3f})"
                )
                st.caption(f"Inference: {t_pred:.1f} ms")
                st.image(img.resize(IMG_SIZE), caption="Input (224×224)", use_container_width=True)

else:
    st.info("Pilih file lalu klik **Proses**.")
