# inspect_layers.py
import os, glob
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR  = r"C:\Users\.CPG\Documents\PrototipeNasnet"
MODEL_DIR = os.path.join(BASE_DIR, "Model")

def find_latest_model(model_dir: str):
    cands = []
    for p in ("*.keras", "*.h5"):
        cands += glob.glob(os.path.join(model_dir, p))
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]

def list_layers(m, prefix=""):
    rows = []
    for lyr in m.layers:
        name = prefix + lyr.name
        try:
            shape = getattr(lyr, "output_shape", None)
        except Exception:
            shape = None
        rows.append((name, str(shape), lyr.__class__.__name__))
        # kalau ini submodel, masuk lagi
        if isinstance(lyr, tf.keras.Model):
            rows += list_layers(lyr, prefix=name + "/")
    return rows

def main():
    path = find_latest_model(MODEL_DIR)
    if not path:
        print(f"Tidak ada file model di {MODEL_DIR}")
        return
    print(f"Memuat model: {path}")
    model = load_model(path, compile=False)

    rows = list_layers(model)
    # cetak ke konsol
    for name, shape, cls in rows:
        print(f"{name:60s}  {shape:20s}  {cls}")

    # simpan ke file
    out_txt = os.path.join(BASE_DIR, "layers_list.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("name\toutput_shape\tclass\n")
        for name, shape, cls in rows:
            f.write(f"{name}\t{shape}\t{cls}\n")
    print(f"\n>> Disimpan ke: {out_txt}")
    print("Cari layer TERAKHIR dengan output 4D (contoh: (None, H, W, C)).")

if __name__ == "__main__":
    main()
