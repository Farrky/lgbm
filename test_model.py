"""
Script diagnostik: jalankan file ini langsung dengan
    python test_model.py
(BUKAN lewat streamlit run) untuk memastikan LightGBM
bisa predict dengan benar di environment ini, terpisah
dari masalah Streamlit.
"""

import numpy as np
import lightgbm as lgb
import joblib

print("Versi LightGBM:", lgb.__version__)
print("Versi NumPy   :", np.__version__)

# 1. Load model
model = lgb.Booster(model_file="model_lgb.txt")
print("Jumlah fitur yang diharapkan model:", model.num_feature())

# 2. Load daftar kolom fitur yang dipakai saat training
feature_columns = joblib.load("feature_columns.pkl")
print("Jumlah kolom di feature_columns.pkl:", len(feature_columns))
print("Daftar kolom:", feature_columns)

# 3. Buat data dummy sesuai jumlah fitur yang benar, lalu coba predict
n_features = model.num_feature()
dummy = np.zeros((1, n_features), dtype=np.float64, order="C")

print("\nMencoba predict dengan data dummy (angka nol semua)...")
try:
    hasil = model.predict(dummy)
    print("BERHASIL! Hasil prediksi (probabilitas tiap kelas):", hasil)
except Exception as e:
    print("GAGAL. Error:", type(e).__name__, e)