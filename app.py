import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="🩺",
    layout="centered"
)

# =========================================================
# LOAD MODEL & ARTEFAK (di-cache agar tidak load ulang tiap interaksi)
# =========================================================
@st.cache_resource
def load_artifacts():
    # Model dimuat dari format native (.txt) LightGBM -> aman lintas versi
    model = lgb.Booster(model_file="model_lgb.txt")
    scaler = joblib.load("scaler.pkl")
    le_dict = joblib.load("le_dict.pkl")
    le_target = joblib.load("le_target.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, le_dict, le_target, feature_columns

try:
    model, scaler, le_dict, le_target, feature_columns = load_artifacts()
except (FileNotFoundError, lgb.basic.LightGBMError) as e:
    st.error(
        "File model tidak ditemukan atau gagal dimuat. Pastikan file "
        "model_lgb.txt, scaler.pkl, le_dict.pkl, le_target.pkl, "
        "dan feature_columns.pkl ada di folder yang sama dengan app.py.\n\n"
        f"Detail error: {e}"
    )
    st.stop()

# =========================================================
# JUDUL & DESKRIPSI
# =========================================================
st.title("🩺 Prediksi Tingkat Obesitas")
st.write(
    "Aplikasi ini memprediksi tingkat obesitas seseorang berdasarkan "
    "kebiasaan makan dan gaya hidup, menggunakan model **LightGBM**."
)
st.divider()

# =========================================================
# FORM INPUT
# Kolom kategorikal -> selectbox (opsi diambil otomatis dari le_dict
#   yang disimpan saat training, jadi selalu konsisten dengan data latih)
# Kolom numerik -> number_input
# =========================================================
st.subheader("Masukkan Data")

user_input = {}

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    # --- Kolom numerik ---
    with col1:
        user_input["Usia"] = st.number_input(
            "Usia", min_value=1, max_value=120, value=25, step=1
        )
        user_input["Frekuensi_Makan_Sayur"] = st.slider(
            "Frekuensi Makan Sayur (1 = jarang, 3 = sering)", 1.0, 3.0, 2.0, 0.1
        )
        user_input["Jumlah_Makan_Utama"] = st.slider(
            "Jumlah Makan Utama per Hari", 1.0, 4.0, 3.0, 0.1
        )

    with col2:
        user_input["Konsumsi_Air_Putih"] = st.slider(
            "Konsumsi Air Putih per Hari (liter)", 1.0, 3.0, 2.0, 0.1
        )
        user_input["Frekuensi_Olahraga"] = st.slider(
            "Frekuensi Olahraga per Minggu (0 = tidak pernah, 3 = sering)", 0.0, 3.0, 1.0, 0.1
        )
        user_input["Durasi_Layar_Gadget"] = st.slider(
            "Durasi Pemakaian Gadget per Hari (0 = rendah, 2 = tinggi)", 0.0, 2.0, 1.0, 0.1
        )

    st.markdown("---")
    col3, col4 = st.columns(2)

    # --- Kolom kategorikal, opsi diambil dari encoder yang tersimpan ---
    with col3:
        for kolom in ["Jenis_Kelamin", "Riwayat_Keluarga", "Makan_Kalori_Tinggi", "Merokok"]:
            if kolom in le_dict:
                opsi = list(le_dict[kolom].classes_)
                user_input[kolom] = st.selectbox(kolom.replace("_", " "), opsi)

    with col4:
        for kolom in ["Sering_Ngemil", "Monitor_Kalori", "Konsumsi_Alkohol", "Transportasi_Utama"]:
            if kolom in le_dict:
                opsi = list(le_dict[kolom].classes_)
                user_input[kolom] = st.selectbox(kolom.replace("_", " "), opsi)

    submitted = st.form_submit_button("Prediksi Sekarang", use_container_width=True)

# =========================================================
# PROSES PREDIKSI
# =========================================================
if submitted:
    # 1. Buat DataFrame dari input, urutan kolom harus sama seperti saat training
    df_input = pd.DataFrame([user_input])
    df_input = df_input[feature_columns]

    # 2. Encode kolom kategorikal menggunakan encoder yang sama dari training
    for kolom, le in le_dict.items():
        if kolom in df_input.columns:
            df_input[kolom] = le.transform(df_input[kolom])

    # 3. Scaling menggunakan scaler yang sama dari training
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)

    # 4. Prediksi
    # lgb.Booster.predict() untuk kasus multi-kelas langsung mengembalikan
    # probabilitas tiap kelas (bukan label), jadi kita ambil argmax-nya.
    proba = model.predict(df_scaled.values)[0]
    pred_encoded = int(proba.argmax())
    pred_label = le_target.inverse_transform([pred_encoded])[0]

    # 5. Probabilitas tiap kelas (sudah didapat di atas)
    proba_df = pd.DataFrame({
        "Kelas": le_target.classes_,
        "Probabilitas": proba
    }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

    # =========================================================
    # TAMPILKAN HASIL
    # =========================================================
    st.divider()
    st.subheader("Hasil Prediksi")
    st.success(f"Prediksi tingkat obesitas: **{pred_label}**")

    st.write("Probabilitas untuk tiap kelas:")
    st.bar_chart(proba_df.set_index("Kelas"))
    st.dataframe(proba_df, use_container_width=True, hide_index=True)