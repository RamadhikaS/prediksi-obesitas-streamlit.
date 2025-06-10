import streamlit as st
import pandas as pd
import joblib
import pickle

# --- KONFIGURASI HALAMAN (HARUS MENJADI PERINTAH STREAMLIT PERTAMA) ---
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="⚖️",
    layout="centered"
)

# --- FUNGSI UNTUK MEMUAT ASET ---
# Menggunakan cache agar tidak perlu memuat ulang model setiap kali ada interaksi
@st.cache_data
def load_assets():
    model = joblib.load('obesity_model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    with open('target_mapping.pkl', 'rb') as f:
        target_mapping = pickle.load(f)
    return model, scaler, columns, target_mapping

# --- FUNGSI UNTUK PREPROCESSING INPUT ---
def preprocess_input(input_data, scaler, columns):
    # Buat DataFrame dari input
    df = pd.DataFrame([input_data])

    # --- Encoding Manual (sesuai preprocessing awal) ---
    # Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    # Fitur Biner (Yes/No)
    for col in ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    # CAEC & CALC
    caec_map = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_map = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df['CAEC'] = df['CAEC'].map(caec_map)
    df['CALC'] = df['CALC'].map(calc_map)

    # --- One-Hot Encoding untuk MTRANS ---
    # Pastikan semua kategori MTRANS ada sebagai kolom
    mtrans_categories = ['MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']
    for cat in mtrans_categories:
        df[cat] = 0
    # Set nilai 1 untuk kategori yang dipilih
    if 'MTRANS_' + input_data['MTRANS'] in mtrans_categories:
        df['MTRANS_' + input_data['MTRANS']] = 1
    df = df.drop('MTRANS', axis=1)

    # --- Urutkan Kolom ---
    # Pastikan urutan kolom sesuai dengan saat training
    df = df.reindex(columns=columns, fill_value=0)

    # --- Scaling ---
    # Scaling data menggunakan scaler yang sudah di-fit
    scaled_data = scaler.transform(df)
    
    return scaled_data

# --- TAMPILAN APLIKASI STREAMLIT ---

# Muat aset
model, scaler, columns, target_mapping = load_assets()

# Judul Aplikasi
st.title("⚖️ Prediksi Tingkat Obesitas")
st.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik. Silakan masukkan data Anda di bawah ini.")

st.divider()

# Buat kolom untuk input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Informasi Pribadi")
    Age = st.number_input("Usia", min_value=1, max_value=100, value=25)
    Gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    Height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.75, format="%.2f")
    Weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=70.0, format="%.1f")
    family_history_with_overweight = st.radio("Riwayat keluarga dengan kelebihan berat badan?", ["Yes", "No"])

with col2:
    st.subheader("Kebiasaan Makan & Minum")
    FAVC = st.radio("Sering makan makanan tinggi kalori (FAVC)?", ["Yes", "No"])
    FCVC = st.slider("Frekuensi makan sayuran (FCVC)", 1, 3, 2, help="1: Tidak pernah, 2: Terkadang, 3: Selalu")
    NCP = st.slider("Jumlah makan besar per hari (NCP)", 1, 4, 3)
    CAEC = st.selectbox("Makan cemilan di antara waktu makan (CAEC)?", ["No", "Sometimes", "Frequently", "Always"])
    CH2O = st.slider("Konsumsi air per hari (liter) (CH2O)", 1.0, 3.0, 2.0, step=0.5)
    CALC = st.selectbox("Frekuensi konsumsi alkohol (CALC)?", ["No", "Sometimes", "Frequently", "Always"])

st.divider()
st.subheader("Aktivitas dan Kebiasaan Lain")

col3, col4 = st.columns(2)

with col3:
    SCC = st.radio("Memantau asupan kalori (SCC)?", ["Yes", "No"])
    FAF = st.slider("Frekuensi aktivitas fisik per minggu (FAF)", 0, 3, 1, help="0: Tidak sama sekali, 1: 1-2 hari, 2: 2-4 hari, 3: 4-5 hari")
    TUE = st.slider("Waktu penggunaan perangkat teknologi per hari (jam) (TUE)", 0, 2, 1, help="0: 0-2 jam, 1: 3-5 jam, 2: >5 jam")
    
with col4:
    SMOKE = st.radio("Apakah Anda merokok?", ["Yes", "No"])
    MTRANS = st.selectbox("Transportasi utama yang digunakan (MTRANS)", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])


# Tombol untuk prediksi
if st.button("Prediksi Tingkat Obesitas Saya", type="primary"):
    # Kumpulkan input menjadi dictionary
    input_data = {
        'Age': Age, 'Height': Height, 'Weight': Weight, 'FCVC': float(FCVC), 'NCP': float(NCP),
        'CH2O': float(CH2O), 'FAF': float(FAF), 'TUE': float(TUE), 'Gender': Gender,
        'family_history_with_overweight': family_history_with_overweight, 'FAVC': FAVC,
        'CAEC': CAEC, 'SMOKE': SMOKE, 'SCC': SCC, 'CALC': CALC, 'MTRANS': MTRANS
    }

    # Preprocess input
    processed_input = preprocess_input(input_data, scaler, columns)

    # Lakukan prediksi
    prediction = model.predict(processed_input)
    
    # Ambil label hasil prediksi dari mapping
    predicted_label = target_mapping[prediction[0]]

    # Tampilkan hasil
    st.success(f"**Hasil Prediksi: {predicted_label}**")
    
    with st.expander("Lihat Detail Input yang Diproses"):
        st.write("Data mentah yang Anda masukkan:")
        st.write(input_data)
        st.write("Data setelah diproses untuk model:")
        st.write(pd.DataFrame(processed_input, columns=columns))
