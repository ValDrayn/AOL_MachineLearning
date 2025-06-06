import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Slot Parkir",
    page_icon="🚗",
    layout="wide"
)

# --- Judul dan Deskripsi ---
st.title("🅿️ Deteksi Status Slot Parkir")
st.write(
    "Unggah gambar area parkir untuk mendeteksi slot yang kosong (empty) dan terisi (filled). "
    "Aplikasi ini menggunakan model YOLO yang telah di-fine-tune."
)

MODEL_PATH = 'best.pt'

@st.cache_resource
def load_model(model_path):
    """Memuat model YOLO dari path yang diberikan."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Muat model
model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Pilih sebuah gambar...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, caption="Gambar yang Anda unggah.", use_column_width=True)

    # Tombol untuk memulai deteksi
    if st.button("Deteksi Slot Parkir"):
        if model is not None:
            with st.spinner('Sedang memproses...'):
                # Lakukan prediksi
                results = model.predict(image, conf=0.1)

                annotated_image_bgr = results[0].plot() 
                
                annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

                with col2:
                    st.subheader("Hasil Deteksi")
                    st.image(annotated_image_rgb, caption="Gambar dengan deteksi slot parkir.", use_column_width=True)
        else:
            st.error("Model tidak berhasil dimuat, proses tidak dapat dilanjutkan.")

else:
    st.info("Silakan unggah sebuah gambar untuk memulai.")