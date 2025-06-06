import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
    page_title="Deteksi Slot Parkir",
    page_icon="🚗",
    layout="wide"
)

st.title("🅿️ Deteksi Status Slot Parkir")
st.write(
    "Unggah gambar area parkir untuk mendeteksi slot yang kosong (empty) dan terisi (filled). "
)

MODEL_PATH = 'best.pt'


@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def detect_and_draw_on_image(
    input_image: Image.Image, 
    model: YOLO,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> (np.ndarray, dict):
    
    results = model.predict(
        source=input_image,
        conf=conf_threshold,
        iou=iou_threshold
    )
    

    annotated_image_bgr = np.array(input_image.convert('RGB'))
    annotated_image_bgr = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_RGB2BGR)

    stats = {'empty': 0, 'filled': 0}
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])

        if class_id == 1: # 1: filled
            color = (0, 0, 255) 
            stats['filled'] += 1
        else: # 0: empty
            color = (0, 255, 0) 
            stats['empty'] += 1
            
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        
        cv2.rectangle(
            img=annotated_image_bgr, 
            pt1=(x1, y1), 
            pt2=(x2, y2), 
            color=color, 
            thickness=1 
        )


    stats['total'] = stats['empty'] + stats['filled']

    def draw_text_with_background(img, text, position, font, scale, text_color, bg_color, thickness=2, padding=10, alpha=0.6):
        overlay = img.copy()
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size
        x, y = position
        cv2.rectangle(overlay, (x - padding, y - text_h - padding), (x + text_w + padding, y + padding), bg_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_with_background(annotated_image_bgr, f"Filled Spots: {stats['filled']}", (30, 50), font, 1, (0, 0, 255), (255,255,255))
    draw_text_with_background(annotated_image_bgr, f"Empty Spots: {stats['empty']}", (30, 100), font, 1, (0, 255, 0), (255,255,255))
    draw_text_with_background(annotated_image_bgr, f"Total Spots: {stats['total']}", (30, 150), font, 1, (0,0,0), (255,255,255))

    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

    return annotated_image_rgb, stats

model = load_model(MODEL_PATH)

if 'final_image' not in st.session_state:
    st.session_state.final_image = None

if 'stats' not in st.session_state:
    st.session_state.stats = None

# UI untuk unggah file
uploaded_file = st.file_uploader(
    "Pilih sebuah gambar...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2, col3, col4, col5 = st.columns([1, 3.5, 1, 3.5, 1])
    col123 = st.columns(2)
    col11,col12 = st.columns([1, 1])

    with col2:
        header_col, button_col = st.columns([3, 1])

        with header_col:
            st.subheader("Gambar Asli")

        with button_col:
            st.write("") 
            start_detection = st.button("Deteksi!", use_container_width=True)

        st.image(image, caption="Gambar yang Anda unggah.", use_column_width=True)

    if start_detection:
        if model is not None:
            # Tampilkan spinner di kolom tengah saat memproses
            with col3:
                with st.spinner('Memproses...'):
                    # Panggil fungsi deteksi
                    final_image, stats = detect_and_draw_on_image(
                        input_image=image, 
                        model=model,
                        conf_threshold=0.25
                    )
                    st.session_state.final_image = final_image
                    st.session_state.stats = stats
        else:
            st.error("Model tidak berhasil dimuat.")

    if st.session_state.final_image is not None:
        with col4:
            st.subheader("Hasil Deteksi")
            st.image(st.session_state.final_image, caption="Gambar dengan deteksi.", use_column_width=True)
            
            st.divider()
            stats = st.session_state.stats
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Total Slot", value=stats['total'])
            with m_col2:
                st.metric(label="Terisi", value=stats['filled'])
            with m_col3:
                st.metric(label="Kosong", value=stats['empty'])
    else:
        st.info("Silakan upload gambar untuk memulai.")

else:
    st.info("Silakan unggah sebuah gambar untuk memulai.")