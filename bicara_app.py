import streamlit as st
import cv2
import numpy as np
import whisper
import tempfile
import os
import google.generativeai as genai
import time
import requests
from dotenv import load_dotenv

st.set_page_config(
    page_title="BICARA AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))


# --- Fungsi Helper untuk Tampilan (UI/UX) ---
def render_gauge(label, value, max_value, unit):
    """Membuat gauge chart sederhana menggunakan HTML/SVG."""
    percent_value = (value / max_value) * 100 if max_value > 0 else 0
    
    if percent_value < 40: color = "#FF4B4B"  # Merah
    elif percent_value < 70: color = "#FFFD80"  # Kuning
    else: color = "#28A745"  # Hijau

    gauge_html = f"""
    <div style="text-align: center; background-color: #0E1117; padding: 10px; border-radius: 10px; height: 100%;">
        <svg width="150" height="80">
            <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#FF4B4B" /><stop offset="50%" stop-color="#FFFD80" /><stop offset="100%" stop-color="#28A745" />
                </linearGradient>
            </defs>
            <path d="M 10 70 A 65 65 0 0 1 140 70" stroke="#444444" stroke-width="15" fill="none"></path>
            <path d="M 10 70 A 65 65 0 0 1 140 70" stroke="url(#gradient)" stroke-width="15" fill="none" 
                  stroke-dasharray="210" stroke-dashoffset="{210 - (percent_value / 100 * 210)}"></path>
        </svg>
        <div style="font-size: 1rem; font-weight: bold; color: white;">{label}</div>
        <div style="font-size: 1.3rem; font-weight: bolder; color: {color};">{int(value)}{unit}</div>
    </div>
    """
    st.html(gauge_html)

# --- Fungsi Cache & Load Model ---
@st.cache_resource(ttl=3600)
def load_whisper_model():
    with st.spinner("Memuat model Whisper (hanya sekali)..."):
        model = whisper.load_model("tiny")
    return model

def download_file(url, file_path):
    if not os.path.exists(file_path):
        with st.spinner(f"Mengunduh model deteksi wajah..."):
            r = requests.get(url, allow_redirects=True)
            with open(file_path, 'wb') as f:
                f.write(r.content)

@st.cache_resource(ttl=3600)
def load_face_detector():
    with st.spinner("Memuat model deteksi wajah (hanya sekali)..."):
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        model_path = "face_detection_yunet_2023mar.onnx"
        download_file(model_url, model_path)
        detector = cv2.FaceDetectorYN.create(model_path, "", (0, 0))
    return detector

# --- Fungsi-fungsi Analisis (Backend) ---
def analyze_video(video_path):
    # PERBAIKAN: Menambahkan st.error untuk menampilkan pesan error jika ada
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError("Gagal membuka file video.")
        face_detected_frames, processed_frame_count = 0, 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) if fps > 0 else 1
        current_frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if current_frame_idx % frame_interval == 0:
                h, w, _ = frame.shape
                face_detector.setInputSize((w, h))
                faces = face_detector.detect(frame)
                if faces[1] is not None and faces[1][0][14] > 0.9:
                    face_detected_frames += 1
                processed_frame_count += 1
            current_frame_idx += 1
        cap.release()
        score = (face_detected_frames / processed_frame_count) * 100 if processed_frame_count > 0 else 0
        return {"eye_contact_score": score}
    except Exception as e:
        st.error(f"Error di dalam fungsi analisis video: {e}")
        return None

def analyze_audio(file_path):
    try:
        result = whisper_model.transcribe(file_path, language="id", fp16=False)
        text = result.get("text", "").lower()
        words = text.split()
        duration_seconds = result["segments"][-1]["end"] if result.get("segments") else 1
        duration_minutes = max(1, duration_seconds) / 60.0
        wpm = len(words) / duration_minutes
        filler_words = {"eh": 0, "hmm": 0, "anu": 0, "ehm": 0, "ya": 0, "lah": 0}
        for word in words:
            if word in filler_words: filler_words[word] += 1
        return {"wpm": wpm, "filler_words": filler_words, "transcript": result.get("text", "Gagal mentranskripsi audio.")}
    except Exception as e:
        st.error(f"Error di dalam fungsi analisis audio: {e}")
        return None

@st.cache_data(ttl=300)
def analyze_media_from_bytes(_file_bytes, media_type):
    suffix = ".mp4" if media_type == 'video' else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(_file_bytes)
        temp_path = tfile.name
    
    video_result, audio_result = None, None
    try:
        if media_type == 'video':
            video_result = analyze_video(temp_path)
        audio_result = analyze_audio(temp_path)
    finally:
        os.unlink(temp_path)
    return video_result, audio_result

# --- Fungsi Chatbot ---
def chatbot_response(input_text):
    keywords = ["presentasi", "tips", "trik", "audiens", "public speaking", "ngomong", "belibet", "lantang", "tegas", "keras", "jelas", "struktur", "kecemasan", "bicara", "pengantar", "suara", "grogi", "pembuka", "penutup"]
    if not input_text or not any(keyword in input_text.lower() for keyword in keywords):
        return "Maaf, saya hanya AI Coach untuk topik presentasi. Coba tanyakan tips pembukaan atau cara mengatasi grogi."
    prompt = f"Sebagai seorang ahli public speaking, berikan jawaban singkat dan praktis (maksimal 3-4 kalimat) mengenai: '{input_text}'. Gunakan bahasa yang memotivasi dan mudah dimengerti."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Terjadi kesalahan saat menghubungi AI: {e}"


# --- Sidebar ---
with st.sidebar:
    st.title("🎙️ BICARA AI")
    st.info("Asisten virtual untuk melatih keterampilan presentasi Anda. Proyek untuk LIDM 2025.")
    st.caption("Dikembangkan oleh NeedNamaTim")

# --- Muat model di awal ---
whisper_model = load_whisper_model()
face_detector = load_face_detector()

# --- Navigasi Utama ---
tab1, tab2, tab3 = st.tabs(["**🚀 Analisis Presentasi**", "**💬 Tanya AI Coach**", "**ℹ️ Tentang Proyek**"])

# --- TAB 1: Analisis ---
with tab1:
    st.header("Unggah Media untuk Dianalisis")
    st.markdown("Pilih jenis media. Untuk hasil terbaik, gunakan video/audio dengan suara jernih dan durasi **maksimal 3 menit**.")

    analysis_type_tab1, analysis_type_tab2 = st.tabs(["🎬 **Analisis Video Lengkap**", "🎵 **Analisis Audio Saja**"])

    with analysis_type_tab1:
        video_file = st.file_uploader("Unggah Video Presentasi (MP4, MOV maks. 25 MB)", type=["mp4", "mov"], key="video_uploader")
        if video_file:
            video_bytes = video_file.getvalue()
            st.video(video_bytes)
            # PERBAIKAN: Menambahkan key unik pada tombol
            if st.button("Analisis Video Ini", use_container_width=True, type="primary", key="analyze_video_button"):
                with st.spinner("Menganalisis video... Proses ini mungkin memakan waktu 1-2 menit."):
                    video_result, audio_result = analyze_media_from_bytes(video_bytes, 'video')
                    
                    if video_result and audio_result:
                        st.success("Analisis Selesai!")
                        st.subheader("📊 Dasbor Umpan Balik")
                        with st.container(border=True):
                            col1, col2 = st.columns(2)
                            with col1: render_gauge("Kontak Mata", video_result.get('eye_contact_score', 0), 100, "%")
                            with col2: render_gauge("Kecepatan Bicara", audio_result.get('wpm', 0), 180, " WPM")
                            st.divider()
                            st.markdown("##### 🗣️ Kata Pengisi yang Terdeteksi")
                            st.bar_chart(audio_result.get('filler_words', {}))

                        st.subheader("💡 Rekomendasi Praktis")
                        with st.container(border=True):
                            recommendations = []
                            wpm = audio_result.get('wpm', 0)
                            fillers = audio_result.get('filler_words', {})
                            eye_contact = video_result.get('eye_contact_score', 0)
                            if wpm > 160: recommendations.append("🏃‍♂️ **Kecepatan Bicara Agak Tinggi.** Coba ambil jeda sejenak.")
                            elif wpm < 110 and wpm > 0: recommendations.append("🐢 **Kecepatan Bicara Agak Lambat.** Coba tingkatkan antusiasme.")
                            if sum(fillers.values()) > 5:
                                most_filler = max(fillers, key=fillers.get)
                                recommendations.append(f"🤔 **Kata Pengisi Cukup Banyak.** Anda sering menggunakan '{most_filler}'.")
                            if eye_contact < 60: recommendations.append("👀 **Kontak Mata Bisa Ditingkatkan.** Pastikan wajah selalu menghadap kamera.")
                            
                            if not recommendations: st.success("🎉 **Kerja Bagus!** Metrik utama Anda sudah ideal.")
                            else:
                                for rec in recommendations: st.warning(rec)

                        st.subheader("📄 Transkrip Teks")
                        with st.expander("Lihat Transkrip Lengkap"):
                            st.info(audio_result.get('transcript', 'Transkrip tidak tersedia.'))
    
    with analysis_type_tab2:
        audio_file = st.file_uploader("Unggah File Audio (MP3, WAV, M4A maks. 10 MB)", type=["mp3", "wav", "m4a"], key="audio_uploader")
        if audio_file:
            # PERBAIKAN: Menambahkan key unik pada tombol
            if st.button("Analisis Audio Ini", use_container_width=True, type="primary", key="analyze_audio_button"):
                audio_bytes = audio_file.getvalue()
                with st.spinner("Menganalisis audio..."):
                    _, audio_result = analyze_media_from_bytes(audio_bytes, 'audio')
                    if audio_result:
                        st.success("Analisis Selesai!")
                        st.subheader("📊 Dasbor Umpan Balik Audio")
                        with st.container(border=True):
                            render_gauge("Kecepatan Bicara", audio_result.get('wpm', 0), 180, " WPM")
                            st.divider()
                            st.markdown("##### 🗣️ Kata Pengisi yang Terdeteksi")
                            st.bar_chart(audio_result.get('filler_words', {}))
                        st.subheader("📄 Transkrip Teks")
                        with st.expander("Lihat Transkrip Lengkap"):
                            st.info(audio_result.get('transcript', 'Transkrip tidak tersedia.'))

# --- TAB 2: Tanya AI Coach ---
with tab2:
    st.header("Tanya AI Coach")
    st.markdown("Punya pertanyaan seputar *public speaking*? Tanyakan pada AI Coach kami.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait presentasi Anda?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tulis pertanyaan Anda di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI Coach sedang berpikir..."):
                response = chatbot_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- TAB 3: Tentang Proyek ---
with tab3:
    st.header("Tentang Proyek BICARA")
    # ... (Isi bagian ini tidak diubah)
    st.markdown("""
    **BICARA (Bimbingan Cerdas Retorika Anda)** adalah purwarupa fungsional yang dikembangkan sebagai bagian dari Lomba Inovasi Digital Mahasiswa (LIDM) 2025.
    #### **Tujuan Utama**
    Proyek ini bertujuan untuk menyediakan alat bantu yang dapat diakses oleh semua kalangan pelajar untuk melatih keterampilan presentasi secara mandiri.
    #### **Teknologi yang Digunakan**
    - **Framework Aplikasi:** Streamlit
    - **Analisis Visual:** OpenCV (FaceDetectorYN - YuNet)
    - **Analisis Vokal:** OpenAI Whisper (Model 'tiny')
    - **AI Chatbot:** Google Gemini (Model 'gemini-1.5-flash')
    #### **Tim Pengembang**
    - Zaky Muhammad Fauzi, Syauqi Gathan Setyapratama, Ghifary Wibisono, Jondri (Dosen Pembimbing)
    """)
