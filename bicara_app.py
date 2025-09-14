import streamlit as st
import cv2
import numpy as np
import whisper
import tempfile
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
from mtcnn import MTCNN

# ===============================
# Konfigurasi API Gemini
# Ganti dengan API key kamu langsung (hardcode)
# ===============================
API_KEY = "AIzaSyDul_w9C1brfAq2ujvh_mLY-EyTnTHq5Ro"
genai.configure(api_key=API_KEY)

# --- FUNGSI CACHE & LOAD MODEL ---
@st.cache_resource(ttl=3600)
def load_whisper_model():
    st.write("Memuat model Whisper...")
    model = whisper.load_model("tiny", device="cpu")
    return model

@st.cache_resource(ttl=3600)
def load_mtcnn_model():
    st.write("Memuat model deteksi wajah MTCNN...")
    detector = MTCNN()
    return detector

whisper_model = load_whisper_model()
mtcnn_detector = load_mtcnn_model()

# --- FUNGSI ANALISIS UTAMA ---
def analyze_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Gagal membuka video dari path")

        face_detected_frames = 0
        processed_frame_count = 0
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps if fps > 0 else 1 # Proses 1 frame per detik

        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = mtcnn_detector.detect_faces(frame_rgb)
                
                if len(faces) > 0 and faces[0]['confidence'] > 0.90:
                    face_detected_frames += 1
                processed_frame_count += 1
            
            current_frame += 1

        cap.release()
        eye_contact_score = (face_detected_frames / processed_frame_count) * 100 if processed_frame_count > 0 else 0
        return {"eye_contact_score": eye_contact_score}
    except Exception as e:
        st.error(f"Error dalam analisis video: {str(e)}")
        return None

def analyze_audio(file_path):
    try:
        result = whisper_model.transcribe(file_path, language="id")
        full_text = result.get("text", "").lower()
        words = full_text.split()
        
        duration_seconds = result["segments"][-1]["end"] if result.get("segments") else 1
        duration_minutes = duration_seconds / 60.0
        
        wpm = len(words) / duration_minutes if duration_minutes > 0 else 0
        
        filler_words_count = {"eh": 0, "hmm": 0, "anu": 0, "ehm": 0, "ya": 0, "lah": 0}
        for word in words:
            if word in filler_words_count:
                filler_words_count[word] += 1
        
        return {"wpm": wpm, "filler_words": filler_words_count, "transcript": result.get("text", "Gagal mentranskripsi audio.")}
    except Exception as e:
        st.error(f"Error dalam analisis audio: {str(e)}")
        return None

@st.cache_data(ttl=300)
def analyze_media_cached(_file_bytes, file_type):
    suffix = ".mp4" if file_type == 'video' else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(_file_bytes)
        tfile_path = tfile.name

    video_result = None
    if file_type == 'video':
        video_result = analyze_video(tfile_path)
    
    audio_result = analyze_audio(tfile_path)

    os.unlink(tfile_path)
    return video_result, audio_result

# --- FUNGSI CHATBOT ---
def chatbot_response(input_text):
    if not input_text or not any(keyword in input_text.lower() for keyword in ["presentasi", "tips", "struktur", "kecemasan", "bicara", "pengantar"]):
        return "Maaf, chatbot hanya mendukung diskusi seputar presentasi. Coba tanyakan tips presentasi atau cara mengatasi kecemasan."
    
    prompt = f"Sebagai seorang ahli public speaking di Indonesia, berikan jawaban singkat dan praktis (maksimal 3-4 kalimat) mengenai: '{input_text}'. Gunakan bahasa yang memotivasi dan mudah dimengerti."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Terjadi kesalahan saat menghubungi AI. Coba lagi nanti. Error: {str(e)}"

# --- UI STREAMLIT ---
st.title("BICARA - Bimbingan Cerdas Retorika Anda")
st.write("Asisten Virtual Berbasis AI untuk Masyarakat Indonesia dalam Melatih Keterampilan Presentasi")

# Section 1: Chatbot
st.header("1. Diskusi dengan Chatbot")
user_input = st.text_input("Tanyakan tentang presentasi (contoh: tips pembukaan, cara atasi grogi):", key="chatbot_input")
if user_input:
    response = chatbot_response(user_input)
    st.write("**Jawaban:**", response)

# Section 2: Upload Audio
st.header("2. Analisis Audio (Hanya Vokal)")
audio_file = st.file_uploader("Unggah File Audio (MP3, WAV, M4A maks. 10 MB)", type=["mp3", "wav", "m4a"])
if audio_file:
    if st.button("Analisis Audio"):
        audio_bytes = audio_file.getvalue()
        with st.spinner("Menganalisis audio, mohon tunggu..."):
            start_time = time.time()
            _, result = analyze_media_cached(audio_bytes, 'audio')
            st.write(f"Waktu analisis: {time.time() - start_time:.2f} detik")
            if result:
                st.success("Analisis audio selesai!")
                st.subheader("Hasil Analisis Audio")
                st.metric("Kecepatan Bicara (WPM)", f"{result['wpm']:.1f}")
                st.subheader("Kata Pengisi")
                st.bar_chart(result['filler_words'])
                st.subheader("Transkrip Teks")
                st.info(result['transcript'])


# Section 3: Upload Video
st.header("3. Analisis Video Lengkap (Vokal & Visual)")
video_file = st.file_uploader("Unggah Video Presentasi (MP4, MOV maks. 25 MB)", type=["mp4", "mov"])
if video_file:
    video_bytes = video_file.getvalue()
    st.video(video_bytes)
    if st.button("Analisis Video"):
        with st.spinner("Menganalisis video, ini mungkin butuh waktu lebih lama..."):
            start_time = time.time()
            video_result, audio_result = analyze_media_cached(video_bytes, 'video')
            st.write(f"Waktu analisis: {time.time() - start_time:.2f} detik")
            if video_result and audio_result:
                st.success("Analisis video selesai!")
                st.subheader("Hasil Analisis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skor Kontak Mata (Wajah Terdeteksi)", f"{video_result['eye_contact_score']:.1f}%")
                with col2:
                    st.metric("Kecepatan Bicara (WPM)", f"{audio_result['wpm']:.1f}")
                
                st.subheader("Kata Pengisi")
                st.bar_chart(audio_result['filler_words'])
                
                st.subheader("Transkrip Teks")
                st.info(audio_result['transcript'])
                
                st.subheader("Rekomendasi Praktis")
                recommendations = []
                if audio_result['wpm'] > 160:
                    recommendations.append("Kecepatan bicara Anda agak tinggi. Coba ambil jeda sejenak antar kalimat agar audiens lebih mudah mengikuti.")
                elif audio_result['wpm'] < 110:
                     recommendations.append("Kecepatan bicara Anda agak lambat. Coba tingkatkan sedikit antusiasme agar audiens tetap terlibat.")
                
                if sum(audio_result['filler_words'].values()) > 5:
                    most_filler = max(audio_result['filler_words'], key=audio_result['filler_words'].get)
                    recommendations.append(f"Anda cukup sering menggunakan kata pengisi, terutama '{most_filler}'. Latih kesadaran untuk mengurangi penggunaannya.")
                
                if video_result['eye_contact_score'] < 60:
                    recommendations.append("Skor kontak mata Anda masih bisa ditingkatkan. Pastikan wajah selalu menghadap ke depan/kamera.")
                
                st.write("- " + "\n- ".join(recommendations) if recommendations else "Secara keseluruhan, presentasi Anda sudah sangat baik dari metrik yang diukur!")

st.markdown("**Catatan:** Untuk hasil optimal, pastikan video/audio memiliki suara yang jelas dan durasi tidak lebih dari 3 menit.")
