import streamlit as st
import cv2
import numpy as np
import whisper
import tempfile
import os
import google.generativeai as genai
import time

# ===============================
# Konfigurasi API Gemini
# Ganti dengan API key kamu langsung (hardcode)
# ===============================
API_KEY = "AIzaSyDul_w9C1brfAq2ujvh_mLY-EyTnTHq5Ro"
genai.configure(api_key=API_KEY)

# Preload model Whisper multibahasa (tiny untuk Indonesia)
@st.cache_resource(ttl=3600)  # Cache selama 1 jam
def load_whisper_model():
    st.write("Memuat model Whisper multibahasa... (ini hanya sekali di awal)")
    start_time = time.time()
    model = whisper.load_model("tiny", device="cpu", compute_type="int8", fp16=False)  # Optimasi CPU
    st.write(f"Model dimuat dalam {time.time() - start_time:.2f} detik.")
    return model

whisper_model = load_whisper_model()

# Cache hasil analisis untuk video yang sama
@st.cache_data(ttl=300)  # Cache 5 menit
def analyze_video_cached(video_file):
    return analyze_video(video_file), analyze_audio_from_video(video_file)

# Fungsi analisis video (fokus kontak mata)
def analyze_video(video_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(video_file.read())
            tfile_path = tfile.name
        
        cap = cv2.VideoCapture(tfile_path)
        if not cap.isOpened():
            raise ValueError("Gagal membuka video")

        eye_contact_score = 0
        frame_count = 0
        max_duration = 180
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise ValueError("Gagal memuat classifier wajah")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count % 50 != 0:  # Sampling 50 frame
                frame_count += 1
                continue
            if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > max_duration:
                break
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3)
            if len(faces) > 0:
                eye_contact_score += 1
            
            frame_count += 1
        
        cap.release()
        total_frames = frame_count // 50 if frame_count > 0 else 1
        eye_contact_score = (eye_contact_score / total_frames) * 100 if total_frames > 0 else 0

        return {"eye_contact_score": eye_contact_score}
    except Exception as e:
        st.error(f"Error dalam analisis video: {str(e)}")
        return None

# Fungsi analisis audio dari video (terpisah untuk paralel)
def analyze_audio_from_video(video_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(video_file.read())
            tfile_path = tfile.name
        
        result = whisper_model.transcribe(tfile_path, language="id", chunk_length_s=2)  # Chunk 2 detik
        full_text = result["text"].lower()
        words = full_text.split()
        duration = result["segments"][-1]["end"] if result["segments"] else 1
        wpm = len(words) / duration * 60 if duration > 0 else 0
        filler_words = {"eh": 0, "hmm": 0, "anu": 0, "ehm": 0, "ya": 0, "lah": 0}
        for word in words:
            if word in filler_words:
                filler_words[word] += 1
        
        os.unlink(tfile_path)
        return {"wpm": wpm, "filler_words": filler_words}
    except Exception as e:
        st.error(f"Error dalam analisis audio: {str(e)}")
        return None

# Fungsi analisis audio (standalone)
def analyze_audio(audio_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tfile:
            tfile.write(audio_file.read())
            tfile_path = tfile.name
        
        result = whisper_model.transcribe(tfile_path, language="id", chunk_length_s=2)
        full_text = result["text"].lower()
        words = full_text.split()
        duration = result["segments"][-1]["end"] if result["segments"] else 1
        wpm = len(words) / duration * 60 if duration > 0 else 0
        filler_words = {"eh": 0, "hmm": 0, "anu": 0, "ehm": 0, "ya": 0, "lah": 0}
        for word in words:
            if word in filler_words:
                filler_words[word] += 1
        
        os.unlink(tfile_path)
        return {"wpm": wpm, "filler_words": filler_words}
    except Exception as e:
        st.error(f"Error dalam analisis audio: {str(e)}")
        return None

# Fungsi chatbot dengan Gemini
def chatbot_response(input_text):
    if not input_text or not any(keyword in input_text.lower() for keyword in ["presentasi", "tips", "struktur", "kecemasan", "bicara", "pengantar"]):
        return "Maaf, chatbot hanya mendukung diskusi seputar presentasi. Coba tanyakan tips presentasi, struktur, atau cara mengatasi kecemasan."
    
    prompt = f"Berikan jawaban singkat dan informatif tentang presentasi dalam konteks Indonesia terkait: {input_text}. Fokus pada tips, struktur, atau pengelolaan kecemasan, gunakan bahasa yang relevan untuk presenter lokal."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}. Coba lagi nanti."

# UI dengan tiga section
st.title("BICARA - Bimbingan Cerdas Retorika Anda")
st.write("Asisten Virtual Berbasis AI untuk Masyarakat Indonesia dalam Melatih Keterampilan Presentasi")

# Section 1: Chatbot
st.header("1. Diskusi dengan Chatbot")
user_input = st.text_input("Tanyakan tentang presentasi (contoh: tips pembukaan, cara atasi grogi):")
if user_input:
    response = chatbot_response(user_input)
    st.write("**Jawaban:**", response)

# Section 2: Upload Audio
st.header("2. Analisis Audio")
audio_file = st.file_uploader("Unggah File Audio (MP3, maks. 3 menit)", type=["mp3"])
if audio_file:
    if st.button("Analisis Audio"):
        with st.spinner("Menganalisis audio, mohon tunggu..."):
            start_time = time.time()
            result = analyze_audio(audio_file)
            st.write(f"Waktu analisis: {time.time() - start_time:.2f} detik")
            if result:
                st.success("Analisis audio selesai!")
                st.subheader("Hasil Analisis Audio")
                st.metric("Kecepatan Bicara (WPM)", f"{result['wpm']:.1f}")
                st.subheader("Kata Pengisi")
                st.bar_chart(result['filler_words'])
                st.subheader("Rekomendasi Praktis")
                recommendations = []
                if result['wpm'] > 150:
                    recommendations.append("Coba kurangi kecepatan bicara agar lebih jelas.")
                if max(result['filler_words'].values()) > 0:
                    most_filler = max(result['filler_words'], key=result['filler_words'].get)
                    recommendations.append(f"Kurangi penggunaan kata '{most_filler}' untuk kesan profesional.")
                st.write("- " + "\n- ".join(recommendations) if recommendations else "Presentasi audio Anda sudah sangat baik!")

# Section 3: Upload Video
st.header("3. Analisis Video")
video_file = st.file_uploader("Unggah Video Presentasi (MP4/MOV, maks. 3 menit)", type=["mp4", "mov"])
if video_file:
    st.video(video_file)
    if st.button("Analisis Video"):
        with st.spinner("Menganalisis video, mohon tunggu..."):
            start_time = time.time()
            video_result, audio_result = analyze_video_cached(video_file)
            st.write(f"Waktu analisis: {time.time() - start_time:.2f} detik")
            if video_result and audio_result:
                st.success("Analisis video selesai!")
                st.subheader("Hasil Analisis Video")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skor Kontak Mata", f"{video_result['eye_contact_score']:.1f}%")
                with col2:
                    st.metric("Kecepatan Bicara (WPM)", f"{audio_result['wpm']:.1f}")
                st.subheader("Kata Pengisi")
                st.bar_chart(audio_result['filler_words'])
                st.subheader("Rekomendasi Praktis")
                recommendations = []
                if audio_result['wpm'] > 150:
                    recommendations.append("Coba kurangi kecepatan bicara agar lebih jelas.")
                if max(audio_result['filler_words'].values()) > 0:
                    most_filler = max(audio_result['filler_words'], key=audio_result['filler_words'].get)
                    recommendations.append(f"Kurangi penggunaan kata '{most_filler}' untuk kesan profesional.")
                if video_result['eye_contact_score'] < 50:
                    recommendations.append("Pertahankan kontak mata lebih lama untuk terhubung dengan audiens.")
                st.write("- " + "\n- ".join(recommendations) if recommendations else "Presentasi video Anda sudah sangat baik!")

st.markdown("**Catatan:** Pastikan video memiliki pencahayaan cukup, resolusi minimal 720p, dan durasi maksimal 3 menit untuk hasil optimal.")
