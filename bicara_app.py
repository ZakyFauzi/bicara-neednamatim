import streamlit as st
import cv2
import numpy as np
import whisper
import tempfile
import os
from datetime import datetime

# Inisialisasi Whisper model (ringan: "tiny" untuk kecepatan)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

whisper_model = load_whisper_model()

# Fungsi analisis video
def analyze_video(video_file):
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile_path = tfile.name
        
        cap = cv2.VideoCapture(tfile_path)
        if not cap.isOpened():
            raise ValueError("Gagal membuka video")

        eye_contact_score = 0
        posture_score = 0
        frame_count = 0
        max_duration = 180  # Batas 3 menit
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise ValueError("Gagal memuat classifier wajah")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count % 10 != 0:  # Tingkatkan sampling ke 10 frame
                frame_count += 1
                continue
            if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > max_duration:
                break
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                eye_contact_score += 1
            
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                posture_score += 1
            
            frame_count += 1
        
        cap.release()
        total_frames = frame_count // 10 if frame_count > 0 else 1
        eye_contact_score = (eye_contact_score / total_frames) * 100 if total_frames > 0 else 0
        posture_score = (posture_score / total_frames) * 100 if total_frames > 0 else 0

        # Analisis Audio
        result = whisper_model.transcribe(tfile_path)
        full_text = result["text"].lower()
        words = full_text.split()
        duration = result["segments"][-1]["end"] if result["segments"] else 1
        wpm = len(words) / duration * 60 if duration > 0 else 0
        filler_words = {"eh": 0, "hmm": 0, "anu": 0, "ehm": 0}
        for word in words:
            if word in filler_words:
                filler_words[word] += 1
        
        os.unlink(tfile_path)
        return {
            "eye_contact_score": eye_contact_score,
            "posture_score": posture_score,
            "wpm": wpm,
            "filler_words": filler_words
        }
    except Exception as e:
        st.error(f"Error dalam analisis video: {str(e)}")
        return None

# Fungsi analisis audio
def analyze_audio(audio_file):
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tfile.write(audio_file.read())
        tfile_path = tfile.name
        
        result = whisper_model.transcribe(tfile_path)
        full_text = result["text"].lower()
        words = full_text.split()
        duration = result["segments"][-1]["end"] if result["segments"] else 1
        wpm = len(words) / duration * 60 if duration > 0 else 0
        filler_words = {"eh": 0, "hmm": 0, "anu": 0, "ehm": 0}
        for word in words:
            if word in filler_words:
                filler_words[word] += 1
        
        os.unlink(tfile_path)
        return {"wpm": wpm, "filler_words": filler_words}
    except Exception as e:
        st.error(f"Error dalam analisis audio: {str(e)}")
        return None

# Fungsi chatbot sederhana
def chatbot_response(input_text):
    if "tips" in input_text.lower():
        return "Tips presentasi: Jaga kontak mata, kurangi kata pengisi, dan bicara dengan ritme stabil."
    elif "contoh" in input_text.lower():
        return "Contoh pembukaan: 'Halo semua, terima kasih telah hadir, hari ini saya akan membahas...'"
    else:
        return "Silakan tanyakan tips atau contoh presentasi untuk bantuan lebih lanjut!"

# UI dengan tiga section
st.title("BICARA - Bimbingan Cerdas Retorika Anda")
st.write("Asisten Virtual Berbasis AI untuk Melatih Keterampilan Presentasi")

# Section 1: Chatbot
st.header("1. Diskusi dengan Chatbot")
user_input = st.text_input("Tanyakan tentang presentasi (contoh: tips, contoh):")
if user_input:
    response = chatbot_response(user_input)
    st.write("**Jawaban:**", response)

# Section 2: Upload Audio
st.header("2. Analisis Audio")
audio_file = st.file_uploader("Unggah File Audio (MP3, maks. 3 menit)", type=["mp3"])
if audio_file:
    if st.button("Analisis Audio"):
        with st.spinner("Menganalisis audio, mohon tunggu..."):
            result = analyze_audio(audio_file)
            if result:
                st.success("Analisis audio selesai!")
                st.subheader("Hasil Analisis Audio")
                st.metric("Kecepatan Bicara (WPM)", f"{result['wpm']:.1f}")
                st.subheader("Kata Pengisi")
                st.bar_chart(result['filler_words'])
                st.subheader("Rekomendasi Praktis")
                recommendations = []
                if result['wpm'] > 150:
                    recommendations.append("Coba kurangi kecepatan bicara untuk kejelasan.")
                if max(result['filler_words'].values()) > 0:
                    most_filler = max(result['filler_words'], key=result['filler_words'].get)
                    recommendations.append(f"Kurangi penggunaan kata '{most_filler}' untuk tampil lebih percaya diri.")
                st.write("- " + "\n- ".join(recommendations) if recommendations else "Presentasi audio Anda sudah sangat baik!")

# Section 3: Upload Video
st.header("3. Analisis Video")
video_file = st.file_uploader("Unggah Video Presentasi (MP4/MOV, maks. 3 menit)", type=["mp4", "mov"])
if video_file:
    st.video(video_file)
    if st.button("Analisis Video"):
        with st.spinner("Menganalisis video, mohon tunggu..."):
            result = analyze_video(video_file)
            if result:
                st.success("Analisis video selesai!")
                st.subheader("Hasil Analisis Video")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skor Kontak Mata", f"{result['eye_contact_score']:.1f}%")
                    st.metric("Skor Postur", f"{result['posture_score']:.1f}%")
                with col2:
                    st.metric("Kecepatan Bicara (WPM)", f"{result['wpm']:.1f}")
                st.subheader("Kata Pengisi")
                st.bar_chart(result['filler_words'])
                st.subheader("Rekomendasi Praktis")
                recommendations = []
                if result['wpm'] > 150:
                    recommendations.append("Coba kurangi kecepatan bicara untuk kejelasan.")
                if max(result['filler_words'].values()) > 0:
                    most_filler = max(result['filler_words'], key=result['filler_words'].get)
                    recommendations.append(f"Kurangi penggunaan kata '{most_filler}' untuk tampil lebih percaya diri.")
                if result['eye_contact_score'] < 50:
                    recommendations.append("Pertahankan kontak mata lebih lama ke audiens.")
                st.write("- " + "\n- ".join(recommendations) if recommendations else "Presentasi video Anda sudah sangat baik!")

st.markdown("**Catatan:** Pastikan file memiliki pencahayaan cukup (untuk video), resolusi minimal 720p, dan durasi maksimal 3 menit untuk hasil optimal.")
