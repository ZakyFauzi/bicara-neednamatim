import streamlit as st
import cv2
import numpy as np
from faster_whisper import WhisperModel
import asyncio
import tempfile
import os

# Inisialisasi Whisper untuk transkripsi audio
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Fungsi analisis video menggunakan OpenCV untuk visual dan Whisper untuk audio
async def analyze_video(video_file):
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            raise ValueError("Gagal membuka video")

        eye_contact_score = 0
        posture_score = 0
        frame_count = 0
        max_duration = 180  # Batas maksimal 3 menit dalam detik
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise ValueError("Gagal memuat classifier wajah")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count % 5 != 0:  # Sampling frame
                frame_count += 1
                continue
            if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > max_duration:
                break
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                # Estimasi sederhana kontak mata berdasarkan deteksi wajah
                eye_contact_score += 1
            
            # Estimasi sederhana postur berdasarkan ukuran frame (placeholder)
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                posture_score += 1
            
            frame_count += 1
        
        cap.release()
        total_frames = frame_count // 5 if frame_count > 0 else 1
        eye_contact_score = (eye_contact_score / total_frames) * 100 if total_frames > 0 else 0
        posture_score = (posture_score / total_frames) * 100 if total_frames > 0 else 0

        # Analisis Audio dengan Whisper
        segments, _ = whisper_model.transcribe(tfile.name, beam_size=1)
        wpm = 0
        filler_words = {"eh": 0, "hmm": 0, "anu": 0, "ehm": 0}
        for segment in segments:
            text = segment.text.lower()
            words = text.split()
            wpm += len(words) / (segment.end - segment.start) * 60
            for word in words:
                if word in filler_words:
                    filler_words[word] += 1
        
        wpm = wpm / len(segments) if len(segments) > 0 else 0
        
        os.unlink(tfile.name)
        return {
            "eye_contact_score": eye_contact_score,
            "posture_score": posture_score,
            "wpm": wpm,
            "filler_words": filler_words
        }
    except Exception as e:
        st.error(f"Error dalam analisis video: {str(e)}")
        return None

# UI Streamlit sesuai proposal
st.title("BICARA - Bimbingan Cerdas Retorika Anda")
st.write("Asisten Virtual Berbasis AI untuk Melatih Keterampilan Presentasi")

st.sidebar.header("Panduan Pengguna")
st.sidebar.write("1. Unggah video presentasi (maks. 3 menit, format MP4/MOV).")
st.sidebar.write("2. Klik 'Analisis Presentasi' untuk memproses.")
st.sidebar.write("3. Lihat hasil dan rekomendasi di dasbor.")

uploaded_file = st.file_uploader("Unggah Video Presentasi", type=["mp4", "mov"])
if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button("Analisis Presentasi"):
        with st.spinner("Menganalisis video, mohon tunggu..."):
            result = asyncio.run(analyze_video(uploaded_file))
            if result:
                st.success("Analisis selesai!")
                
                # Dasbor Hasil sesuai arsitektur output layer
                st.subheader("Hasil Analisis")
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
                st.write("- " + "\n- ".join(recommendations) if recommendations else "Presentasi Anda sudah sangat baik!")

st.markdown("**Catatan:** Pastikan video memiliki pencahayaan cukup, resolusi minimal 720p, dan durasi maksimal 3 menit untuk hasil optimal.")
