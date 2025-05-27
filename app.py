import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class MediaPipeProcessor(VideoProcessorBase):
    def __init__(self):
        # MediaPipe Hands dengan konfigurasi real-time
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Ambil frame dari webrtc (format BGR)
        img = frame.to_ndarray(format="bgr24")

        # 2. Konversi BGR ke RGB (MediaPipe memproses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Proses frame dengan MediaPipe Hands
        results = self.hands.process(img_rgb)

        # 4. Jika ada tangan terdeteksi, gambar landmark pada frame asli (BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 5. Kembalikan frame yang sudah diolah ke WebRTC (format BGR)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Integrasi streamlit-webrtc + OpenCV + MediaPipe")

    webrtc_streamer(
        key="mediapipe-hands",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=MediaPipeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
