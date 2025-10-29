import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from utils.preprocessing import preprocess_frame, detect_emotion
import pyttsx3
import random
import speech_recognition as sr
from googletrans import Translator
import pyaudio

# ------------------ Load TFLite Model ------------------
interpreter = tf.lite.Interpreter(model_path="model/sign_language_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

prediction_buffer = deque(maxlen=5)
if "captured_frame" not in st.session_state:
    st.session_state.captured_frame = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

# ------------------ Mic Device Listing ------------------
def list_microphones():
    audio = pyaudio.PyAudio()
    devices = []
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices.append((i, info['name']))
    audio.terminate()
    return devices

# ------------------ Mic Status Check ------------------
def check_mic_status(device_index):
    try:
        audio = pyaudio.PyAudio()
        info = audio.get_device_info_by_index(device_index)
        audio.terminate()
        if info['maxInputChannels'] > 0:
            return "✅ Mic is ON"
        else:
            return "⚠️ Mic is OFF"
    except Exception:
        return "❌ Mic Not Found"

# ------------------ Speech Capture + Translation ------------------
def start_listening(device_index):
    r = sr.Recognizer()
    try:
        with sr.Microphone(device_index=device_index) as source:
            st.info("🎤 Listening... Speak now")
            audio = r.listen(source, timeout=5)
            st.session_state.audio_data = audio
            st.success("✅ Speech captured. Click Submit to translate.")
    except Exception:
        st.session_state.audio_data = None
        st.error("❌ Mic error or no speech detected.")

def translate_audio():
    r = sr.Recognizer()
    try:
        text = r.recognize_google(st.session_state.audio_data, language="hi-IN")
        translated = Translator().translate(text, src='hi', dest='en').text
        return text, translated
    except Exception:
        return "Could not understand", "Translation failed"

# ------------------ Other Modules ------------------
class SignPredictor(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img.copy()
        return img

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def get_vitals():
    heart_rate = random.randint(70, 100)
    oxygen = random.randint(95, 100)
    return heart_rate, oxygen

# ------------------ Streamlit Layout ------------------
st.set_page_config(layout="wide")
st.title("🧠 Unified Accessibility Dashboard")

# Webcam stream
ctx = webrtc_streamer(
    key="sign",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=SignPredictor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Capture button
if st.button("📸 Capture"):
    if ctx.video_transformer and ctx.video_transformer.frame is not None:
        st.session_state.captured_frame = ctx.video_transformer.frame.copy()
    else:
        st.warning("Webcam frame not available yet. Try again in a few seconds.")

# Sign prediction + vitals + emotion
if st.session_state.captured_frame is not None:
    frame = st.session_state.captured_frame
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = int(np.argmax(output_data))
    prediction_buffer.append(predicted_label)

    stable_label = max(set(prediction_buffer), key=prediction_buffer.count)
    predicted_letter = label_map.get(stable_label, '?')
    emotion = detect_emotion(frame)
    hr, oxy = get_vitals()

    col1, col2 = st.columns(2)
    with col1:
        st.image(frame, caption=f"Sign: {predicted_letter}", channels="BGR")
        st.text(f"Emotion: {emotion}")
        st.button("🔊 Speak Sign", on_click=lambda: speak_text(predicted_letter))
    with col2:
        st.metric("❤️ Heart Rate", f"{hr} bpm")
        st.metric("🫁 Oxygen Level", f"{oxy}%")

# ------------------ Mic Device Selector ------------------
st.subheader("🎧 Select Input Device")

if st.button("🔄 Refresh Devices"):
    st.session_state.devices = list_microphones()

devices = st.session_state.get("devices", list_microphones())
device_names = [name for i, name in devices]

if device_names:
    selected_device = st.selectbox("Available Microphones", device_names)
    device_index = next(i for i, name in devices if name == selected_device)
    st.markdown(f"**Mic Status:** {check_mic_status(device_index)}")

    st.subheader("🎙️ Speech Translation")

    if st.button("🎤 Start Listening"):
        start_listening(device_index)

    if st.session_state.audio_data is not None:
        if st.button("📤 Submit for Translation"):
            original, translated = translate_audio()
            st.write(f"🗣️ You said: {original}")
            st.write(f"🌐 Translated: {translated}")
else:
    st.error("No microphone devices found. Please plug in a mic and refresh.")