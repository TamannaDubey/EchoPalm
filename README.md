# 🧠 EchoPlam – AI-Powered Sign Language to Speech Dashboard

**EchoPlam** is a real-time accessibility dashboard that detects sign language gestures using a Convolutional Neural Network (CNN) and translates them into readable text and spoken audio. Designed to assist individuals with speech and hearing impairments, the app combines gesture recognition, emotion detection, vitals simulation, and speech translation into a unified, user-friendly interface.

---

## 📖 Project Description

EchoPlam captures hand gestures via webcam, preprocesses the frames, and classifies them into alphabet signs (A–Y) using a trained CNN model deployed with TensorFlow Lite. The predicted sign is displayed on-screen and spoken aloud using pyttsx3. To enhance stability, a rolling buffer smooths predictions over time.

The app also includes emotion detection from facial expressions, simulated vitals (heart rate and oxygen level), and a Hindi-to-English speech translation module. Users can select their microphone, view live mic status, and follow a two-step flow — “Start Listening” and “Submit” — for better control. The translated speech is displayed and spoken aloud, bridging communication between signers and non-signers.

EchoPlam is built with modularity and scalability in mind, making it suitable for classrooms, clinics, and public service environments. It showcases practical AI integration, real-time computer vision, and inclusive design.

---

## 📌 Key Features

- ✋ Real-time sign language detection (A–Y)
- 🧠 CNN model deployed with TensorFlow Lite
- 🔊 Text-to-speech output using pyttsx3
- 🎧 Mic selector with live status (ON/OFF/Not Found)
- 🎤 Hindi-to-English speech translation with two-step flow
- 😊 Emotion detection from facial expressions
- ❤️ Simulated vitals: heart rate and oxygen level
- 🔄 Prediction buffer for stable output
- 🧾 Modular design for future expansion

---

## 🧪 Technical Architecture

### 🔍 Sign Detection Pipeline
- **Input**: Webcam frames via OpenCV and streamlit-webrtc
- **Preprocessing**: Resize, grayscale, normalize frames
- **Model**: CNN trained on static hand gestures (A–Y)
- **Deployment**: TensorFlow Lite (.tflite) for fast inference
- **Buffering**: `deque` used to smooth predictions
- **Output**: Text + speech via pyttsx3

### 🗣️ Speech Translation Module
- **Input**: Hindi speech via selected microphone
- **Processing**: SpeechRecognition + Googletrans
- **Output**: Translated English text + spoken audio

### 😊 Emotion & Vitals
- **Emotion Detection**: Image-based classifier
- **Vitals Simulation**: Randomized heart rate and oxygen level

---

## 🧰 Tech Stack

| Layer         | Tools Used                          |
|---------------|-------------------------------------|
| UI            | Streamlit                           |
| Gesture Input | OpenCV, streamlit-webrtc            |
| Sign Detection| TensorFlow Lite (CNN)               |
| Emotion       | Custom image classifier             |
| Audio Input   | SpeechRecognition, PyAudio          |
| Translation   | Googletrans                         |
| Voice Output  | pyttsx3                             |
| Language      | Python 3.9+                         |

---

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/EchoPlam.git
cd EchoPlam

