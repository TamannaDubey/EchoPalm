import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False)

def get_hand_bbox(landmarks, shape):
    h, w, _ = shape
    x_coords = [int(lm.x * w) for lm in landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max - x_min, y_max - y_min

def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    hand_landmarks_list = result.multi_hand_landmarks

    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            x, y, w, h = get_hand_bbox(hand_landmarks, frame.shape)
            hand_img = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            normalized = resized / 255.0
            return normalized.reshape(1, 28, 28, 1).astype(np.float32)

    return np.zeros((1, 28, 28, 1), dtype=np.float32)

def detect_emotion(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if result.multi_face_landmarks:
        return "Neutral"  # Placeholder
    return "No face detected"