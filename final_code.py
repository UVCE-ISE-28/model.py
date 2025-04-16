import cv2
import dlib
import numpy as np
import pygame
from tensorflow.keras.models import load_model
from collections import deque

# Initialize pygame for sound alert
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("Path to alarm or use the mp3 file in the same folder")

# Load the trained model
model = load_model("Path of the model")

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"Path to shape predicter file dlib")

# Eye landmark indexes
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Prediction buffer
N = 5
recent_preds = deque(maxlen=N)
alert_playing = False

def extract_eye_image(frame, landmarks, eye_indices):
    x = [landmarks.part(i).x for i in eye_indices]
    y = [landmarks.part(i).y for i in eye_indices]
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    margin = 15  # Increased margin for better cropping
    return frame[max(0, min_y-margin):max_y+margin, max(0, min_x-margin):max_x+margin]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = extract_eye_image(frame, landmarks, LEFT_EYE_IDX)
        right_eye = extract_eye_image(frame, landmarks, RIGHT_EYE_IDX)

        eye_preds = []

        for eye_img in [left_eye, right_eye]:
            if eye_img.size == 0:
                continue
            eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
            eye_resized = cv2.resize(eye_rgb, (224, 224))
            eye_normalized = eye_resized / 255.0
            eye_input = np.expand_dims(eye_normalized, axis=0)

            prediction = model.predict(eye_input, verbose=0)
            eye_preds.append(prediction[0][0])

        if eye_preds:
            avg_conf = sum(eye_preds) / len(eye_preds)
            predicted_class = "Drowsy" if avg_conf > 0.3 else "Non-Drowsy"
            recent_preds.append(predicted_class)

        if recent_preds:
            majority_vote = max(set(recent_preds), key=recent_preds.count)
            label = f"{majority_vote}"
            color = (0, 0, 255) if majority_vote == "Drowsy" else (0, 255, 0)
            cv2.putText(frame, label, (face.left(), face.top()-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)

            if majority_vote == "Drowsy" and not alert_playing:
                alert_sound.play()
                alert_playing = True
            elif majority_vote == "Non-Drowsy" and alert_playing:
                alert_sound.stop()
                alert_playing = False

    cv2.imshow('AutoSense - Drowsiness Detection (dlib)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
