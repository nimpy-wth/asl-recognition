import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder
from collections import deque

# Load trained CNN model
model = load_model("cnn_hand_landmark_model.h5")

# Extract number of output classes dynamically
num_classes = model.output_shape[-1]

# Define class labels dynamically
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") 
class_labels.extend(["CONFIRM", "DELETE", "SPACE"])

# Convert to LabelEncoder format
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Start Webcam
cap = cv2.VideoCapture(0)  

# Stores detected letters/words
detected_text = "" 

# Stores confirmed words
confirmed_words = []

previous_letter = None 
last_hand_landmarks = None
repeat_cooldown = 20 
last_repeat_frame = 0

last_space_time = 0 
last_delete_time = 0 

# Store last N predictions
smoothing_window = 5  
prediction_queue = deque(maxlen=smoothing_window)

frame_counter = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Extract hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract (x, y, z) coordinates of 21 hand landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Convert landmarks into a NumPy array & reshape for CNN
            landmarks = landmarks.reshape(1, 21, 3, 1)

            # Predict sign letter
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]  

            # Add prediction to smoothing queue
            prediction_queue.append(predicted_label)

            # Apply smoothing: Find most frequent label in the queue
            if len(prediction_queue) == smoothing_window:
                smoothed_prediction = max(set(prediction_queue), key=prediction_queue.count) 
                
                frame_counter += 1  
                
                # SPACE: Only add one space per detection
                if smoothed_prediction == "SPACE":
                    if frame_counter - last_space_time > 15:
                        detected_text += " "
                        last_space_time = frame_counter
                        previous_letter = None

                # DELETE: Remove one letter per detection, slow down deletion
                elif smoothed_prediction == "DELETE":
                    if frame_counter - last_delete_time > 15: 
                        detected_text = detected_text[:-1]  
                        last_delete_time = frame_counter
                        previous_letter = None  

                # CONFIRM: Add full word to list
                elif smoothed_prediction == "CONFIRM":
                    if detected_text.strip(): 
                        confirmed_words.append(detected_text.strip()) 
                        detected_text = "" 
                        previous_letter = None

                # Allow repeated letter only if:
                # 1. Hand has changed shape significantly
                # 2. Cooldown has passed
                elif smoothed_prediction != previous_letter or (last_hand_landmarks is not None 
                                                                and np.linalg.norm(landmarks.flatten() - last_hand_landmarks.flatten()) > 0.05 
                                                                and frame_counter - last_repeat_frame > repeat_cooldown):
                    detected_text += smoothed_prediction
                    previous_letter = smoothed_prediction  
                    last_hand_landmarks = landmarks 
                    last_repeat_frame = frame_counter 

                cv2.putText(frame, f"Prediction: {smoothed_prediction}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Detected Text: {detected_text}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(frame, f"Words: {' '.join(confirmed_words)}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show Webcam Output
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
