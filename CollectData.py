import cv2
import mediapipe as mp
import numpy as np
import os

dataset_dir = "ASL_Dataset"
os.makedirs(dataset_dir, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    label = input("Enter the ASL letter (or 'exit' to quit): ").strip().upper()
    if label == 'EXIT':
        break

    label_dir = os.path.join(dataset_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        count = 0
        while count < 500:
            success, frame = capture.read()
            if not success:
                print("Warning: Empty frame received from camera.")
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            white_canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255  

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(white_canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    img_path = os.path.join(label_dir, f"{label}_{count}.png")
                    cv2.imwrite(img_path, white_canvas)
                    count += 1
                    print(f"Saved: {img_path}")

            cv2.imshow("Hand Landmarks", white_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

capture.release()
cv2.destroyAllWindows()
