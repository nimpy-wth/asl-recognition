import cv2
import mediapipe as mp
import numpy as np
import os
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

csv_file = os.path.join(dataset_dir, "landmarks_data.csv")
csv_header = ["label"]
for i in range(21):
    csv_header.extend([f"x{i}", f"y{i}", f"z{i}"])

if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

while True:
    label = input("Enter the ASL letter (or 'exit' to quit): ").strip().upper()
    if label == 'EXIT':
        break

    label_dir = os.path.join(dataset_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Could not open camera.")
        exit()

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

                    with open(csv_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        row = [label]
                        for landmark in hand_landmarks.landmark:
                            row.extend([landmark.x, landmark.y, landmark.z])
                        writer.writerow(row)

            cv2.imshow("Hand Landmarks", white_canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
    print(f"Data collection for {label} is complete.")

print("Data collection is complete!")
