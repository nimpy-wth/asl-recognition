import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder
from collections import deque
from PyQt6.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, 
                            QSizePolicy, QCheckBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt


class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet('''
            QCheckBox::indicator {
                width: 60px;  
                height: 40px;
            }
            QCheckBox::indicator:unchecked {
                image: url("toggle_off.png"); 
            }
            QCheckBox::indicator:checked {
                image: url("toggle_on.png"); 
            }
        ''')

class ASLRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        # UI Setup
        self.setWindowTitle("ASL Recognition with PyQt")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Video Display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.video_label, stretch=1)

        # Toggle Switch for Left-Hand Mode
        self.left_hand_mode = False
        self.toggle_switch = ToggleSwitch()
        self.toggle_switch.setChecked(False)
        self.toggle_switch.setText(" Left-Hand Mode")
        self.toggle_switch.toggled.connect(self.on_left_hand_toggled)
        self.layout.addWidget(self.toggle_switch, 0, Qt.AlignmentFlag.AlignHCenter)

        # Detected Text Display
        self.text_label = QLabel("Detected Text: ")
        text_font = self.text_label.font()
        text_font.setPointSize(20)
        self.text_label.setFont(text_font)
        self.layout.addWidget(self.text_label)

        # Confirmed Words Display
        self.words_label = QLabel("Confirmed Words: ")
        words_font = self.words_label.font()
        words_font.setPointSize(20)
        self.words_label.setFont(words_font)
        self.layout.addWidget(self.words_label)

        # Reset Button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_text)
        self.layout.addWidget(self.reset_button)

        # Start Webcam Capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # ASL Processing Variables
        self.detected_text = ""
        self.confirmed_words = []
        self.previous_letter = None
        self.last_hand_landmarks = None
        self.repeat_cooldown = 15
        self.last_repeat_frame = 0
        self.last_space_time = 0
        self.last_delete_time = 0
        self.smoothing_window = 10
        self.prediction_queue = deque(maxlen=self.smoothing_window)
        self.frame_counter = 0

        # Load trained model
        self.model = load_model("cnn_hand_landmark_model.h5")

        # Number of output classes 
        num_classes = self.model.output_shape[-1]

        # Define class labels 
        self.class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.class_labels.extend(["CONFIRM", "DELETE", "SPACE"])

        # Convert to LabelEncoder format
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_labels)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )

    def on_left_hand_toggled(self, checked):
        """Enable or disable left-hand mode based on toggle switch."""
        self.left_hand_mode = checked

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  
        frame_bgr = frame.copy()   

        # Process frame with MediaPipe Hands
        results = self.hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                landmarks_for_bbox = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                landmarks_for_pred = landmarks_for_bbox.copy()
                if self.left_hand_mode:
                    landmarks_for_pred[:, 0] = 1.0 - landmarks_for_pred[:, 0]

                # Compute bounding box
                x_min, y_min = np.min(landmarks_for_bbox[:, :2], axis=0)
                x_max, y_max = np.max(landmarks_for_bbox[:, :2], axis=0)
                h, w, _ = frame_bgr.shape
                x_min, y_min = int(x_min * w), int(y_min * h)
                x_max, y_max = int(x_max * w), int(y_max * h)
                cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Predict gesture
                landmarks_pred_reshaped = landmarks_for_pred.reshape(1, 21, 3, 1)
                prediction = self.model.predict(landmarks_pred_reshaped)
                predicted_class = np.argmax(prediction)
                predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

                self.prediction_queue.append(predicted_label)
                if len(self.prediction_queue) == self.smoothing_window:
                    smoothed_prediction = max(set(self.prediction_queue), key=self.prediction_queue.count)
                    self.frame_counter += 1

                    # Handle Space Gesture
                    if smoothed_prediction == "SPACE":
                        if self.frame_counter - self.last_space_time > 15:
                            self.detected_text += " "
                            self.last_space_time = self.frame_counter
                            self.previous_letter = None

                    # Handle Delete Gesture
                    elif smoothed_prediction == "DELETE":
                        if self.frame_counter - self.last_delete_time > 10:
                            self.detected_text = self.detected_text[:-1]
                            self.last_delete_time = self.frame_counter
                            self.previous_letter = None

                    # Handle Confirm Gesture
                    elif smoothed_prediction == "CONFIRM":
                        if self.detected_text.strip():
                            self.confirmed_words.append(self.detected_text.strip())
                            self.detected_text = ""
                            self.previous_letter = None

                    elif (smoothed_prediction != self.previous_letter 
                            or (self.last_hand_landmarks is not None 
                            and np.linalg.norm(landmarks_pred_reshaped.flatten() - self.last_hand_landmarks.flatten()) > 0.05 
                            and self.frame_counter - self.last_repeat_frame > self.repeat_cooldown)):
                        self.detected_text += smoothed_prediction
                        self.previous_letter = smoothed_prediction
                        self.last_hand_landmarks = landmarks_pred_reshaped
                        self.last_repeat_frame = self.frame_counter

                    # Draw predicted letter on top of bounding box
                    cv2.putText(frame_bgr, smoothed_prediction, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize frame to fit QLabel
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        frame_resized = cv2.resize(frame_bgr, (label_width, label_height))

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        qt_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

        self.text_label.setText(f"Detected Text: {self.detected_text}")
        self.words_label.setText(f"Confirmed Words: {' '.join(self.confirmed_words)}")

    def reset_text(self):
        self.detected_text = ""
        self.confirmed_words = []
        self.text_label.setText("Detected Text: ")
        self.words_label.setText("Confirmed Words: ")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLRecognitionApp()
    window.show()
    sys.exit(app.exec())
