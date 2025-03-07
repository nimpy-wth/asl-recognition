# Real-Time American Sign Language Recognition
241-202 MACHINE LEARNING II - 2/67

## Overview
This mini project is an American Sign Language (ASL) recognition system utilizing computer vision and deep learning techniques. The system captures hand gestures through a webcam, processes them using MediaPipe for landmark extraction, and classifies gestures with a trained Convolutional Neural Network (CNN). The final application provides real-time recognition and translation of ASL gestures into text using PyQt for a graphical user interface.

## Features
- Real-time ASL Recognition using a webcam
- Hand Landmark Detection with Mediapipe
- CNN-Based Prediction Model for classification
- Live Display of predictions with OpenCV

## Setup & Installation
Ensure that you have Python installed on your system before proceeding.
1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-folder>
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application following the usage instructions.
   ```bash
   python app.py
   ```
   
## Usage

- Use the toggle switch to enable left-hand recognition mode.
- Recognized gestures will be translated to text displayed in real-time.
- Special gestures:
   - SPACE: Adds a space to the text.
   - DELETE: Deletes the last character.
   - CONFIRM: Saves the recognized text as a confirmed word.

## Results
###
