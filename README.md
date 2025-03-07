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
4. Run the application:
   ```bash
   python app.py
   ```
   
## Usage
- Use the toggle switch to enable left-hand recognition mode.
- Recognized gestures will be translated to text displayed in real-time.
![46652784-アメリカ手話-asl-のアルファベットのスペルの指](https://github.com/user-attachments/assets/9559f733-fb9b-44d2-b048-03ef638c70df)
- Special gestures:
   - SPACE: Adds a space to the text.
   - DELETE: Deletes the last character.
   - CONFIRM: Saves the recognized text as a confirmed word.

## Results
- #### Video Result : 
- #### Presentation : 

#### Model Accuracy : 98.71%
#### Training vs Validation Loss & Accuracy
![2acc_loss](https://github.com/user-attachments/assets/9225ca8a-c7f4-4828-bd51-20e2b3319359)
#### Confusion Matrix
![2confu](https://github.com/user-attachments/assets/4d50c72f-b19c-494b-bf14-348b44f03af8)

