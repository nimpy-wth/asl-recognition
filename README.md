# Real-Time American Sign Language Recognition
241-202 MACHINE LEARNING II - 2/67

## Overview
This project is an American Sign Language (ASL) recognition system that uses Mediapipe to detect hand landmarks and Machine Learning (MLP + LSTM or CNN + LSTM) to recognize ASL letters and predict words. The system captures real-time hand gestures through a webcam, processes hand landmarks, and stores them for training an ASL classification model.

## Features
- Real-time ASL Recognition using a webcam
- Hand Landmark Detection with Mediapipe
- MLP-Based Letter Classification from landmark coordinates
- CNN-Based Prediction Model for classification
- LSTM-Based Word Prediction & Auto-Suggestion
- Live Display of predictions with OpenCV
