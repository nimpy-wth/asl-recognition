import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping   # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

#Load Hand Landmarks CSV
data_path = "landmarks_data.csv"
df = pd.read_csv(data_path)

# Extract Features and Labels
X = df.iloc[:, 1:].values # Features (21 Landmarks)
y = df.iloc[:, 0].values  # Labels (A-Z)

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#Data Augmentation Function
def augment_landmarks(landmarks, shift_range=10, scale_range=0.1, rotation_range=10, noise_std=0.02):
    """Applies small transformations to improve generalization"""
    landmarks = np.array(landmarks).reshape(21, 3) 

    # Shifting 
    shift_x = random.uniform(-shift_range, shift_range)
    shift_y = random.uniform(-shift_range, shift_range)
    landmarks[:, 0] += shift_x
    landmarks[:, 1] += shift_y

    # Scaling
    scale_factor = 1 + random.uniform(-scale_range, scale_range)
    center = np.mean(landmarks, axis=0)
    landmarks = (landmarks - center) * scale_factor + center

    # Rotation
    angle = np.radians(random.uniform(-rotation_range, rotation_range))
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    landmarks[:, :2] = np.dot(landmarks[:, :2] - center[:2], rotation_matrix) + center[:2]

    # Jittering
    noise = np.random.normal(0, noise_std, landmarks.shape)
    landmarks += noise

    return landmarks.flatten()

#Augment Data
X_augmented, y_augmented = [], []
for i in range(len(X)):
    original = X[i]
    X_augmented.append(original) 
    y_augmented.append(y_encoded[i])
    
    for _ in range(5): 
        X_augmented.append(augment_landmarks(original))
        y_augmented.append(y_encoded[i])

# Convert to NumPy arrays
X_augmented = np.array(X_augmented)
y_augmented = np.array(y_augmented)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

# Reshape Data for CNN
X_train = X_train.reshape(-1, 21, 3, 1)
X_test = X_test.reshape(-1, 21, 3, 1)

#Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(21, 3, 1)),
    MaxPooling2D(pool_size=(2, 1)),  
    Conv2D(64, (3, 3), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(2, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer for A-Z
])

#Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

#Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#Save Model
model.save("cnn_hand_landmark_model.h5")
print("Model Saved!")

#Plot Training vs Validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.show()

#Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
