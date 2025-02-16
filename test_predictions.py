import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import random

# test dataset path
test_dataset_path = "/Users/nitasneemliala/Dataset/Test" 

img_size = 64
batch_size = 32

# Load new test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")

# Get class labels mapping
class_labels = test_generator.class_indices
print("Class Labels:", class_labels) 

# Get test images
test_images, test_labels = next(test_generator)  

# Get true class labels
true_classes = np.argmax(test_labels, axis=1) 

# Make predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Show test images with predictions
num_images = 25  
plt.figure(figsize=(15, 10))

for i in range(num_images):
    plt.subplot(5, 5, i+1)  # 2 rows, 5 columns
    img = test_images[i]  # Get image
    true_label = class_labels[true_classes[i]] 
    # true_label = class_labels[np.argmax(true_label[i])]
    predicted_label = class_labels[predicted_classes[i]]
    
    # Convert image from TensorFlow format (0-1 range) back to normal
    img = (img * 255).astype("uint8")
    
    # Display image
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()