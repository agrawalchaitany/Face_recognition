import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define the paths
raw_data_path = 'data/raw/lfw'
processed_data_path = 'data/processed'
dataset_path = 'data/datasets'
model_save_path = 'models/final/face_recognition_model.h5'

# Create necessary directories
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Define image size and parameters
image_size = (128, 128)
batch_size = 32
epochs = 10

# Load and preprocess the dataset
def load_data(processed_data_path, image_size):
    images = []
    labels = []
    for person_name in os.listdir(processed_data_path):
        person_folder = os.path.join(processed_data_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for image_name in os.listdir(person_folder):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(person_folder, image_name)
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
                image = tf.keras.preprocessing.image.img_to_array(image)
                images.append(image)
                labels.append(person_name)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

images, labels = load_data(processed_data_path, image_size)

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Input(shape=(image_size[0], image_size[1], 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Train the model
model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=val_datagen.flow(X_val, y_val)
)

# Save the model
model.save(model_save_path)

# Save the label encoder classes
np.save(os.path.join(dataset_path, 'classes.npy'), label_encoder.classes_)

print(f"Model and classes saved successfully at {model_save_path}")
