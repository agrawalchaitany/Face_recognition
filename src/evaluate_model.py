import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def load_images(dataset_path, image_size):
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, image_size)
                    images.append(image)
                    labels.append(label)
                else:
                    print(f"Warning: Unable to load image at {image_path}")
    return np.array(images), np.array(labels)


def evaluate_model():
    final_model_path = r'models\final'
    process_data_path = r'data\processed'
    image_size = (128, 128)

    # Debugging: list all labels and image counts
    print("Listing all labels and image counts in the processed data path:")
    for label in os.listdir(process_data_path):
        label_path = os.path.join(process_data_path, label)
        if os.path.isdir(label_path):
            image_count = len([img for img in os.listdir(
                label_path) if img.endswith('.jpg')])
            print(f"Name: {label}, Image Count: {image_count}")

    images, labels = load_images(process_data_path, image_size)

    print(f"Total images loaded: {len(images)}")
    print(f"Total names loaded: {len(labels)}")

    if len(images) == 0:
        print("No images found in the dataset path.")
        return

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)
    images = images.astype('float32') / 255.0

    model = load_model(os.path.join(
        final_model_path, 'face_recognition_model.h5'))

    predictions = model.predict(images)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(categorical_labels, axis=1)

    print(classification_report(y_pred, y_true, zero_division=0))

    report = classification_report(
        y_pred, y_true, zero_division=0, output_dict=True)

    print("Overall accuracy:", report["accuracy"])
    print("Weighted avg:", report["weighted avg"])


if __name__ == "__main__":
    evaluate_model()