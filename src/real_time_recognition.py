from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Accuracy
import numpy as np
import cv2
import os


def preprocess_faces(image, face_cascade, image_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_images = []
    rects = []
    for (x, y, w, h) in faces:
        face = cv2.resize(image[y:y+h, x:x+w], image_size)
        face_images.append(face)
        rects.append((x, y, w, h))
    return face_images, rects


def preprocess_input(image):
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def main():
    model_path = 'models/final/face_recognition_model.h5'
    datasets_path = 'data/datasets'
    image_size = (128, 128)

    # Load the model
    model = load_model(model_path)

    # Explicitly compile the model with metrics
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=[Accuracy()])

    # Load label encoder
    classes = np.load(os.path.join(
        datasets_path, 'classes.npy'), allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = classes

    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    set_camera_resolution(cap, 640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_images, rects = preprocess_faces(frame, face_cascade, image_size)
        for i, face in enumerate(face_images):
            face = preprocess_input(face)
            prediction = model.predict(face)
            predicted_class = np.argmax(prediction, axis=1)

            # Draw rectangle around face
            (x, y, w, h) = rects[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Real-Time Face Recognition', frame)

        # Check if window is maximized (full screen)
        if cv2.getWindowProperty('Real-Time Face Recognition', cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(
                'Real-Time Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                'Real-Time Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()