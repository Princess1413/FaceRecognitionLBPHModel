import cv2
import numpy as np
import os

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def train_recognizer(data_path="dataset"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []
    label_map = {}
    current_label = 0

    for person in os.listdir(data_path):
        person_path = os.path.join(data_path, person)
        if os.path.isdir(person_path):
            label_map[current_label] = person  # Assign label to name
            print(f"Processing images for: {person} -> Label {current_label}")

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Skipping unreadable image: {img_path}")
                    continue

                faces = face_cascade.detectMultiScale(img, 1.3, 5)

                for (x, y, w, h) in faces:
                    face = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (100, 100))  # Normalize size
                    images.append(face_resized)
                    labels.append(current_label)

            current_label += 1

    if len(images) == 0:
        print("No valid faces found. Training aborted.")
        return

    print(f"Training model with {len(images)} images from {len(label_map)} labels.")

    recognizer.train(images, np.array(labels))
    recognizer.save("face_recognizer.yml")
    np.save("label_map.npy", label_map)

    print(f"Training complete. {len(label_map)} persons trained.")

# Run the training function
train_recognizer()



