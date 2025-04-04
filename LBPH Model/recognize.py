import cv2
import numpy as np

# Load trained face recognizer and label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")

try:
    label_map = np.load("label_map.npy", allow_pickle=True).item()
    print("Label Map Loaded:", label_map)  # Debugging
except Exception as e:
    print(f"Error loading label map: {e}")
    label_map = {}

# Load face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def recognize_faces(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image {image_path}. Check the file path.")
        return

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:  # No faces detected
        print("No face detected in the image.")
        return

    img_color = cv2.imread(image_path)  # Load colored image for visualization

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))  # Normalize size

        # Recognize the face
        label, confidence = recognizer.predict(face)

        # Debugging confidence scores
        print(f"Detected: {label_map.get(label, 'Unknown')} with confidence {confidence}")

        # If confidence is too high, label it as "Unknown"
        if confidence > 120:  # Adjust threshold
            name = "Unknown"
        else:
            name = label_map.get(label, "Unknown")

        # Draw rectangle and label
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_color, f"{name} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Face Recognition", img_color)
    cv2.waitKey(10000)  # Wait for a key press
    cv2.destroyAllWindows()  # Close all image windows
    exit()  # Ensure the script exits


# Test the function
recognize_faces("test.jpg")

