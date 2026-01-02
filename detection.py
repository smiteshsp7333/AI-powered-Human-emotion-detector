import cv2
import numpy as np
import tensorflow as tf

# --- 1. Import the necessary Keras components ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# --- 2. Manually define the model architecture ---
# This structure matches your facialemotionmodel.json file.
model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7, activation='softmax'))  # 7 emotions

# --- 3. Load the pre-trained weights ---
try:
    model.load_weights("facialemotionmodel.h5")
    print("Model weights (facialemotionmodel.h5) loaded successfully.")
except OSError:
    print("Error: facialemotionmodel.h5 not found.")
    exit()
except Exception as e:
    print(f"Error loading weights: {e}")
    exit()

# --- 4. Load the Haar Cascade ---
haar_file = 'C:/Users/smite/OneDrive/Desktop/aies mini project/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

if face_cascade.empty():
    print(f"Error: Could not load Haar cascade from {haar_file}")
    exit()

labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# --- 5. Preprocessing function ---
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# --- 6. Initialize Webcam ---
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")

# --- 7. Main Loop ---
while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # ✅ MIRROR THE CAMERA (ADDED LINE)
    im = cv2.flip(im, 1)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]

            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

            image = cv2.resize(image, (48, 48))
            img = extract_features(image)

            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            cv2.putText(
                im,
                prediction_label,
                (p-10, q-10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 0, 255),
                2
            )

        cv2.imshow("Output", im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error as e:
        print(f"An OpenCV error occurred: {e}")
        pass

# --- 8. Cleanup ---
print("Closing webcam and windows.")
webcam.release()
cv2.destroyAllWindows()
