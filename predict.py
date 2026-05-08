import cv2
import mediapipe as mp
import numpy as np
import joblib

print("Loading model...")

model = joblib.load("models/sign_model.pkl")

print("Model loaded!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not detected")
    exit()

print("Camera opened successfully")

# Confidence threshold (IMPORTANT)
THRESHOLD = 0.50

while True:
    success, frame = cap.read()

    if not success:
        print("Frame not received")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction = "NONE"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract features
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(1, -1)

            # Get probabilities (IMPORTANT)
            probs = model.predict_proba(data)[0]

            best_index = np.argmax(probs)
            confidence = probs[best_index]

            prediction = model.classes_[best_index]

            # NONE LOGIC (KEY UPGRADE)
            if confidence < THRESHOLD:
                prediction = "NONE"

    # Show result
    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()