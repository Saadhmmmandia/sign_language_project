import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

DATA_FILE = "dataset/hand_sign_data.csv"

# Write header ONCE
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

# IMPORTANT: take label BEFORE loop
label = input("Enter label (A-Z): ").strip()

if label == "":
    print("ERROR: Label cannot be empty!")
    exit()

cap = cv2.VideoCapture(0)

print("Collecting data for:", label)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            data = []

            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            # ⚠️ IMPORTANT: ALWAYS add label at end
            data.append(label)

            # Save row
            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()