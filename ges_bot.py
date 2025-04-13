import cv2
import mediapipe as mp
import serial
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
arduino = serial.Serial('COM3', 9600, timeout=1)  # Adjust COM port
time.sleep(2)  # Serial stabilize

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Wave (fingers up, thumb out)
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y:  # Index up
                arduino.write(b'W')  # Wave command
                print("Wave detected!")
            # Point (index up, others down)
            elif hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and \
                 all(hand_landmarks.landmark[i].y > hand_landmarks.landmark[8].y for i in [4, 12, 16, 20]):
                arduino.write(b'P')  # Point command
                print("Point detected!")
            # Thumbs up (thumb up, fingers closed)
            elif hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y and \
                 all(hand_landmarks.landmark[i].y > hand_landmarks.landmark[4].y for i in [8, 12, 16, 20]):
                arduino.write(b'T')  # Thumbs up command
                print("Thumbs up detected!")
    
    cv2.imshow('Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()