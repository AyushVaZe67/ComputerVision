import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
pTime = 0

detector = htm.HandDetector()

while True:
    success, img = cap.read()
    detector.findHands(img)
    if not success or img is None:
        print("Failed to capture frame.")
        continue

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {fps:.2f}', (40, 50), cv2.FONT_ITALIC, 1, (0, 0, 0), 3)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
