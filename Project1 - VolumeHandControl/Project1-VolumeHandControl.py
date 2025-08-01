import os
# Suppress TensorFlow oneDNN messages early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success,img = cap.read()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {fps:.2f}', (40, 70), cv2.FONT_ITALIC, 2, (255, 255, 255), 1)
    cv2.imshow('Image',img)
    cv2.waitKey(1)