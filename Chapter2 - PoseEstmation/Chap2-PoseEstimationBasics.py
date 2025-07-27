import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mediapipe as mp

cap = cv2.VideoCapture('pose_estimation_2.mp4')

while True:
    success, img = cap.read()
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break