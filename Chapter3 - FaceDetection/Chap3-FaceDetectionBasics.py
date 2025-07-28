import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('face_detection1')

while True:
    success, img = cap.read()
    cv2.imshow('img',img)
    cTime = time.time()
    fps = 1(cTime-pTime)
    pTime = cTime