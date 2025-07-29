import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('face_detection1.mp4')
pTime = 0

while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,6,(0,0,0))
    cv2.imshow('img', img)
    cv2.waitKey(10)