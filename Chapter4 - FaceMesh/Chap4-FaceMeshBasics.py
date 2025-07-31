import os
# Suppress TensorFlow oneDNN messages early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import mediapipe as mp
import cv2

cap = cv2.VideoCapture('face_detection1.mp4')
pTime = 0


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

while True:
    success,img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),2)
    cv2.imshow('Image',img)
    cv2.waitKey(1)
