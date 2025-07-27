import os
import time
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose

pTime = 0
cap = cv2.VideoCapture('pose_estimation_2.mp4')

while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255))

    cv2.imshow('Image', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break