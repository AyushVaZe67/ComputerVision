import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('face_detection1.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            # print(id,detection)
            print(detection.score)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,6,(0,0,0),5)
    cv2.imshow('img', img)
    cv2.waitKey(1)