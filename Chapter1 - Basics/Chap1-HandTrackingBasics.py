import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
    sucess,img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    cv2.imshow('Image',img)
    cv2.waitKey(1)

print('Ayush')