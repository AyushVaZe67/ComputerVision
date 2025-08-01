import os
# Suppress TensorFlow oneDNN messages early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    cv2.imshow('Image',img)
    cv2.waitKey(1)