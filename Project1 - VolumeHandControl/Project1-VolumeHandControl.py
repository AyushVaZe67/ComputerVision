import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import HandTrackingModule as htm
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize camera
cap = cv2.VideoCapture(0)
pTime = 0

# Create hand detector
detector = htm.HandDetector(detectionCon=0.7)

# Setup audio control
device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Constants for volume range
minLength = 30
maxLength = 180

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture frame.")
        continue

    # Detect hands
    img, _ = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get coordinates
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw UI elements
        cv2.circle(img, (x1, y1), 15, (0, 255, 200), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 255, 200), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 50, 50), 3)
        cv2.circle(img, (cx, cy), 12, (255, 0, 100), cv2.FILLED)

        # Distance between fingers
        length = math.hypot(x2 - x1, y2 - y1)
        length = np.clip(length, minLength, maxLength)

        # Map length to volume percent (0.0 to 1.0)
        volScalar = np.interp(length, [minLength, maxLength], [0.0, 1.0])
        volPer = int(volScalar * 100)

        # Set volume
        volume.SetMasterVolumeLevelScalar(volScalar, None)

        # Volume bar position
        volBar = np.interp(volScalar, [0.0, 1.0], [400, 150])

        # Volume bar color based on level
        if volPer <= 30:
            barColor = (0, 0, 255)      # Red
        elif volPer <= 70:
            barColor = (0, 255, 255)    # Yellow
        else:
            barColor = (0, 255, 0)      # Green

        # Special circle when muted
        if volPer == 0:
            cv2.circle(img, (cx, cy), 20, (0, 0, 255), cv2.FILLED)

        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), barColor, 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), barColor, cv2.FILLED)
        cv2.putText(img, f'{volPer}%', (40, 430),
                    cv2.FONT_HERSHEY_PLAIN, 2.5, barColor, 3)

    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-8)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 60),
                cv2.FONT_HERSHEY_PLAIN, 2, (200, 255, 255), 2)

    # Show window
    cv2.imshow("Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
