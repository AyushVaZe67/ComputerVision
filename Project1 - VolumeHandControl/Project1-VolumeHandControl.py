import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import HandTrackingModule as htm

# Initialize camera
cap = cv2.VideoCapture(0)
pTime = 0

# Create hand detector
detector = htm.HandDetector(detectionCon=0.7)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture frame.")
        continue


    # Detect hands
    img, _ = detector.findHands(img)

    # Get landmark positions
    lmList = detector.findPosition(img, draw=False)


    if len(lmList) != 0:
        print(lmList[4],lmList[8])

        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]

        cx,cy = (x1+x2)//2,(x2+y2)//2
        cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FONT_HERSHEY_PLAIN)
        cv2.circle(img, (x2, y2), 15, (0,255,0), cv2.FONT_HERSHEY_PLAIN)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
        cv2.circle(img,(cx,cy),15,(255,0,55),cv2.FILLED)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-8)
    pTime = cTime

    try:
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hand Tracking", img)
    except cv2.error as e:
        print("OpenCV error:", e)
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
