import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mediapipe as mp

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 5, 5), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture('pose_estimation_3.mp4')
    pTime = 0
    detector = PoseDetector()

    x_coords = []

    plt.ion()  # Interactive mode ON
    fig, ax = plt.subplots()
    line, = ax.plot(x_coords)
    ax.set_ylim(0, 1280)  # Assuming 1280 is max width of video frame
    ax.set_title("Landmark 20 X-Coordinate")
    ax.set_xlabel("Frame")
    ax.set_ylabel("X Position")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.getPosition(img, draw=False)

        if len(lmList) > 20:
            x = lmList[20][2]
            x_coords.append(x)

            # Draw on frame
            cv2.circle(img, (x, lmList[20][2]), 15, (5, 255, 5), cv2.FILLED)

            # Update matplotlib plot
            line.set_ydata(x_coords)
            line.set_xdata(range(len(x_coords)))
            ax.set_xlim(0, len(x_coords) if len(x_coords) > 50 else 50)
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))

        cv2.imshow('Pose Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
