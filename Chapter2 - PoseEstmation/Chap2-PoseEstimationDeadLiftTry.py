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
    cap = cv2.VideoCapture('pose_estimation_6.mp4')
    pTime = 0
    detector = PoseDetector()

    y_coords = []

    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot(y_coords)
    ax.set_ylim(0, 720)  # Adjust based on your video height
    ax.invert_yaxis()    # 🔄 Reverse Y-axis
    ax.set_title("Landmark 20 Y-Coordinate (Inverted)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Y Position (Top to Bottom)")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.getPosition(img, draw=False)

        if len(lmList) > 20:
            y = lmList[20][2]  # Y-coordinate
            y_coords.append(y)

            # Draw a circle on the tracked point
            cv2.circle(img, (lmList[20][1], y), 15, (5, 255, 5), cv2.FILLED)

            # Update matplotlib plot
            line.set_ydata(y_coords)
            line.set_xdata(range(len(y_coords)))
            ax.set_xlim(0, len(y_coords) if len(y_coords) > 50 else 50)
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

        # Show video frame
        cv2.imshow('Pose Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
