import os
# Suppress TensorFlow oneDNN messages early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import mediapipe as mp
import cv2


class FaceMeshDetector:
    def __init__(self, static_mode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.static_mode = static_mode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.static_mode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self,img,draw=True):

        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih,iw,ic = img.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)
                    # print(id,x,y)
                    face.append([x,y])
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture('face_mesh2.mp4')
    pTime = 0

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img,faces = detector.findFaceMesh(img)

        if len(faces) != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()