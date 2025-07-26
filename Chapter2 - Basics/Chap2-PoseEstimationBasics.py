import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import mediapipe as mp


mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,  # For video/live stream
    model_complexity=1,       # 0=light, 1=balanced, 2=heavy
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


print('Ayush')