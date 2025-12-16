import mediapipe as mp
from src.kinematic_math import Point


class LandmarkIndex:
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

class PoseDetector:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame):
        results = self.pose.process(frame)
        if results.pose_landmarks is None:
            return None
        return results.pose_landmarks.landmark
    
    def get_landmark_point(self, landmarks, index):
        lm = landmarks[index]
        return Point(lm.x, lm.y)
    
    def get_knee_angle_points(self, landmarks, side="left"):
        if side == "left":
            hip = self.get_landmark_point(landmarks, LandmarkIndex.LEFT_HIP)
            knee = self.get_landmark_point(landmarks, LandmarkIndex.LEFT_KNEE)
            ankle = self.get_landmark_point(landmarks, LandmarkIndex.LEFT_ANKLE)
        else:
            hip = self.get_landmark_point(landmarks, LandmarkIndex.RIGHT_HIP)
            knee = self.get_landmark_point(landmarks, LandmarkIndex.RIGHT_KNEE)
            ankle = self.get_landmark_point(landmarks, LandmarkIndex.RIGHT_ANKLE)
        return hip, knee, ankle
    
    def close(self):
        self.pose.close()
