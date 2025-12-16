import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.pose_detector import PoseDetector, LandmarkIndex


class TestLandmarkIndex:
    def test_indices_exist(self):
        assert LandmarkIndex.LEFT_HIP == 23
        assert LandmarkIndex.LEFT_KNEE == 25
        assert LandmarkIndex.LEFT_ANKLE == 27
        assert LandmarkIndex.RIGHT_HIP == 24
        assert LandmarkIndex.RIGHT_KNEE == 26
        assert LandmarkIndex.RIGHT_ANKLE == 28


class TestPoseDetector:
    def test_detect_no_pose(self):
        """Empty/black frame should return None."""
        detector = PoseDetector()
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(black_frame)
        
        assert result is None
        detector.close()
    
    def test_get_landmark_point(self):
        """Extract Point from mock landmarks."""
        detector = PoseDetector()
        
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.6
        mock_landmarks = [mock_landmark] * 33
        
        point = detector.get_landmark_point(mock_landmarks, 25)
        
        assert point.x == 0.5
        assert point.y == 0.6
        detector.close()
    
    def test_get_knee_angle_points_left(self):
        """Get left leg points."""
        detector = PoseDetector()
        
        mock_landmarks = []
        for i in range(33):
            m = Mock()
            m.x = i * 0.01
            m.y = i * 0.02
            mock_landmarks.append(m)
        
        hip, knee, ankle = detector.get_knee_angle_points(mock_landmarks, "left")
        
        assert hip.x == 23 * 0.01
        assert knee.x == 25 * 0.01
        assert ankle.x == 27 * 0.01
        detector.close()
    
    def test_get_knee_angle_points_right(self):
        """Get right leg points."""
        detector = PoseDetector()
        
        mock_landmarks = []
        for i in range(33):
            m = Mock()
            m.x = i * 0.01
            m.y = i * 0.02
            mock_landmarks.append(m)
        
        hip, knee, ankle = detector.get_knee_angle_points(mock_landmarks, "right")
        
        assert hip.x == 24 * 0.01
        assert knee.x == 26 * 0.01
        assert ankle.x == 28 * 0.01
        detector.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
