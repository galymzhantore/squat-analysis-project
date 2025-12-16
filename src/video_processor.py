import cv2
import json
from src.pose_detector import PoseDetector, LandmarkIndex
from src.kinematic_math import calculate_angle, apply_ema
from src.rep_counter import RepCounter


# Связи скелета для ног
SKELETON_CONNECTIONS = [
    (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE),
    (LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE),
    (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE),
    (LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_ANKLE),
    (LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP),
]


class VideoProcessor:
    def __init__(self, bottom_threshold=90.0, rise_threshold=20.0, ema_alpha=0.3):
        self.detector = PoseDetector()
        self.counter = RepCounter(bottom_threshold, rise_threshold)
        self.ema_alpha = ema_alpha
        self.prev_angle = None
        self.deep_threshold = 90.0
    
    def process(self, input_path, output_path=None, side="left"):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = self.detector.detect(frame_rgb)
            
            if landmarks:
                hip, knee, ankle = self.detector.get_knee_angle_points(landmarks, side)
                raw_angle = calculate_angle(hip, knee, ankle)
                angle = apply_ema(raw_angle, self.prev_angle, self.ema_alpha)
                self.prev_angle = angle
                
                reps, _ = self.counter.update(angle)
                status = "DEEP" if angle < self.deep_threshold else "UP"
                
                if writer:
                    self._draw_skeleton(frame, landmarks, width, height)
                    self._draw_angle_at_knee(frame, knee, angle, width, height)
                    self._draw_overlay(frame, reps, status)
                
                results.append({
                    "frame": frame_num,
                    "angle": round(angle, 1),
                    "status": status,
                    "reps": reps
                })
            else:
                if writer:
                    self._draw_overlay(frame, self.counter.rep_count, "NO POSE")
                results.append({
                    "frame": frame_num,
                    "angle": None,
                    "status": "NO POSE",
                    "reps": self.counter.rep_count
                })
            
            if writer:
                writer.write(frame)
            
            frame_num += 1
        
        cap.release()
        if writer:
            writer.release()
        self.detector.close()
        
        return results
    
    def _draw_skeleton(self, frame, landmarks, width, height):
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            pt1 = (int(start.x * width), int(start.y * height))
            pt2 = (int(end.x * width), int(end.y * height))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 3)
            cv2.circle(frame, pt1, 6, (255, 0, 255), -1)
            cv2.circle(frame, pt2, 6, (255, 0, 255), -1)
    
    def _draw_angle_at_knee(self, frame, knee, angle, width, height):
        x = int(knee.x * width) + 15
        y = int(knee.y * height)
        cv2.putText(frame, f"{angle:.0f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    def _draw_overlay(self, frame, reps, status):
        color = (0, 255, 0) if status == "UP" else (0, 165, 255)
        cv2.putText(frame, f"Status: {status}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"Reps: {reps}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    def save_results(self, results, path):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
