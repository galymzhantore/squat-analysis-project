import cv2
import json
from src.pose_detector import PoseDetector, LandmarkIndex
from src.kinematic_math import Point, calculate_angle, apply_ema, apply_ema_point
from src.rep_counter import RepCounter


SKELETON_CONNECTIONS = [
    (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE),
    (LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE),
    (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE),
    (LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_ANKLE),
    (LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP),
]

SMOOTHED_INDICES = [
    LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE,
    LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_ANKLE,
]


class VideoProcessor:
    def __init__(self, bottom_threshold=90.0, rise_threshold=20.0, ema_alpha=0.3):
        self.detector = PoseDetector()
        self.counter = RepCounter(bottom_threshold, rise_threshold)
        self.ema_alpha = ema_alpha
        self.prev_angle = None
        self.prev_points = {}  # smoothed landmark positions
        self.deep_threshold = 90.0
    
    def _smooth_point(self, idx, current):
        prev = self.prev_points.get(idx)
        smoothed = apply_ema_point(current, prev, self.ema_alpha)
        self.prev_points[idx] = smoothed
        return smoothed
    
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
                # Get raw points
                hip_raw, knee_raw, ankle_raw = self.detector.get_knee_angle_points(landmarks, side)
                
                # Smooth all skeleton points
                smoothed_points = {}
                for idx in SMOOTHED_INDICES:
                    raw = Point(landmarks[idx].x, landmarks[idx].y)
                    smoothed_points[idx] = self._smooth_point(idx, raw)
                
                # Get smoothed hip/knee/ankle for angle calculation
                if side == "left":
                    hip = smoothed_points[LandmarkIndex.LEFT_HIP]
                    knee = smoothed_points[LandmarkIndex.LEFT_KNEE]
                    ankle = smoothed_points[LandmarkIndex.LEFT_ANKLE]
                else:
                    hip = smoothed_points[LandmarkIndex.RIGHT_HIP]
                    knee = smoothed_points[LandmarkIndex.RIGHT_KNEE]
                    ankle = smoothed_points[LandmarkIndex.RIGHT_ANKLE]
                
                angle = calculate_angle(hip, knee, ankle)
                reps, _ = self.counter.update(angle)
                status = "DEEP" if angle < self.deep_threshold else "UP"
                
                if writer:
                    self._draw_skeleton(frame, smoothed_points, width, height)
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
    
    def _draw_skeleton(self, frame, smoothed_points, width, height):
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx in smoothed_points and end_idx in smoothed_points:
                start = smoothed_points[start_idx]
                end = smoothed_points[end_idx]
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
