import cv2
import json
from src.pose_detector import PoseDetector
from src.kinematic_math import calculate_angle, apply_ema
from src.rep_counter import RepCounter


class VideoProcessor:
    def __init__(self, bottom_threshold=90.0, rise_threshold=20.0, ema_alpha=0.3):
        self.detector = PoseDetector()
        self.counter = RepCounter(bottom_threshold, rise_threshold)
        self.ema_alpha = ema_alpha
        self.prev_angle = None
    
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
                
                reps, just_completed = self.counter.update(angle)
                
                if writer:
                    self._draw_overlay(frame, angle, reps)
                
                results.append({
                    "frame": frame_num,
                    "angle": round(angle, 1),
                    "reps": reps
                })
            else:
                results.append({
                    "frame": frame_num,
                    "angle": None,
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
    
    def _draw_overlay(self, frame, angle, reps):
        cv2.putText(frame, f"Angle: {angle:.0f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"Reps: {reps}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    def save_results(self, results, path):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
