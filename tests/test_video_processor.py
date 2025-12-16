import pytest
import os
import json
import tempfile
import numpy as np
import cv2
from unittest.mock import Mock, patch
from src.video_processor import VideoProcessor


class TestVideoProcessor:
    def test_invalid_file_raises_error(self):
        processor = VideoProcessor()
        with pytest.raises(FileNotFoundError):
            processor.process("nonexistent.mp4")
    
    def test_process_synthetic_video(self):
        """Create minimal synthetic video and process it."""
        # Create temp video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        # Generate 10-frame black video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_path, fourcc, 30, (640, 480))
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        
        # Process
        processor = VideoProcessor()
        results = processor.process(temp_path)
        
        # Cleanup
        os.unlink(temp_path)
        
        assert len(results) == 10
        assert all(r["state"] == "no_pose" for r in results)
    
    def test_save_results_json(self):
        processor = VideoProcessor()
        results = [
            {"frame": 0, "angle": 170.0, "state": "standing", "reps": 0},
            {"frame": 1, "angle": 80.0, "state": "bottom", "reps": 0},
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        processor.save_results(results, temp_path)
        
        with open(temp_path) as f:
            loaded = json.load(f)
        
        os.unlink(temp_path)
        
        assert loaded == results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
