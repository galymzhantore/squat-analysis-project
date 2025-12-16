import pytest
import math
from src.kinematic_math import (
    Point,
    calculate_angle,
    apply_ema,
    apply_ema_point,
)


class TestCalculateAngle:
    def test_right_angle(self):
        a = Point(0, 1)
        b = Point(0, 0)
        c = Point(1, 0)
        
        angle = calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 0.1, f"Expected 90°, got {angle}°"
    
    def test_straight_line(self):
        a = Point(0, 0)
        b = Point(1, 0)
        c = Point(2, 0)
        
        angle = calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 0.1, f"Expected 180°, got {angle}°"
    
    def test_acute_angle_45(self):
        a = Point(0, 1)
        b = Point(0, 0)
        c = Point(1, 1)
        
        angle = calculate_angle(a, b, c)
        assert abs(angle - 45.0) < 0.1, f"Expected 45°, got {angle}°"
    
    def test_acute_angle_60(self):
        a = Point(0, 1)
        b = Point(0, 0)
        c = Point(math.sqrt(3)/2, 0.5)
        
        angle = calculate_angle(a, b, c)
        assert abs(angle - 60.0) < 0.1, f"Expected 60°, got {angle}°"
    
    def test_obtuse_angle_120(self):
        a = Point(1, 0)
        b = Point(0, 0)
        c = Point(-0.5, math.sqrt(3)/2)
        
        angle = calculate_angle(a, b, c)
        assert abs(angle - 120.0) < 0.1, f"Expected 120°, got {angle}°"
    
    def test_squat_bottom_position(self):
        hip = Point(0.5, 0.3)
        knee = Point(0.5, 0.5)
        ankle = Point(0.7, 0.6)
        
        angle = calculate_angle(hip, knee, ankle)
        assert 60 < angle < 120, f"Squat bottom angle should be 60-120°, got {angle}°"
    
    def test_standing_position(self):
        hip = Point(0.5, 0.3)
        knee = Point(0.5, 0.5)
        ankle = Point(0.5, 0.7)
        
        angle = calculate_angle(hip, knee, ankle)
        assert angle > 150, f"Standing angle should be >150°, got {angle}°"


class TestEMAFilter:
    def test_first_value_passthrough(self):
        result = apply_ema(100.0, None, alpha=0.3)
        assert result == 100.0
    
    def test_ema_smoothing(self):
        previous = 100.0
        current = 200.0
        alpha = 0.3
        
        result = apply_ema(current, previous, alpha)
        expected = alpha * current + (1 - alpha) * previous
        
        assert abs(result - expected) < 0.01
        assert result == 130.0
    
    def test_ema_convergence(self):
        value = 0.0
        target = 100.0
        alpha = 0.3
        
        for _ in range(20):
            value = apply_ema(target, value, alpha)
        
        assert abs(value - target) < 1.0, f"EMA should converge to {target}, got {value}"
    
    def test_high_alpha_more_responsive(self):
        previous = 0.0
        current = 100.0
        
        low_alpha_result = apply_ema(current, previous, alpha=0.1)
        high_alpha_result = apply_ema(current, previous, alpha=0.9)
        
        assert high_alpha_result > low_alpha_result
        assert abs(low_alpha_result - 10.0) < 0.01
        assert abs(high_alpha_result - 90.0) < 0.01


class TestEMAPoint:
    def test_point_first_value(self):
        current = Point(100.0, 200.0)
        result = apply_ema_point(current, None)
        
        assert result.x == 100.0
        assert result.y == 200.0
    
    def test_point_smoothing(self):
        previous = Point(0.0, 0.0)
        current = Point(100.0, 200.0)
        alpha = 0.5
        
        result = apply_ema_point(current, previous, alpha)
        
        assert abs(result.x - 50.0) < 0.01
        assert abs(result.y - 100.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
