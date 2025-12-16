import pytest
from src.rep_counter import RepCounter


class TestRepCounter:
    def test_full_rep_from_any_start(self):
        counter = RepCounter(bottom_threshold=90, rise_threshold=20)
        
        counter.update(120)
        counter.update(100)
        counter.update(80)
        counter.update(70)
        counter.update(80)
        reps, completed = counter.update(95)
        
        assert reps == 1
        assert completed == True
    
    def test_multiple_reps(self):
        counter = RepCounter(bottom_threshold=90, rise_threshold=20)
        
        counter.update(80)
        counter.update(70)
        counter.update(100)
        
        counter.update(80)
        counter.update(65)
        reps, _ = counter.update(95)
        
        assert reps == 2
    
    def test_no_rep_if_not_deep_enough(self):
        counter = RepCounter(bottom_threshold=90, rise_threshold=20)
        
        counter.update(150)
        counter.update(100)
        counter.update(95)
        reps, _ = counter.update(150)
        
        assert reps == 0
    
    def test_no_rep_if_not_risen_enough(self):
        counter = RepCounter(bottom_threshold=90, rise_threshold=20)
        
        counter.update(80)
        counter.update(70)
        reps, _ = counter.update(85)
        
        assert reps == 0
    
    def test_handles_none_angles(self):
        counter = RepCounter(bottom_threshold=90, rise_threshold=20)
        
        counter.update(80)
        counter.update(70)
        counter.update(None)
        counter.update(None)
        reps, _ = counter.update(100)
        
        assert reps == 1
    
    def test_reset(self):
        counter = RepCounter()
        counter.update(80)
        counter.update(70)
        counter.update(100)
        counter.reset()
        
        assert counter.rep_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
