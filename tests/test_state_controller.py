import pytest
from src.squat_state_controller import SquatStateController, SquatState


class TestSquatStateController:
    def test_full_rep(self):
        """170 → 70 → 170 = 1 rep"""
        ctrl = SquatStateController()
        ctrl.update(170)  # standing
        ctrl.update(70)   # bottom
        state, reps, just_completed = ctrl.update(170)  # back up
        
        assert reps == 1
        assert just_completed == True
        assert state == SquatState.STANDING
    
    def test_dead_zone_no_chatter(self):
        """Fluctuations in dead zone (85-90) shouldn't change state or count reps"""
        ctrl = SquatStateController()
        ctrl.update(70)  # go to bottom first
        
        # oscillate in dead zone
        ctrl.update(85)
        ctrl.update(90)
        state, reps, _ = ctrl.update(85)
        
        assert reps == 0
        assert state == SquatState.ASCENDING
    
    def test_multiple_reps(self):
        """170 → 70 → 170 → 70 → 170 = 2 reps"""
        ctrl = SquatStateController()
        ctrl.update(170)
        ctrl.update(70)
        ctrl.update(170)
        ctrl.update(70)
        state, reps, _ = ctrl.update(170)
        
        assert reps == 2
    
    def test_partial_squat_no_rep(self):
        """170 → 100 → 170 = 0 reps (didn't go low enough)"""
        ctrl = SquatStateController()
        ctrl.update(170)
        ctrl.update(100)  # above bottom threshold
        state, reps, _ = ctrl.update(170)
        
        assert reps == 0
    
    def test_reset(self):
        ctrl = SquatStateController()
        ctrl.update(70)
        ctrl.update(170)
        ctrl.reset()
        
        assert ctrl.rep_count == 0
        assert ctrl.state == SquatState.STANDING
    
    def test_state_transitions(self):
        ctrl = SquatStateController()
        
        state, _, _ = ctrl.update(170)
        assert state == SquatState.STANDING
        
        state, _, _ = ctrl.update(120)  # dead zone, descending
        assert state == SquatState.DESCENDING
        
        state, _, _ = ctrl.update(70)
        assert state == SquatState.BOTTOM
        
        state, _, _ = ctrl.update(120)  # dead zone, ascending
        assert state == SquatState.ASCENDING
        
        state, _, _ = ctrl.update(170)
        assert state == SquatState.STANDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
