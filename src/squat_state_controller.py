from enum import Enum


class SquatState(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


class SquatStateController:
    """
    Подсчёт приседаний.
    Использует два порога для предотвращения джиттеринга (как триггер Шмитта).
    """
    
    def __init__(self, bottom_threshold=80.0, top_threshold=160.0):
        self.bottom_threshold = bottom_threshold
        self.top_threshold = top_threshold
        self.state = SquatState.STANDING
        self.rep_count = 0
        self._was_at_bottom = False
    
    def update(self, knee_angle):
        just_completed = False
        
        # Позиция в приседе
        if knee_angle <= self.bottom_threshold:
            self.state = SquatState.BOTTOM
            self._was_at_bottom = True
            return self.state, self.rep_count, just_completed
        
        # Позиция в стоячем положении
        if knee_angle >= self.top_threshold:
            just_completed = self._was_at_bottom
            if just_completed:
                self.rep_count += 1
                self._was_at_bottom = False
            self.state = SquatState.STANDING
            return self.state, self.rep_count, just_completed
        
        # переходы между состояниями
        if self.state == SquatState.BOTTOM:
            self.state = SquatState.ASCENDING
        elif self.state == SquatState.STANDING:
            self.state = SquatState.DESCENDING
        
        return self.state, self.rep_count, just_completed
    
    def reset(self):
        self.state = SquatState.STANDING
        self.rep_count = 0
        self._was_at_bottom = False
