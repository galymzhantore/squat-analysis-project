class RepCounter:
    def __init__(self, bottom_threshold=90.0, rise_threshold=20.0):
        self.bottom_threshold = bottom_threshold
        self.rise_threshold = rise_threshold
        self.rep_count = 0
        self.min_angle = None
        self.went_below_threshold = False
    
    def update(self, angle):
        if angle is None:
            return self.rep_count, False
        
        just_completed = False
        
        # Фиксируем что угол опустился ниже порога
        if angle <= self.bottom_threshold:
            self.went_below_threshold = True
            if self.min_angle is None or angle < self.min_angle:
                self.min_angle = angle
        
        # Проверяем подъём после достижения минимума
        if self.went_below_threshold and self.min_angle is not None:
            if angle >= self.min_angle + self.rise_threshold:
                self.rep_count += 1
                just_completed = True
                self.went_below_threshold = False
                self.min_angle = None
        
        return self.rep_count, just_completed
    
    def reset(self):
        self.rep_count = 0
        self.min_angle = None
        self.went_below_threshold = False
