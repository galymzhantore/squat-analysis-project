import math
from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float


def calculate_angle(a, b, c):
    """
    Вычисляет угол в точке B, образованный точками A-B-C, используя atan2.
    Возвращает угол в градусах [0, 180].
    """
    angle_ba = math.atan2(a.y - b.y, a.x - b.x)
    angle_bc = math.atan2(c.y - b.y, c.x - b.x)
    angle = abs(math.degrees(angle_ba - angle_bc))
    return 360 - angle if angle > 180 else angle


def apply_ema(current, previous, alpha=0.3):
    """
    Фильтр экспоненциального скользящего среднего (EMA). Более устойчиво вблизи 0 и 180 градусов.
    EMA_t = alpha * current + (1 - alpha) * previous
    """
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


def apply_ema_point(current, previous, alpha=0.3):
    if previous is None:
        return current
    return Point(
        apply_ema(current.x, previous.x, alpha),
        apply_ema(current.y, previous.y, alpha)
    )
