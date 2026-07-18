"""
bot.py — Контроллеры бота.

BotController  — скриптовый (геометрия, без нейросети).
SelfPlayBot    — использует копию RL-модели как противника.
               Если модель ещё не задана — падбэк на BotController.
"""

import math

from bullet import BULLET_SPEED
from player import RADIUS

SHOOT_ARC     = 14
APPROACH_DIST = 280
SHOOT_DIST    = 760
FLANK_ARC     = 18


def _angle_diff(a: float, b: float) -> float:
    """Кратчайшая разница углов a-b со знаком, градусы, диапазон [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def _line_clear(world, x1: float, y1: float, x2: float, y2: float,
                step: float = 6.0) -> bool:
    """Свободна ли прямая между точками от стен (шагаем по step пикселей)."""
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist <= 1e-6:
        return True
    steps = max(1, int(dist / step))
    for i in range(1, steps + 1):
        t = i / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if world.is_solid(x, y):
            return False
    return True


def _front_clear(actor, world, dist: float = 2.4 * RADIUS) -> bool:
    """Свободно ли прямо перед носом (чтобы не жать MOVE в стену)."""
    rad = math.radians(actor.angle)
    return world.raycast(actor.x, actor.y, rad, max_dist=dist, step=4.0) >= dist


def _lead_angle(shooter, target) -> float:
    """Угол выстрела с упреждением: куда целиться, чтобы пуля встретила
    движущуюся цель.

    Решаем квадратное уравнение относительно времени встречи t
    (пуля летит с BULLET_SPEED, цель — со своей скоростью), берём
    наименьший положительный корень; если корней нет — целимся в
    текущую позицию. Возвращает угол в градусах [0, 360).
    """
    rx = target.x - shooter.x
    ry = target.y - shooter.y
    vx = target.vx
    vy = target.vy

    a = vx * vx + vy * vy - BULLET_SPEED * BULLET_SPEED
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry
    t = None

    if abs(a) < 1e-9:
        if abs(b) > 1e-9:
            candidate = -c / b
            if candidate > 0.0:
                t = candidate
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            root = math.sqrt(disc)
            candidates = [
                (-b - root) / (2.0 * a),
                (-b + root) / (2.0 * a),
            ]
            candidates = [v for v in candidates if v > 0.0]
            if candidates:
                t = min(candidates)

    if t is None:
        t = math.hypot(rx, ry) / BULLET_SPEED

    px = target.x + target.vx * t
    py = target.y + target.vy * t
    return math.degrees(math.atan2(py - shooter.y, px - shooter.x)) % 360.0


class BotController:
    """Скриптовый бот: агрессивно ищет line-of-sight и стреляет с упреждением.

    Никакого обучения — чистая геометрия. Служит противником на старте
    self-play (пока RL-модель агента ещё слаба) и запасным вариантом.
    """

    def get_action(self, bot, target, world, bot_obs=None) -> int:
        """Выбирает действие бота (0/1/2) по простым правилам.

        Если линия огня открыта: пушка довернулась до угла упреждения и
        перезарядка готова — стрелять; смотрит на цель и цель далеко —
        сближаться; иначе ждать доворота (пушка крутится сама).
        Если линия перекрыта: не стрелять в стену, а двигаться вбок
        (фланговый манёвр), чтобы открыть прострел.
        """
        dx   = target.x - bot.x
        dy   = target.y - bot.y
        dist = math.hypot(dx, dy)
        if dist < 1:
            return 0

        direct_angle = math.degrees(math.atan2(dy, dx)) % 360.0
        shot_angle   = _lead_angle(bot, target)
        cur_angle    = bot.angle % 360.0
        shot_diff    = abs(_angle_diff(shot_angle, cur_angle))
        direct_diff  = abs(_angle_diff(direct_angle, cur_angle))
        line_clear   = _line_clear(world, bot.x, bot.y, target.x, target.y)

        if line_clear:
            if shot_diff < SHOOT_ARC and bot.shoot_cooldown == 0 and dist < SHOOT_DIST:
                return 2
            if direct_diff < SHOOT_ARC and dist > APPROACH_DIST and _front_clear(bot, world):
                return 1
            return 0

        # Если прямой линии нет, не стреляем в стену. Поворачиваемся до флангового
        # угла и двигаемся боком, чтобы открыть прострел между центрами.
        flank_sign = 1.0 if bot.x < target.x else -1.0
        flank_angle = (direct_angle + flank_sign * 90.0) % 360.0
        flank_diff = abs(_angle_diff(flank_angle, cur_angle))
        if flank_diff < FLANK_ARC and _front_clear(bot, world):
            return 1
        return 0


class SelfPlayBot:
    """
    Бот, использующий замороженную копию RL-политики агента.
    Пока модель не задана — ведёт себя как BotController.
    Модель обновляется каждые N шагов через update_bot_model() в env.
    """

    def __init__(self):
        self._model    = None
        self._fallback = BotController()

    def set_model(self, model):
        """Копируем веса агента в бота через сохранение/загрузку."""
        import tempfile, os
        from stable_baselines3 import PPO

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "bot_snap")
            model.save(tmp_path)            # пишет bot_snap.zip
            self._model = PPO.load(tmp_path)  # грузит из bot_snap.zip в память

    def get_action(self, bot, target, world, bot_obs=None) -> int:
        """Действие бота: RL-модель, если она уже подложена (и есть её
        наблюдение), иначе скриптовый BotController.
        deterministic=False — бот слегка случайный, чтобы агенту не
        достался идеально предсказуемый противник."""
        if self._model is None or bot_obs is None:
            return self._fallback.get_action(bot, target, world)
        action, _ = self._model.predict(bot_obs, deterministic=False)
        return int(action)
