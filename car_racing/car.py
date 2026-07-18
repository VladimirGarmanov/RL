"""
car.py — Физика машинки с окружностью трения (friction circle).

Ключевая идея (реальная физика):
  Шина может создать суммарную силу не превышающую MU_TIRE (сила сцепления).
  Эта сила делится между:
    - тягой двигателя (engine)
    - продольным торможением (brake)
    - боковым сцеплением (lateral grip)

  ВЕКТОР этих сил не может превысить MU_TIRE:
    sqrt(long² + lat_grip²) ≤ MU_TIRE

  Следствие: в повороте с высоким боковым скольжением — тяга и тормоза слабее.
  На прямой (lat ≈ 0) — двигатель и тормоза работают на полную.
  Сопротивление качению — отдельно от шинного трения.
"""

import pygame
import math
import numpy as np

RAY_ANGLES    = [-90, -60, -30, 0, 30, 60, 90]
MAX_RAY_LENGTH = 200
RAY_STEP      = 3

# Нормировочные константы для observation
OBS_MAX_LONG_SPEED = 14.0
OBS_MAX_LAT_SPEED  = 6.0

COLOR_CAR    = (220, 50, 50)
COLOR_WINDOW = (150, 210, 255)


class Car:
    """
    Машинка с реалистичной физикой (friction circle model).

    Параметры физики:
        ENGINE_FORCE   — тяга двигателя (ограничена MU_TIRE при высоком боковом сносе)
        MU_TIRE        — суммарный бюджет сцепления шины (делится на тягу/торм. + боковое)
        BRAKE_FORCE    — желаемая сила торможения (ограничена MU_TIRE)
        LAT_GRIP       — доля бокового скольжения к исправлению (ограничена MU_TIRE)
        LONG_FRICTION  — сопротивление качению (отдельно от шинного трения)
    """

    # Двигатель — ВНУТРИ окружности трения
    ENGINE_FORCE  = 0.25   # снижено: макс. скорость ~12 px/шаг, повороты достижимы
    REVERSE_FORCE = 0.15

    # Окружность трения — общий бюджет шины
    MU_TIRE     = 0.75   # суммарная сила (px/шаг²). Меньше → больше скольжения
    BRAKE_FORCE = 0.68   # желаемое торможение (будет обрезано MU_TIRE в повороте)
    LAT_GRIP    = 0.90   # доля бокового скольжения к исправлению за шаг

    # Отдельные трения (не шинные)
    LONG_FRICTION = 0.020  # увеличено: ограничивает разгон на длинных прямых

    # Руль
    MAX_STEER      = 5.5   # увеличено: радиус поворота ~115px при макс. скорости
    STEER_DEADZONE = 0.20  # ниже этой скорости руль не работает вообще

    # Размеры
    WIDTH  = 18
    HEIGHT = 34

    def __init__(self, x: float, y: float, angle: float = 0.0):
        self.x     = x
        self.y     = y
        self.angle = angle   # градусы: 0=вправо, 90=вниз

        # Скорость в мировом пространстве — главное состояние физики
        self.vx = 0.0
        self.vy = 0.0

        self._ray_endpoints = []

    # ------------------------------------------------------------------
    # Производные скорости
    # ------------------------------------------------------------------

    @property
    def long_speed(self) -> float:
        """Продольная скорость вдоль капота (< 0 = задний ход)."""
        rad = math.radians(self.angle)
        return self.vx * math.cos(rad) + self.vy * math.sin(rad)

    @property
    def lat_speed(self) -> float:
        """Боковая скорость (= 0 нет сноса, > 0 = снос вправо)."""
        rad = math.radians(self.angle)
        return self.vx * (-math.sin(rad)) + self.vy * math.cos(rad)

    @property
    def speed(self) -> float:
        """Алиас long_speed для HUD."""
        return self.long_speed

    @property
    def total_speed(self) -> float:
        """Полная скорость (модуль вектора), независимо от направления кузова."""
        return math.hypot(self.vx, self.vy)

    # ------------------------------------------------------------------
    # Управление
    # ------------------------------------------------------------------

    def reset(self, x: float, y: float, angle: float = 0.0):
        """Телепортирует машинку в точку старта и обнуляет скорость
        (вызывается в начале каждого эпизода)."""
        self.x, self.y, self.angle = x, y, angle
        self.vx = self.vy = 0.0
        self._ray_endpoints = []

    def update(self, action: int, dt: float = 1.0):
        """
        Физический шаг с friction circle (тяга + торможение + боковое сцепление).

        dt=1.0 — нормальный шаг (как при обучении).
        dt<1.0 — под-шаг: N вызовов с dt=1/N дают идентичный результат одному
                 вызову с dt=1 (масштабирование точное для мультипликативных членов,
                 линейное для аддитивных сил).
        """
        gas         = action in (1, 2, 3)
        brake       = action == 4
        steer_left  = action in (2, 5)
        steer_right = action in (3, 6)

        rad = math.radians(self.angle)
        hx, hy =  math.cos(rad),  math.sin(rad)   # курс
        px, py = -math.sin(rad),  math.cos(rad)   # перпендикуляр (влево)

        # --- 1. Декомпозиция ---
        long_v = self.vx * hx + self.vy * hy
        lat_v  = self.vx * px + self.vy * py

        # --- 2. Сопротивление качению (мультипликативное — точное масштабирование) ---
        long_v *= (1.0 - self.LONG_FRICTION) ** dt

        # --- 3. FRICTION CIRCLE: тяга + боковое сцепление ---
        grip_dt     = 1.0 - (1.0 - self.LAT_GRIP) ** dt   # точное масштабирование grip
        lat_desired = abs(lat_v) * grip_dt

        if gas:
            long_desired = self.ENGINE_FORCE * dt
        elif brake and long_v > 0.05:
            long_desired = self.BRAKE_FORCE * dt
        else:
            long_desired = 0.0

        # Суммарная нагрузка — не должна превысить MU_TIRE * dt
        combined = math.hypot(lat_desired, long_desired)
        mu_dt = self.MU_TIRE * dt
        if combined > mu_dt:
            s            = mu_dt / combined
            lat_desired  *= s
            long_desired *= s

        # Применяем боковое исправление
        lat_correction = min(lat_desired, abs(lat_v))
        lat_v -= math.copysign(lat_correction, lat_v)

        # Применяем тягу / торможение
        if gas:
            long_v += long_desired
        elif brake:
            if long_v > 0.05:
                long_v -= long_desired
                if long_v < 0.0:
                    long_v = 0.0
            else:
                # Задний ход — отдельная механика, не ограничена friction circle
                long_v -= self.REVERSE_FORCE * dt
                long_v = max(long_v, -3.0)

        # --- 5. Руль ---
        v_factor = min(abs(long_v) / 5.0, 1.0)
        if abs(long_v) < self.STEER_DEADZONE:
            v_factor = 0.0

        # При заднем ходе руль инвертируется
        sl, sr = steer_left, steer_right
        if long_v < 0:
            sl, sr = sr, sl

        steer = self.MAX_STEER * v_factor * dt
        if sl: self.angle -= steer
        if sr: self.angle += steer
        self.angle %= 360.0

        # --- 6. Пересборка вектора скорости из нового угла ---
        rad = math.radians(self.angle)
        hx, hy =  math.cos(rad),  math.sin(rad)
        px, py = -math.sin(rad),  math.cos(rad)

        self.vx = long_v * hx + lat_v * px
        self.vy = long_v * hy + lat_v * py

        # Перемещение (масштабируется на dt)
        self.x += self.vx * dt
        self.y += self.vy * dt

    # ------------------------------------------------------------------
    # Collision — проверяем все 4 угла кузова
    # ------------------------------------------------------------------

    def get_corners(self):
        """4 угла машинки в мировых координатах."""
        hw  = self.WIDTH  / 2
        hh  = self.HEIGHT / 2
        rad = math.radians(self.angle)
        ca, sa = math.cos(rad), math.sin(rad)
        local = [(-hh, -hw), (hh, -hw), (hh, hw), (-hh, hw)]
        return [
            (self.x + lx * ca - ly * sa,
             self.y + lx * sa + ly * ca)
            for lx, ly in local
        ]

    def is_any_corner_off_road(self, track) -> bool:
        """True если хоть один угол машинки за пределами дороги."""
        return any(not track.is_on_road(cx, cy) for cx, cy in self.get_corners())

    # ------------------------------------------------------------------
    # Сенсоры
    # ------------------------------------------------------------------

    def cast_rays(self, track) -> np.ndarray:
        """
        7 лучей. Возвращает расстояния в [0, 1].

        ВАЖНО: dist ограничен MAX_RAY_LENGTH до деления,
        чтобы результат никогда не превышал 1.0.
        """
        distances = []
        self._ray_endpoints = []

        for delta in RAY_ANGLES:
            angle_rad = math.radians(self.angle + delta)
            dx, dy = math.cos(angle_rad), math.sin(angle_rad)

            dist   = 0.0
            hit_x  = self.x
            hit_y  = self.y

            while dist < MAX_RAY_LENGTH:
                next_dist = dist + RAY_STEP
                cx = self.x + dx * next_dist
                cy = self.y + dy * next_dist

                if not track.is_on_road(cx, cy):
                    # Граница найдена между dist и next_dist
                    hit_x = self.x + dx * dist
                    hit_y = self.y + dy * dist
                    break

                dist  = next_dist
                hit_x = cx
                hit_y = cy

            # dist уже ≤ MAX_RAY_LENGTH, поэтому dist/MAX ≤ 1.0 гарантировано
            distances.append(dist / MAX_RAY_LENGTH)
            self._ray_endpoints.append((hit_x, hit_y))

        return np.array(distances, dtype=np.float32)

    # ------------------------------------------------------------------
    # Отрисовка
    # ------------------------------------------------------------------

    def draw(self, screen: pygame.Surface, draw_rays: bool = False,
             cam_x: float = 0, cam_y: float = 0):
        """Рисует машинку (и опционально лучи сенсоров).
        cam_x/cam_y — смещение камеры: мировые координаты минус эти
        значения дают экранные."""
        if draw_rays:
            self._draw_rays(screen, cam_x, cam_y)
        self._draw_car(screen, cam_x, cam_y)

    def _draw_rays(self, screen, cam_x: float = 0, cam_y: float = 0):
        """Рисует лучи-сенсоры: жёлтые линии до точки касания границы дороги,
        красные точки — сами точки касания (для отладки «что видит агент»)."""
        ox, oy = int(self.x - cam_x), int(self.y - cam_y)
        for hx, hy in self._ray_endpoints:
            pygame.draw.line(screen, (255, 255, 100),
                             (ox, oy),
                             (int(hx - cam_x), int(hy - cam_y)), 1)
            pygame.draw.circle(screen, (255, 80, 80),
                               (int(hx - cam_x), int(hy - cam_y)), 3)

    def _draw_car(self, screen, cam_x: float = 0, cam_y: float = 0):
        """Рисует кузов, лобовое стекло и центр машинки.

        Цвет кузова кодирует занос: при боковой скорости > 1.2 корпус
        плавно синеет — визуальная индикация скольжения.
        """
        corners = self.get_corners()
        ipts = [(int(x - cam_x), int(y - cam_y)) for x, y in corners]

        lat = abs(self.lat_speed)
        if lat > 1.2:
            t = min((lat - 1.2) / 3.5, 1.0)
            body = (int(220*(1-t)+70*t), int(50*(1-t)+120*t), int(50*(1-t)+255*t))
        else:
            body = COLOR_CAR

        pygame.draw.polygon(screen, body, ipts)
        pygame.draw.polygon(screen, (120, 0, 0), ipts, 2)

        # Лобовое стекло (все координаты в screen-пространстве)
        rad = math.radians(self.angle)
        ca, sa = math.cos(rad), math.sin(rad)
        sx, sy = self.x - cam_x, self.y - cam_y
        wcx = sx + ca * self.HEIGHT * 0.15
        wcy = sy + sa * self.HEIGHT * 0.15
        hw, hh = self.WIDTH * 0.28, self.HEIGHT * 0.14
        wpts = [
            (wcx - ca*hh + sa*hw, wcy - sa*hh - ca*hw),
            (wcx + ca*hh + sa*hw, wcy + sa*hh - ca*hw),
            (wcx + ca*hh - sa*hw, wcy + sa*hh + ca*hw),
            (wcx - ca*hh - sa*hw, wcy - sa*hh + ca*hw),
        ]
        pygame.draw.polygon(screen, COLOR_WINDOW,
                             [(int(x), int(y)) for x, y in wpts])
        pygame.draw.circle(screen, (255, 255, 255), (int(sx), int(sy)), 2)
