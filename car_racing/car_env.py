"""
car_env.py — Gymnasium-среда для гоночного агента.

Observation (13 значений):
    ray_0..ray_6        — 7 лучей до края дороги [0..1]
    long_speed_norm     — продольная скорость (−1..1)
    lat_speed_norm      — боковая скорость / дрейф (−1..1)
    sin(heading)        — направление машинки
    cos(heading)
    sin(rel_to_cp)      — компас: угол до следующего чекпоинта
    cos(rel_to_cp)

Reward design (исправленный, без эксплойтов):
    progress shaping    — приближение к следующему cp (> 0 → плюс, < 0 → сильный минус)
    alignment * k       — маленький плюс только при реальном progress, штраф за разворот
    track alignment     — штраф за движение поперёк/против направления трассы
    turn speed limit    — штраф за высокую скорость перед поворотом по будущим cp
    +5.0                — checkpoint (только на дороге, только по порядку)
    +20.0               — завершение круга
    missed/wrong cp     — сильный terminal penalty за пропуск/не тот checkpoint
    spinning penalty    — вращение без движения вперёд
    lat_slip penalty    — боковое скольжение > 1.2 px/шаг (слишком быстро в поворот)
    corner speed penalty — избыточная скорость когда передние лучи видят близкую стену
    edge penalty         — штраф за езду у края полотна
    stuck penalty       — стоит на месте > 30 шагов
    сильный terminal    — выезд с дороги (6 точек: центр + 4 угла + нос)
"""

import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from track import (Track, generate_random_track,
                   WORLD_WIDTH, WORLD_HEIGHT, MINIMAP_W, MINIMAP_H)
from car import Car, OBS_MAX_LONG_SPEED, OBS_MAX_LAT_SPEED
from utils import draw_hud, make_font

MAX_STEPS          = 4000   # больше шагов для крупных трасс
NUM_TRACK_VARIANTS = 32   # количество случайных вариантов трассы
CHECKPOINT_RADIUS = 45.0
COLLISION_ROAD_MARGIN = 4  # px: белая линия/поребрик не должны давать ложный offroad

PROGRESS_POS_COEFF = 0.05
PROGRESS_NEG_COEFF = 0.14

OFFROAD_BASE_PENALTY  = -40.0
OFFROAD_SPEED_COEFF   = 2.0
OFFROAD_SPEED_CAP     = 30.0

CHECKPOINT_MISS_ARM_DISTANCE = 95.0
CHECKPOINT_MISS_DISTANCE     = 115.0
CHECKPOINT_MISS_PENALTY      = -25.0

MAX_STUCK_STEPS   = 30     # шагов почти без движения → штраф
STUCK_SPEED_THRESH = 0.5   # px/шаг — считаем "стоит"
SPIN_ANGLE_THRESH  = 2.5   # °/шаг — "вращается"
SPIN_SPEED_THRESH  = 1.0   # px/шаг — при такой скорости поворот = вращение

ALIGN_POS_COEFF      = 0.015 # маленький плюс только когда реально приближаемся к cp
ALIGN_NEG_COEFF      = 0.08  # штраф за направление от checkpoint
TRACK_REVERSE_COEFF  = 0.22  # штраф за езду против направления трассы
TRACK_CROSS_COEFF    = 0.28  # штраф за движение поперёк трассы на скорости

TURN_LOOKAHEAD_CHECKPOINTS = 4
TURN_ANGLE_FULL            = 75.0
TURN_SAFE_SPEED_STRAIGHT   = 13.0
TURN_SAFE_SPEED_MIN        = 6.5
TURN_SPEED_COEFF           = 0.07

LAT_SLIP_THRESH      = 1.2   # px/шаг — допустимое боковое скольжение в повороте
LAT_SLIP_COEFF       = 0.08  # штраф за каждый px/шаг сверх порога
FRONT_RAY_THRESH     = 0.75  # нормализованная дистанция (1=200px) — "поворот/стена близко"
FRONT_SPEED_COEFF    = 0.12  # штраф за избыточную скорость относительно просвета
FRONT_SAFE_SPEED_BIAS = 2.0
FRONT_SAFE_SPEED_SCALE = 8.0

EDGE_CENTER_THRESH   = 0.35  # ниже этого score машина едет близко к краю дороги
EDGE_PENALTY_COEFF   = 0.25
ORBIT_PROGRESS_THRESH = 0.5  # мало progress при большом рулении = кружение вокруг цели
ORBIT_PENALTY_COEFF   = 0.04

WINDOW_WIDTH  = 1000
WINDOW_HEIGHT = 700
FPS           = 60


def _angle_diff(a1: float, a2: float) -> float:
    """Нормализованная разность углов в диапазоне [-180, 180]."""
    diff = (a1 - a2) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


class CarRacingEnv(gym.Env):
    """Среда gymnasium «гоночная машинка на случайных трассах».

    Для тех, кто не знаком с RL: среда — это «игра» с формальным
    интерфейсом. reset() начинает эпизод и возвращает наблюдение
    (что «видит» агент), step(action) применяет действие и возвращает
    (наблюдение, награду, конец эпизода, доп. инфо). Алгоритм обучения
    (PPO в train.py) дёргает step миллионы раз и учится выбирать
    действия, максимизирующие сумму наград.

    Действия — 7 дискретных кнопок (spaces.Discrete(7)):
      0 ничего, 1 газ, 2 газ+влево, 3 газ+вправо, 4 тормоз,
      5 влево, 6 вправо.

    Параметры конструктора:
      render_mode        — "human" = окно pygame, None = без графики (обучение);
      render_substeps    — на сколько под-шагов дробить физику при показе
                           (плавная картинка без изменения физики);
      track_variant      — зафиксировать номер трассы (None = случайная);
      random_start_probability — доля эпизодов со стартом из случайной
                           точки трассы (агент видит все секции, не только начало);
      terminate_on_lap   — заканчивать эпизод после полного круга.
    """

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(
        self,
        render_mode=None,
        render_substeps: int = 1,
        track_variant: int | None = None,
        random_start_probability: float = 0.40,
        terminate_on_lap: bool = False,
    ):
        super().__init__()
        self.render_mode      = render_mode
        self._render_substeps = max(1, render_substeps)
        self._fixed_track_variant = track_variant
        self._random_start_probability = float(np.clip(random_start_probability, 0.0, 1.0))
        self._terminate_on_lap = terminate_on_lap

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        self.track          = None
        self.car            = None
        self._track_pool    = None   # 32 варианта, строятся при первом reset()

        # Всё состояние эпизода — инициализируется в reset()
        self._step_count          = 0
        self._next_checkpoint_idx = 0
        self._last_reward         = 0.0
        self._laps_completed      = 0
        self._checkpoints_passed  = 0

        self._prev_dist_to_cp  = 0.0   # для progress reward
        self._best_dist_to_cp  = 0.0   # для missed-checkpoint detection
        self._prev_angle       = 0.0   # для spin detection
        self._prev_x           = 0.0   # для stuck detection
        self._prev_y           = 0.0
        self._stuck_counter    = 0

        # Debug fields — обновляются в _calculate_reward, читаются в render/info
        self._dbg_progress     = 0.0
        self._dbg_alignment    = 0.0
        self._dbg_track_align  = 0.0
        self._dbg_center_score = 1.0
        self._dbg_corner_clearance = 1.0
        self._dbg_turn_severity = 0.0
        self._dbg_turn_safe_speed = TURN_SAFE_SPEED_STRAIGHT
        self._dbg_angle_delta  = 0.0
        self._dbg_offroad_pts  = 0
        self._dbg_dist_to_cp   = 0.0
        self._dbg_terminal_reason = ""

        # Лучи кешируются в step() — используются и в obs, и в reward
        self._current_rays = np.ones(7, dtype=np.float32)

        self.screen = None
        self.clock  = None
        self.font   = None
        self._minimap_cache: dict = {}   # id(track) → scaled surface

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Начинает новый эпизод: выбирает трассу, ставит машинку на старт.

        Возвращает (первое наблюдение, пустой info) — стандарт gymnasium.
        """
        super().reset(seed=seed)

        # Строим пул трасс один раз при первом reset()
        if self._track_pool is None:
            self._track_pool = self._build_track_pool()

        if self._fixed_track_variant is None:
            idx = int(self.np_random.integers(0, NUM_TRACK_VARIANTS))
        else:
            idx = int(self._fixed_track_variant) % len(self._track_pool)
        self.track = self._track_pool[idx]

        checkpoints = self.track.get_checkpoints()
        n_cp = len(checkpoints)

        # 40% эпизодов — старт с рандомного чекпоинта на трассе.
        # Это гарантирует что агент видит ВСЕ секции трассы, а не только начало.
        if n_cp > 0 and self.np_random.random() < self._random_start_probability:
            cp_idx = int(self.np_random.integers(0, n_cp))
            sx, sy, sangle = checkpoints[cp_idx]
            start_cp_idx   = (cp_idx + 1) % n_cp
            init_speed     = 5.0   # небольшая начальная скорость в направлении трассы
        else:
            sx, sy, sangle = self.track.get_start_position()
            start_cp_idx   = 1
            init_speed     = 0.0

        if self.car is None:
            self.car = Car(sx, sy, sangle)
        else:
            self.car.reset(sx, sy, sangle)

        # Если старт с чекпоинта — даём начальную скорость по направлению трассы
        if init_speed > 0.0:
            rad = math.radians(sangle)
            self.car.vx = math.cos(rad) * init_speed
            self.car.vy = math.sin(rad) * init_speed

        # Сбрасываем ВСЕ переменные состояния
        self._step_count          = 0
        self._next_checkpoint_idx = start_cp_idx
        self._last_reward         = 0.0
        self._laps_completed      = 0
        self._checkpoints_passed  = 0

        self._prev_angle       = sangle
        self._prev_x           = sx
        self._prev_y           = sy
        self._stuck_counter    = 0
        self._dbg_progress     = 0.0
        self._dbg_alignment    = 0.0
        self._dbg_track_align  = 0.0
        self._dbg_center_score = 1.0
        self._dbg_corner_clearance = 1.0
        self._dbg_turn_severity = 0.0
        self._dbg_turn_safe_speed = TURN_SAFE_SPEED_STRAIGHT
        self._dbg_angle_delta  = 0.0
        self._dbg_offroad_pts  = 0
        self._dbg_terminal_reason = ""

        # Дистанция до первого целевого чекпоинта
        self._prev_dist_to_cp  = self._dist_to_current_cp()
        self._best_dist_to_cp  = self._prev_dist_to_cp
        self._dbg_dist_to_cp   = self._prev_dist_to_cp

        self._current_rays = self.car.cast_rays(self.track)

        return self._get_obs(), {}

    def step(self, action: int):
        """Один шаг: применить действие -> физика -> сенсоры -> награда.

        Возвращает кортеж gymnasium (obs, reward, terminated, truncated, info):
        terminated — эпизод закончился «по игре» (вылет с трассы и т.п.),
        truncated — просто вышло время (MAX_STEPS).
        """
        self._step_count += 1

        dt = 1.0 / self._render_substeps
        for i in range(self._render_substeps):
            self.car.update(action, dt=dt)
            if self.render_mode == "human" and i < self._render_substeps - 1:
                self._render_sub()

        # Лучи — один раз за шаг (по финальной позиции)
        self._current_rays = self.car.cast_rays(self.track)

        reward, terminated = self._calculate_reward()
        self._last_reward  = reward
        truncated          = self._step_count >= MAX_STEPS

        info = self._build_info()
        return self._get_obs(), reward, terminated, truncated, info

    def _get_camera(self):
        """Камера центрирована на машинке, зажата в границах мира."""
        cx = self.car.x - WINDOW_WIDTH  / 2
        cy = self.car.y - WINDOW_HEIGHT / 2
        cx = max(0.0, min(cx, WORLD_WIDTH  - WINDOW_WIDTH))
        cy = max(0.0, min(cy, WORLD_HEIGHT - WINDOW_HEIGHT))
        return cx, cy

    def _render_sub(self):
        """Промежуточный рендер между под-шагами физики (без лучей)."""
        self._ensure_pygame()
        cam_x, cam_y = self._get_camera()
        self.track.draw_with_active_checkpoint(self.screen, self._next_checkpoint_idx,
                                               cam_x, cam_y)
        self.car.draw(self.screen, draw_rays=False, cam_x=cam_x, cam_y=cam_y)
        self._draw_hud_extended()
        self._draw_minimap(cam_x, cam_y)
        pygame.display.flip()
        self.clock.tick(FPS * self._render_substeps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def render(self):
        """Полный кадр: трасса + машинка с лучами + HUD + мини-карта."""
        if self.render_mode != "human":
            return
        self._ensure_pygame()
        cam_x, cam_y = self._get_camera()
        self.track.draw_with_active_checkpoint(self.screen, self._next_checkpoint_idx,
                                               cam_x, cam_y)
        self.car.draw(self.screen, cam_x=cam_x, cam_y=cam_y)
        self._draw_hud_extended()
        self._draw_minimap(cam_x, cam_y)
        pygame.display.flip()
        self.clock.tick(FPS * self._render_substeps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        """Закрывает окно pygame (если было открыто)."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        13 значений. Лучи из кеша (не пересчитываются).
        Все значения гарантированно в [-1, 1] через np.clip.
        """
        rays      = self._current_rays
        long_norm = np.clip(self.car.long_speed / OBS_MAX_LONG_SPEED, -1.0, 1.0)
        lat_norm  = np.clip(self.car.lat_speed  / OBS_MAX_LAT_SPEED,  -1.0, 1.0)

        rad   = math.radians(self.car.angle)
        sin_h = math.sin(rad)
        cos_h = math.cos(rad)

        checkpoints = self.track.get_checkpoints()
        n  = len(checkpoints)
        tx, ty, _ = checkpoints[self._next_checkpoint_idx % n]
        world_angle = math.atan2(ty - self.car.y, tx - self.car.x)
        rel         = world_angle - rad
        sin_cp      = math.sin(rel)
        cos_cp      = math.cos(rel)

        obs = np.array([*rays, long_norm, lat_norm, sin_h, cos_h, sin_cp, cos_cp],
                       dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _calculate_reward(self):
        """
        Reward без эксплойтов:

        1. OFF-ROAD CHECK (первым делом, до всего остального)
           Проверяем 6 точек: центр + 4 угла + нос.
           Если хотя бы одна вне дороги → terminated=True, сильный speed-scaled penalty.
           Никаких других reward после этого не начисляется.

        2. PROGRESS REWARD
           progress = prev_dist_to_cp - curr_dist_to_cp
           > 0 (едем к cp):  reward += progress * PROGRESS_POS_COEFF
           < 0 (едем от cp): reward += progress * PROGRESS_NEG_COEFF
           Фарм чекпоинтов через кружение невыгоден: отъезд даёт −.

        3. HEADING ALIGNMENT
           dot(car_direction, direction_to_cp) → alignment ∈ [-1, 1]
           > 0: маленький плюс только если реально есть progress.
           < 0: штрафуем за разворот.

        4. TRACK ALIGNMENT / CORNER SPEED
           Штраф за езду поперёк направления трассы, у края и за слишком
           большую скорость, когда передние лучи видят близкую стену.

        5. SPINNING PENALTY
           Если большой угол_delta при малой скорости → фарм кручением.
           reward -= 0.2

        6. STUCK PENALTY
           Если машина почти не движется > 30 шагов → reward -= 0.1/шаг.
           Если позиция почти не изменилась → reward -= 0.05.

        7. CHECKPOINT
           Только на дороге, только следующий по порядку.
           reward += 5.0; после круга += 20.0.
        """
        self._dbg_terminal_reason = ""

        # ---------------------------------------------------------------
        # 1. OFF-ROAD — жёсткий terminal, никакого другого reward
        # ---------------------------------------------------------------
        offroad_count = self._count_offroad_points()
        self._dbg_offroad_pts = offroad_count
        if offroad_count > 0:
            # Обновляем prev-переменные чтобы не было артефактов после reset
            self._prev_angle = self.car.angle
            self._prev_x     = self.car.x
            self._prev_y     = self.car.y
            speed_penalty = min(OFFROAD_SPEED_CAP,
                                self.car.total_speed * OFFROAD_SPEED_COEFF)
            self._dbg_terminal_reason = "offroad"
            return OFFROAD_BASE_PENALTY - speed_penalty, True

        reward = 0.0

        # ---------------------------------------------------------------
        # 2. PROGRESS REWARD
        # ---------------------------------------------------------------
        curr_dist = self._dist_to_current_cp()
        progress  = self._prev_dist_to_cp - curr_dist

        checkpoints = self.track.get_checkpoints()
        n  = len(checkpoints)
        tx, ty, _ = checkpoints[self._next_checkpoint_idx % n]

        rad    = math.radians(self.car.angle)
        track_dir = self.track.get_direction_at(self.car.x, self.car.y)
        track_alignment = math.cos(rad - track_dir)
        self._dbg_track_align = track_alignment

        if progress >= 0:
            # Не даём фармить progress короткой траекторией поперёк трассы.
            track_gate = min(max((track_alignment - 0.15) / 0.65, 0.0), 1.0)
            reward += progress * PROGRESS_POS_COEFF * (0.15 + 0.85 * track_gate)
        else:
            reward += progress * PROGRESS_NEG_COEFF

        self._dbg_progress    = progress
        self._dbg_dist_to_cp  = curr_dist
        self._prev_dist_to_cp = curr_dist

        # ---------------------------------------------------------------
        # 3. HEADING ALIGNMENT
        # ---------------------------------------------------------------
        car_dx = math.cos(rad)
        car_dy = math.sin(rad)

        if curr_dist > 1.0:
            to_nx = (tx - self.car.x) / curr_dist
            to_ny = (ty - self.car.y) / curr_dist
            alignment = car_dx * to_nx + car_dy * to_ny
        else:
            alignment = 1.0

        self._dbg_alignment = alignment
        if alignment >= 0:
            # Не платим просто за "нос смотрит на checkpoint": только когда
            # машина реально приближается и движется вперёд.
            speed_gate    = min(max(self.car.long_speed, 0.0) / 4.0, 1.0)
            progress_gate = min(max(progress, 0.0) / 2.0, 1.0)
            reward += alignment * ALIGN_POS_COEFF * speed_gate * progress_gate
        else:
            reward += alignment * ALIGN_NEG_COEFF

        # ---------------------------------------------------------------
        # 3b. TRACK ALIGNMENT (не даём ездить кругами поперёк трассы)
        # ---------------------------------------------------------------
        if track_alignment < 0.0:
            reward += track_alignment * TRACK_REVERSE_COEFF
        elif self.car.long_speed > 2.0 and track_alignment < 0.45:
            reward -= (0.45 - track_alignment) * TRACK_CROSS_COEFF

        # ---------------------------------------------------------------
        # 3c. EDGE PENALTY (у края меньше права ошибиться на скорости)
        # ---------------------------------------------------------------
        center_score = self.track.get_progress_reward(self.car.x, self.car.y)
        self._dbg_center_score = center_score
        if center_score < EDGE_CENTER_THRESH:
            reward -= (EDGE_CENTER_THRESH - center_score) * EDGE_PENALTY_COEFF

        # ---------------------------------------------------------------
        # 3d. UPCOMING TURN SPEED LIMIT
        # Чем сильнее меняется направление следующих checkpoint, тем ниже
        # допустимая скорость уже до входа в поворот.
        # ---------------------------------------------------------------
        turn_severity = self._upcoming_turn_severity()
        safe_turn_speed = (
            TURN_SAFE_SPEED_STRAIGHT
            - turn_severity * (TURN_SAFE_SPEED_STRAIGHT - TURN_SAFE_SPEED_MIN)
        )
        self._dbg_turn_severity = turn_severity
        self._dbg_turn_safe_speed = safe_turn_speed

        if self.car.long_speed > safe_turn_speed:
            excess = self.car.long_speed - safe_turn_speed
            reward -= (excess * excess) * TURN_SPEED_COEFF * max(turn_severity, 0.25)

        # ---------------------------------------------------------------
        # 4. SPINNING PENALTY (вращение на месте)
        # ---------------------------------------------------------------
        angle_delta          = _angle_diff(self.car.angle, self._prev_angle)
        self._dbg_angle_delta = angle_delta
        speed                = abs(self.car.long_speed)

        if abs(angle_delta) > SPIN_ANGLE_THRESH and speed < SPIN_SPEED_THRESH:
            reward -= 0.2

        # Кружение на скорости: много руления, но почти нет прогресса к цели.
        if abs(angle_delta) > SPIN_ANGLE_THRESH and progress < ORBIT_PROGRESS_THRESH:
            excess_turn = abs(angle_delta) - SPIN_ANGLE_THRESH
            reward -= excess_turn * ORBIT_PENALTY_COEFF

        # ---------------------------------------------------------------
        # 4b. LATERAL SLIP PENALTY (вход в поворот слишком быстро → снос)
        # ---------------------------------------------------------------
        lat_slip = abs(self.car.lat_speed)
        if lat_slip > LAT_SLIP_THRESH:
            reward -= (lat_slip - LAT_SLIP_THRESH) * LAT_SLIP_COEFF

        # ---------------------------------------------------------------
        # 4c. SPEED vs CORNER CLEARANCE
        # Берём центральный и диагональные передние лучи. В повороте один из
        # диагональных лучей обычно видит стену раньше центрального.
        # ---------------------------------------------------------------
        corner_clearance = float(min(self._current_rays[2],
                                     self._current_rays[3],
                                     self._current_rays[4]))
        self._dbg_corner_clearance = corner_clearance
        if corner_clearance < FRONT_RAY_THRESH:
            safe_speed = (FRONT_SAFE_SPEED_BIAS
                          + corner_clearance * FRONT_SAFE_SPEED_SCALE)
            excess     = max(0.0, self.car.long_speed - safe_speed)
            if excess > 0.0:
                reward -= excess * FRONT_SPEED_COEFF

        # ---------------------------------------------------------------
        # 5. STUCK PENALTY (стоит на месте)
        # ---------------------------------------------------------------
        if speed < STUCK_SPEED_THRESH:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        if self._stuck_counter > MAX_STUCK_STEPS:
            reward -= 0.1

        pos_moved = math.hypot(self.car.x - self._prev_x,
                               self.car.y - self._prev_y)
        if pos_moved < 0.3:
            reward -= 0.05

        # Обновляем предыдущие значения
        self._prev_angle = self.car.angle
        self._prev_x     = self.car.x
        self._prev_y     = self.car.y

        # ---------------------------------------------------------------
        # 6. CHECKPOINT (только на дороге, только следующий по порядку)
        # ---------------------------------------------------------------
        on_road = self.track.is_on_road(self.car.x, self.car.y)
        if on_road and curr_dist < CHECKPOINT_RADIUS:
            reward += 5.0
            self._next_checkpoint_idx = (self._next_checkpoint_idx + 1) % n
            self._checkpoints_passed += 1

            if self._next_checkpoint_idx == 0:
                self._laps_completed += 1
                self._checkpoints_passed = max(
                    self._checkpoints_passed,
                    self._laps_completed * n,
                )
                reward += 20.0
                if self._terminate_on_lap:
                    self._dbg_terminal_reason = "lap_completed"
                    return reward, True

            # Сбрасываем дистанцию до НОВОГО следующего чекпоинта
            self._prev_dist_to_cp = self._dist_to_current_cp()
            self._best_dist_to_cp = self._prev_dist_to_cp
        else:
            self._best_dist_to_cp = min(self._best_dist_to_cp, curr_dist)

            missed_current = (
                self._best_dist_to_cp < CHECKPOINT_MISS_ARM_DISTANCE
                and curr_dist > CHECKPOINT_MISS_DISTANCE
            )
            if missed_current:
                self._dbg_terminal_reason = "missed_checkpoint"
                return reward + CHECKPOINT_MISS_PENALTY, True

        return reward, False

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _build_track_pool(self):
        """Генерирует NUM_TRACK_VARIANTS вариантов трассы.
        Вариант 0 — стандартная трасса без изменений (используется при просмотре).
        Варианты 1..N — случайно деформированные трассы для domain randomization.
        """
        print(f"[Track pool] Генерируем {NUM_TRACK_VARIANTS} вариантов трассы...",
              flush=True)
        pool = [Track()]                    # вариант 0: стандартная трасса
        pool.append(Track(figure8=True))    # вариант 1: восьмёрка
        rng_global = np.random.default_rng()   # случайный сид — новые трассы каждый раз
        for i in range(2, NUM_TRACK_VARIANTS):
            wp = generate_random_track(rng_global)
            rw = int(rng_global.integers(72, 94))
            pool.append(Track(waypoints=wp, road_width=rw))
        print(f"[Track pool] Готово.", flush=True)
        return pool

    def _dist_to_current_cp(self) -> float:
        """Расстояние от машинки до текущего целевого чекпоинта."""
        checkpoints = self.track.get_checkpoints()
        n  = len(checkpoints)
        if n == 0:
            return 0.0
        tx, ty, _ = checkpoints[self._next_checkpoint_idx % n]
        return math.hypot(self.car.x - tx, self.car.y - ty)

    def _upcoming_turn_severity(self) -> float:
        """0 = прямая, 1 = резкий поворот впереди по ближайшим checkpoint."""
        checkpoints = self.track.get_checkpoints()
        n = len(checkpoints)
        if n == 0:
            return 0.0

        _, _, base_angle = checkpoints[self._next_checkpoint_idx % n]
        last_angle = base_angle
        max_turn = 0.0

        for offset in range(TURN_LOOKAHEAD_CHECKPOINTS):
            _, _, cp_angle = checkpoints[(self._next_checkpoint_idx + offset) % n]
            max_turn = max(
                max_turn,
                abs(_angle_diff(cp_angle, base_angle)),
                abs(_angle_diff(cp_angle, last_angle)),
            )
            last_angle = cp_angle

        return min(max_turn / TURN_ANGLE_FULL, 1.0)

    def _nearest_wrong_checkpoint(self):
        """Ближайший checkpoint, который не является текущим или только что пройденным."""
        checkpoints = self.track.get_checkpoints()
        n = len(checkpoints)
        if n == 0:
            return -1, float("inf")

        current = self._next_checkpoint_idx % n
        previous = (current - 1) % n
        best_idx = -1
        best_dist = float("inf")

        for idx, (cx, cy, _) in enumerate(checkpoints):
            if idx in (current, previous):
                continue
            dist = math.hypot(self.car.x - cx, self.car.y - cy)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist

        return best_idx, best_dist

    def _count_offroad_points(self) -> int:
        """
        Проверяет 6 точек кузова на выезд с дороги:
          - центр
          - 4 угла
          - передняя центральная точка (нос)

        Возвращает количество точек вне дороги.
        Любое ненулевое значение → terminated.

        Для collision используется небольшой margin, потому что визуальная
        трасса включает белую линию/поребрик, а маска дороги хранит только
        серый асфальт. Без допуска один пиксель на границе выглядит как
        "машина ещё на трассе", но код уже завершает эпизод.
        """
        rad    = math.radians(self.car.angle)
        nose_x = self.car.x + math.cos(rad) * (self.car.HEIGHT / 2)
        nose_y = self.car.y + math.sin(rad) * (self.car.HEIGHT / 2)

        check_points = [
            (self.car.x, self.car.y),
            (nose_x, nose_y),
        ] + self.car.get_corners()

        return sum(
            1
            for x, y in check_points
            if not self.track.is_on_road(x, y, margin=COLLISION_ROAD_MARGIN)
        )

    def _build_info(self) -> dict:
        """Полный debug-словарь для мониторинга обучения."""
        return {
            "on_road":             self.track.is_on_road(self.car.x, self.car.y),
            "body_on_road":        self._dbg_offroad_pts == 0,
            "next_checkpoint_idx": self._next_checkpoint_idx,
            "checkpoints_passed":  self._checkpoints_passed,
            "distance_to_checkpoint": self._dbg_dist_to_cp,
            "best_distance_to_checkpoint": self._best_dist_to_cp,
            "progress":            self._dbg_progress,
            "heading_alignment":   self._dbg_alignment,
            "track_alignment":     self._dbg_track_align,
            "center_score":        self._dbg_center_score,
            "corner_clearance":    self._dbg_corner_clearance,
            "turn_severity":       self._dbg_turn_severity,
            "turn_safe_speed":     self._dbg_turn_safe_speed,
            "speed":               self.car.long_speed,
            "angle_delta":         self._dbg_angle_delta,
            "stuck_counter":       self._stuck_counter,
            "offroad_points_count": self._dbg_offroad_pts,
            "terminal_reason":     self._dbg_terminal_reason,
            "laps":                self._laps_completed,
            "step":                self._step_count,
        }

    # ------------------------------------------------------------------
    # Визуализация
    # ------------------------------------------------------------------

    def _draw_hud_extended(self):
        """HUD с расширенными debug-данными."""
        draw_hud(
            screen=self.screen,
            font=self.font,
            speed=self.car.long_speed,
            checkpoint=self._next_checkpoint_idx,
            total_checkpoints=len(self.track.get_checkpoints()),
            reward=self._last_reward,
            step=self._step_count,
            mode="AI",
        )

        small = make_font("monospace", 13)

        # Полоска дрейфа
        lat   = abs(self.car.lat_speed)
        bar_w = int(min(lat / OBS_MAX_LAT_SPEED, 1.0) * 140)
        pygame.draw.rect(self.screen, (40, 40, 40),    (12, 168, 140, 12))
        if bar_w > 0:
            clr = (80, 150, 255) if lat < 2.0 else (255, 80, 80)
            pygame.draw.rect(self.screen, clr, (12, 168, bar_w, 12))
        pygame.draw.rect(self.screen, (180, 180, 180), (12, 168, 140, 12), 1)

        front_ray = float(self._current_rays[3])
        lines = [
            f"Drift:   {lat:+.2f}",
            f"Front:   {front_ray:.2f}",
            f"Corner:  {self._dbg_corner_clearance:.2f}",
            f"Turn:    {self._dbg_turn_severity:.2f}",
            f"SafeV:   {self._dbg_turn_safe_speed:.1f}",
            f"Align:   {self._dbg_alignment:+.2f}",
            f"Track:   {self._dbg_track_align:+.2f}",
            f"Center:  {self._dbg_center_score:.2f}",
            f"Prog:    {self._dbg_progress:+.2f}",
            f"Stuck:   {self._stuck_counter}",
            f"OffRoad: {self._dbg_offroad_pts}",
            f"Laps:    {self._laps_completed}",
        ]
        for i, line in enumerate(lines):
            surf = small.render(line, True, (180, 180, 180))
            self.screen.blit(surf, (12, 185 + i * 16))

    def _draw_minimap(self, cam_x: float, cam_y: float):
        """Мини-карта всей трассы в правом верхнем углу (200×133 px)."""
        tid = id(self.track)
        if tid not in self._minimap_cache:
            self._minimap_cache.clear()
            self._minimap_cache[tid] = pygame.transform.scale(
                self.track.surface, (MINIMAP_W, MINIMAP_H))
        mm = self._minimap_cache[tid]

        mm_x = WINDOW_WIDTH - MINIMAP_W - 8
        mm_y = 8
        # Рамка
        pygame.draw.rect(self.screen, (30, 30, 30),
                         (mm_x - 2, mm_y - 2, MINIMAP_W + 4, MINIMAP_H + 4))
        self.screen.blit(mm, (mm_x, mm_y))

        # Видимая область (viewport rectangle)
        vx = int(cam_x / WORLD_WIDTH  * MINIMAP_W)
        vy = int(cam_y / WORLD_HEIGHT * MINIMAP_H)
        vw = int(WINDOW_WIDTH  / WORLD_WIDTH  * MINIMAP_W)
        vh = int(WINDOW_HEIGHT / WORLD_HEIGHT * MINIMAP_H)
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (mm_x + vx, mm_y + vy, vw, vh), 1)

        # Позиция машинки
        car_mx = int(self.car.x / WORLD_WIDTH  * MINIMAP_W) + mm_x
        car_my = int(self.car.y / WORLD_HEIGHT * MINIMAP_H) + mm_y
        pygame.draw.circle(self.screen, (255, 80, 80), (car_mx, car_my), 3)

        # Следующий чекпоинт
        n = len(self.track.checkpoints)
        cx, cy, _ = self.track.checkpoints[self._next_checkpoint_idx % n]
        cp_mx = int(cx / WORLD_WIDTH  * MINIMAP_W) + mm_x
        cp_my = int(cy / WORLD_HEIGHT * MINIMAP_H) + mm_y
        pygame.draw.circle(self.screen, (255, 230, 0), (cp_mx, cp_my), 3)

    def _ensure_pygame(self):
        """Ленивая инициализация pygame: окно создаётся при первом рендере,
        а не в конструкторе — при обучении без графики оно не нужно вовсе."""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("RL Car Racing — AI")
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock  = pygame.time.Clock()
            self.font   = make_font("monospace", 18)
