"""
centipede_env.py — Gymnasium-окружение "Робот-многоножка" с управлением по командам скорости.

Политика учится не просто бежать вперёд, а ОТРАБАТЫВАТЬ КОМАНДУ:
    command = (v_x, w_z) — целевая скорость вперёд/назад (м/с, в системе робота)
              и целевая скорость поворота (рад/с).
Во время обучения команда случайно меняется каждые ~3 секунды. Большая часть
команд требует ехать вперёд/дугой, а остановка встречается редко — иначе
политика слишком легко учится стабильной статуе. В play.py эту же команду
задают стрелочки.

Наблюдения: углы и скорости суставов (энкодеры серво), высота и ориентация
            головы (IMU), предыдущая команда серво, текущая команда скорости —
            всё это будет доступно и на реальном роботе.
Действия:   нормализованные целевые углы сервоприводов, [-1, 1] на весь ход
            каждого сустава. Ровно то, что потом поедет в PWM реального робота.

Награда (веса — параметры среды). Ключевая идея: команда (v, w) — это ЕДИНАЯ
цель-"твист", как у машины на круговой развязке. Командная скорость поворота
ИНТЕГРИРУЕТСЯ в целевой курс: цель по углу крутится с командной скоростью,
робот "закручивается" за ней по дуге и, когда команда поворота снята,
доворачивает накопленный остаток — то есть выходит из дуги под тем углом,
который набежал по команде, а не под случайным. За отставание по углу
ошибка НАКАПЛИВАЕТСЯ (П-регулятор курса -> эффективная команда w), поэтому
недокрут не прощается, как при слежении за мгновенной угловой скоростью.
Обе компоненты нормируются на свои максимумы и отслеживаются ОДНИМ
экспоненциальным ядром + одним шейпингом вдоль командного направления —
на дуге нельзя "добрать" награду скоростью вперёд, игнорируя поворот.
    + в простом режиме forward: только плавная ходьба вперёд без поворотов
    + в командном режиме: слежение за твистом (v_x, w_z_eff, дрейф v_y=0)
    + шейпинг: проекция фактического твиста на командное направление
      (обрезана на величине команды — перегонять невыгодно)
    + небольшой бонус "жив"; для команды стоп — только небольшой бонус за покой,
      а не полный максимум слежения за ходьбой
    - простой при ненулевой команде
    - качка: вертикальная скорость корпуса, скорости крена/тангажа
    - наклон корпуса (все сегменты должны быть параллельны полу)
    - отклонение от номинальной высоты и явное падение
    - перекрёст соседних ног
    - момент на валах серво (ток и нагрев)
    - резкое изменение команд политики и размахивание суставами
Скорость и курс меряются по СРЕДНЕМУ курсу всех сегментов — голова виляет
вместе с волной позвоночника, и сигнал поворота по ней получается грязным.
Измеренный твист дополнительно сглаживается EMA: волна походки качает
мгновенные скорости внутри цикла шага, и без фильтра ядро наказывает/
поощряет этот шум вместо среднего движения.

Эпизод завершается, если шасси головы вышло за допустимую высоту
или робот завалился.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit

from centipede_model import ensure_xml, FRAME_SKIP, CONTROL_DT, SERVO_TORQUE

# Версия награды/наблюдений среды. Меняется при любой несовместимой правке
# (train.py пишет её в meta.json, play.py отказывается запускать старые модели:
# сеть, обученная в другой среде, ведёт робота вразнос и падает).
REWARD_VERSION = 3

# Пределы команд (общие для обучения и ручного управления в play.py)
MAX_VX_FORWARD = 0.40   # м/с вперёд
MAX_VX_BACKWARD = 0.25  # м/с назад
MAX_WZ = 0.70           # рад/с поворот

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,     # камера следит за головой
    "distance": 1.8,
    "elevation": -20.0,
    "azimuth": 125.0,
}


def _wrap_angle(a: float) -> float:
    """Приводит угол к диапазону [-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


class CentipedeEnv(MujocoEnv, utils.EzPickle):
    """Среда MuJoCo «робот-многоножка», управляемый командами скорости.

    Наследуется от двух классов:
      * MujocoEnv — базовый класс gymnasium для физических симуляций MuJoCo:
        даёт reset/step/render, симулятор self.data и модель self.model;
      * utils.EzPickle — служебный класс, позволяющий сериализовать среду
        (нужно для параллельного обучения в нескольких процессах).

    Схема работы одного шага step():
      действие политики (нормированные углы серво) -> НЧ-фильтр ->
      целевые углы сервоприводов -> FRAME_SKIP тиков физики ->
      измерение фактических скоростей -> награда -> наблюдение.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": int(round(1.0 / CONTROL_DT)),
    }

    def __init__(
        self,
        n_segments: int = 6,
        reward_mode: str = "forward",         # "forward" — просто идти вперёд;
                                              # "command" — старая командная модель v/w
        forward_target_speed: float = 0.25,   # м/с, целевая скорость простой походки
        forward_reward_weight: float = 4.0,   # награда за прогресс вперёд
        forward_track_weight: float = 1.0,    # бонус за скорость около target
        forward_track_sigma2: float = 0.20,   # ширина target-бонуса в нормированных единицах²
        lateral_cost_weight: float = 1.0,     # не ехать боком
        yaw_drift_cost_weight: float = 0.4,   # не закручиваться при ходе вперёд
        # --- слежение за командой-твистом (v и w нормированы на максимумы) ---
        twist_tracking_weight: float = 4.0,   # единое ядро: обе компоненты сразу
        twist_tracking_sigma2: float = 0.10,  # "ширина" ядра в нормированных единицах²
        stop_tracking_weight: float = 0.6,    # команда стоп не должна быть выгоднее ходьбы
        stop_tracking_sigma2: float = 0.04,   # ширина ядра покоя (скорости близки к нулю)
        twist_shaping_weight: float = 2.0,    # проекция факта на командное направление
        twist_idle_cost_weight: float = 3.0,  # штраф за простой при ненулевой команде
        twist_meas_alpha: float = 0.30,       # EMA измеренного твиста (1.0 = без фильтра)
        yaw_err_weight: float = 1.5,          # вес ошибки поворота в ядре (повороты труднее)
        heading_gain: float = 2.0,            # П-регулятор курса: ошибка -> эфф. команда w, 1/с
        heading_err_max: float = 0.6,         # анти-windup: цель не убегает дальше, рад
        leg_cross_cost_weight: float = 50.0,  # штраф за сближение соседних ног (перекрёст)
        leg_clearance: float = 0.06,          # мин. дистанция между соседними ступнями, м
        command_resample_steps: int = 180,    # менять команду каждые N шагов (3.6 с)
        auto_command_resample: bool = True,   # False в play.py: командой рулит человек
        command_profile: str = "all",         # "all" — вся палитра команд (универсал);
                                              # "forward"/"backward"/"left"/"right" —
                                              # специалист учится только своему навыку
        # --- стабильность ---
        healthy_reward: float = 0.1,
        vert_vel_cost_weight: float = 1.0,    # штраф за вертикальную качку корпуса
        angvel_cost_weight: float = 0.03,     # штраф за скорости крена/тангажа
        flatness_cost_weight: float = 1.5,    # штраф за наклон сегментов
        height_cost_weight: float = 400.0,    # штраф за отклонение от номинальной высоты
        height_target: float = 0.095,         # номинальная высота шасси, м
        # --- плавность и бережём сервоприводы ---
        torque_cost_weight: float = 0.02,     # штраф за момент на валах
        action_rate_weight: float = 0.05,     # штраф за резкие изменения команд политики
        action_filter_alpha: float = 0.3,     # НЧ-фильтр целевых углов серво (1.0 = выкл);
                                              # как рампа PWM на реальном контроллере
        dof_vel_cost_weight: float = 1e-4,    # лёгкий штраф за размахивание суставами
        # --- завершение эпизода ---
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple = (0.065, 0.20),
        min_upright: float = 0.35,            # мин. "горизонтальность" КАЖДОГО сегмента
        fall_cost: float = 8.0,               # разовый штраф за терминальное падение
        reset_noise_scale: float = 0.03,
        # --- шероховатая поверхность ---
        terrain_roughness: float = 0.008,     # макс. высота бугров, м (0 = ровный пол)
        terrain_seed: int | None = None,      # None = свой случайный рельеф в каждом процессе
        render_mode: str | None = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            n_segments,
            reward_mode,
            forward_target_speed,
            forward_reward_weight,
            forward_track_weight,
            forward_track_sigma2,
            lateral_cost_weight,
            yaw_drift_cost_weight,
            twist_tracking_weight,
            twist_tracking_sigma2,
            stop_tracking_weight,
            stop_tracking_sigma2,
            twist_shaping_weight,
            twist_idle_cost_weight,
            twist_meas_alpha,
            yaw_err_weight,
            heading_gain,
            heading_err_max,
            leg_cross_cost_weight,
            leg_clearance,
            action_filter_alpha,
            dof_vel_cost_weight,
            command_resample_steps,
            auto_command_resample,
            healthy_reward,
            vert_vel_cost_weight,
            angvel_cost_weight,
            flatness_cost_weight,
            height_cost_weight,
            height_target,
            torque_cost_weight,
            action_rate_weight,
            terminate_when_unhealthy,
            healthy_z_range,
            min_upright,
            fall_cost,
            reset_noise_scale,
            terrain_roughness,
            terrain_seed,
            render_mode,
            command_profile=command_profile,
            **kwargs,
        )
        self.n_segments = n_segments
        if reward_mode not in ("forward", "command"):
            raise ValueError(f"Неизвестный reward_mode: {reward_mode!r}")
        self._reward_mode = reward_mode
        self._forward_target_speed = forward_target_speed
        self._forward_reward_weight = forward_reward_weight
        self._forward_track_weight = forward_track_weight
        self._forward_track_sigma2 = forward_track_sigma2
        self._lateral_cost_weight = lateral_cost_weight
        self._yaw_drift_cost_weight = yaw_drift_cost_weight
        self._twist_tracking_weight = twist_tracking_weight
        self._twist_tracking_sigma2 = twist_tracking_sigma2
        self._stop_tracking_weight = stop_tracking_weight
        self._stop_tracking_sigma2 = stop_tracking_sigma2
        self._twist_shaping_weight = twist_shaping_weight
        self._twist_idle_cost_weight = twist_idle_cost_weight
        self._twist_meas_alpha = twist_meas_alpha
        self._yaw_err_weight = yaw_err_weight
        self._heading_gain = heading_gain
        self._heading_err_max = heading_err_max
        self._leg_cross_cost_weight = leg_cross_cost_weight
        self._leg_clearance = leg_clearance
        self._action_filter_alpha = action_filter_alpha
        self._dof_vel_cost_weight = dof_vel_cost_weight
        self._command_resample_steps = command_resample_steps
        self._auto_command_resample = auto_command_resample
        if command_profile not in ("all", "forward", "backward", "left", "right"):
            raise ValueError(f"Неизвестный command_profile: {command_profile!r}")
        self._command_profile = command_profile
        self._healthy_reward = healthy_reward
        self._vert_vel_cost_weight = vert_vel_cost_weight
        self._angvel_cost_weight = angvel_cost_weight
        self._flatness_cost_weight = flatness_cost_weight
        self._height_cost_weight = height_cost_weight
        self._height_target = height_target
        self._torque_cost_weight = torque_cost_weight
        self._action_rate_weight = action_rate_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._min_upright = min_upright
        self._fall_cost = fall_cost
        self._reset_noise_scale = reset_noise_scale

        xml_path = ensure_xml(n_segments)

        # Предзагрузка модели, чтобы узнать размерности
        m = mujoco.MjModel.from_xml_path(xml_path)
        # qpos без глобальных x,y + скорости + предыдущая команда серво + команда скорости
        obs_dim = (m.nq - 2) + m.nv + m.nu + 2
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=FRAME_SKIP,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            **kwargs,
        )

        self._head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seg0")
        self.head_body_id = self._head_id  # публичный доступ для камеры в play.py
        self._seg_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"seg{i}")
            for i in range(n_segments)
        ]

        # Ступни по сторонам (для штрафа за перекрёст соседних ног)
        self._left_foot_gids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"seg{i}_left_foot")
            for i in range(n_segments)
        ])
        self._right_foot_gids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"seg{i}_right_foot")
            for i in range(n_segments)
        ])

        # Случайный рельеф пола (свой в каждом процессе, если seed не задан)
        self._fill_terrain(terrain_roughness, terrain_seed)

        # Политика выдаёт нормализованные углы [-1, 1]; в step() они
        # разворачиваются в целевые углы внутри хода каждого сустава
        ctrl_range = self.model.actuator_ctrlrange.copy()
        self._ctrl_center = ctrl_range.mean(axis=1)
        self._ctrl_half = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        self._last_action = np.zeros(self.model.nu)      # сырой выход политики (для штрафа)
        self._filtered_action = np.zeros(self.model.nu)  # сглаженная команда серво (идёт в PWM)
        self._command = np.zeros(2)  # (v_x м/с, w_z рад/с)
        self._steps_since_resample = 0
        self._twist_meas = np.zeros(3)  # EMA измеренного (v_x, v_y, w_z)
        self._target_yaw = 0.0          # интеграл командной w: целевой курс "развязки"
        self._cmd_w_eff = 0.0           # эффективная команда w из ошибки курса (идёт в obs)

    # --- Шероховатая поверхность ---

    def _fill_terrain(self, roughness: float, seed: int | None):
        """Заполняет heightfield пологими случайными буграми.

        Центр поля (зона старта робота) остаётся плоским и плавно переходит
        в неровности, чтобы каждый эпизод начинался из одинаковых условий.
        """
        hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hid < 0:
            return
        nrow = int(self.model.hfield_nrow[hid])
        ncol = int(self.model.hfield_ncol[hid])
        adr = int(self.model.hfield_adr[hid])
        if roughness <= 0:
            self.model.hfield_data[adr:adr + nrow * ncol] = 0.0
            return

        rng = np.random.default_rng(seed)
        heights = rng.standard_normal((nrow, ncol))
        # Сглаживание белого шума в пологие бугры: box-блюр по обеим осям
        kernel = np.ones(7) / 7.0
        for _ in range(2):
            heights = np.apply_along_axis(
                lambda row: np.convolve(row, kernel, mode="same"), 1, heights)
            heights = np.apply_along_axis(
                lambda col: np.convolve(col, kernel, mode="same"), 0, heights)
        heights -= heights.min()
        heights /= max(heights.max(), 1e-9)

        # Плоская площадка в центре с плавным выходом на бугры
        half = float(self.model.hfield_size[hid][0])
        xs = np.linspace(-half, half, ncol)
        ys = np.linspace(-half, half, nrow)
        radius = np.hypot(xs[None, :], ys[:, None])
        heights *= np.clip((radius - 0.45) / 0.55, 0.0, 1.0)

        self.model.hfield_size[hid][2] = roughness
        self.model.hfield_data[adr:adr + nrow * ncol] = heights.ravel()

    # --- Команды скорости ---

    def set_command(self, v_x: float, w_z: float):
        """Задать команду извне (стрелочки в play.py)."""
        self._command[0] = float(np.clip(v_x, -MAX_VX_BACKWARD, MAX_VX_FORWARD))
        self._command[1] = float(np.clip(w_z, -MAX_WZ, MAX_WZ))

    @property
    def command(self):
        """Текущая команда (v_x, w_z); копия, чтобы снаружи не изменили оригинал."""
        return self._command.copy()

    def _sample_command(self):
        """Случайная команда согласно профилю среды.

        "all" — вся палитра (универсальная модель, как раньше).
        Специалисты видят только команды своего навыка + немного команд
        "стоп": в play.py кнопку отпускают, и сеть обязана уметь замереть,
        а не продолжать грести.
        """
        if self._command_profile != "all":
            return self._sample_skill_command()
        r = self.np_random.random()
        if r < 0.06:
            return np.zeros(2)
        if r < 0.30:
            # Разворот на месте — доля увеличена: повороты давались хуже всего.
            # Не миниатюрные развороты: |w| от половины максимума
            v_x = 0.0
            w_z = self.np_random.uniform(0.5 * MAX_WZ, MAX_WZ) * self.np_random.choice((-1.0, 1.0))
        elif r < 0.54:
            v_x = self.np_random.uniform(0.16, MAX_VX_FORWARD)
            w_z = 0.0
        elif r < 0.86:
            v_x = self.np_random.uniform(0.10, MAX_VX_FORWARD)
            w_z = self.np_random.uniform(-MAX_WZ, MAX_WZ)
        else:
            v_x = self.np_random.uniform(-MAX_VX_BACKWARD, -0.08)
            w_z = self.np_random.uniform(-0.5 * MAX_WZ, 0.5 * MAX_WZ)
        return np.array([v_x, w_z])

    def _sample_skill_command(self):
        """Команда для сети-специалиста (профили forward/backward/left/right).

        Повороты — «как машина через круг»: половина команд — разворот на
        месте, половина — дуга с ходом вперёд (в play.py это стрелка
        поворота, зажатая вместе со стрелкой вперёд).
        """
        if self.np_random.random() < 0.08:
            return np.zeros(2)  # стоп: уметь замирать при отпущенной кнопке
        profile = self._command_profile
        if profile == "forward":
            return np.array([self.np_random.uniform(0.16, MAX_VX_FORWARD), 0.0])
        if profile == "backward":
            return np.array([self.np_random.uniform(-MAX_VX_BACKWARD, -0.08), 0.0])
        # left / right: знак угловой скорости, |w| от половины максимума
        sign = 1.0 if profile == "left" else -1.0
        w_z = sign * self.np_random.uniform(0.5 * MAX_WZ, MAX_WZ)
        v_x = 0.0 if self.np_random.random() < 0.5 \
            else self.np_random.uniform(0.10, MAX_VX_FORWARD)
        return np.array([v_x, w_z])

    # --- Геометрия ---

    def _heading(self) -> float:
        """Средний курс тела: векторное среднее yaw всех сегментов.

        Голова виляет вместе с волной позвоночника, поэтому курс одного
        сегмента шумит; среднее по телу даёт чистый сигнал поворота.
        """
        xmat = self.data.xmat[self._seg_ids]
        # Ось X сегмента в мировых координатах: (R00, R10) = (xmat[:,0], xmat[:,3])
        return float(np.arctan2(np.sum(xmat[:, 3]), np.sum(xmat[:, 0])))

    @staticmethod
    def _norm_v(v: float) -> float:
        """Нормирует линейную скорость на её предел (вперёд и назад пределы разные)."""
        return v / MAX_VX_FORWARD if v >= 0 else v / MAX_VX_BACKWARD

    @property
    def is_healthy(self) -> bool:
        """Жив ли робот: высота головы в допуске и ни один сегмент не завалился.

        По этому флагу начисляется бонус выживания и завершается эпизод.
        """
        z = self.data.qpos[2]
        z_min, z_max = self._healthy_z_range
        # Элемент R[2][2] матрицы поворота: 1 = ровно, 0 = на боку, -1 = кверху брюхом.
        # Требуем от КАЖДОГО сегмента — иначе робот сворачивается в устойчивую "кучу"
        # с поднятой головой и так обманывает проверку
        uprights = self.data.xmat[self._seg_ids, 8]
        return (
            z_min <= z <= z_max
            and float(np.min(uprights)) > self._min_upright
            and bool(np.isfinite(self.state_vector()).all())
        )

    # --- Основной цикл ---

    def step(self, action):
        """Один шаг управления (0.02 с = 4 тика физики).

        Вход:  action — вектор нормированных целевых углов серво [-1, 1]
               (по одному числу на каждый из 29 приводов).
        Выход: стандартный кортеж gymnasium
               (наблюдение, награда, terminated, truncated, info);
               в info — разбивка награды по всем слагаемым для отладки.
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # НЧ-фильтр (EMA): резкие скачки политики не доходят до серво —
        # на реальном роботе то же самое делает рампа PWM в контроллере
        alpha = self._action_filter_alpha
        self._filtered_action = alpha * action + (1.0 - alpha) * self._filtered_action
        servo_targets = self._ctrl_center + self._filtered_action * self._ctrl_half

        com_before = self.data.subtree_com[self._head_id].copy()
        yaw_before = self._heading()
        self.do_simulation(servo_targets, self.frame_skip)
        com_after = self.data.subtree_com[self._head_id].copy()
        yaw_after = self._heading()

        # Скорости в системе робота (как их видел бы бортовой IMU + одометрия)
        v_world = (com_after - com_before) / self.dt
        cos_y, sin_y = np.cos(yaw_after), np.sin(yaw_after)
        v_x_local = cos_y * v_world[0] + sin_y * v_world[1]   # вперёд по корпусу
        v_y_local = -sin_y * v_world[0] + cos_y * v_world[1]  # вбок (дрейф)
        v_z = v_world[2]
        w_z = _wrap_angle(yaw_after - yaw_before) / self.dt

        # EMA измеренного твиста: волна походки качает мгновенные скорости
        # внутри цикла шага — награда должна следить за средним движением
        beta = self._twist_meas_alpha
        self._twist_meas = beta * np.array([v_x_local, v_y_local, w_z]) + (1.0 - beta) * self._twist_meas
        v_x_f, v_y_f, w_z_f = (float(x) for x in self._twist_meas)

        # --- Целевой курс: интеграл командной w (машина на развязке) ---
        # Цель по углу крутится с командной скоростью; ошибка курса через
        # П-регулятор превращается в эффективную команду w. Недокрут копится
        # и требует доворота, а после снятия команды робот доворачивает
        # остаток и выходит из дуги под накопленным углом.
        self._target_yaw = _wrap_angle(self._target_yaw + self._command[1] * self.dt)
        heading_err = _wrap_angle(self._target_yaw - yaw_after)
        if abs(heading_err) > self._heading_err_max:
            # Анти-windup: цель не убегает дальше, чем робот способен догнать
            heading_err = float(np.clip(heading_err, -self._heading_err_max, self._heading_err_max))
            self._target_yaw = _wrap_angle(yaw_after + heading_err)
        w_cmd_eff = float(np.clip(self._heading_gain * heading_err, -MAX_WZ, MAX_WZ))
        if self._reward_mode == "forward":
            w_cmd_eff = 0.0
        self._cmd_w_eff = w_cmd_eff

        reward_forward = 0.0
        reward_speed_track = 0.0
        lateral_cost = 0.0
        yaw_drift_cost = 0.0
        twist_tracking = 0.0
        twist_shaping = 0.0
        twist_idle_cost = 0.0

        if self._reward_mode == "forward":
            target_v = max(self._forward_target_speed, 1e-6)
            speed_norm = float(v_x_f / target_v)
            reward_forward = self._forward_reward_weight * float(np.clip(speed_norm, -1.0, 1.0))
            speed_err = float((v_x_f - target_v) / target_v)
            reward_speed_track = self._forward_track_weight * float(
                np.exp(-(speed_err ** 2) / self._forward_track_sigma2)
            )
            lateral_cost = self._lateral_cost_weight * float((v_y_f / MAX_VX_FORWARD) ** 2)
            yaw_drift_cost = self._yaw_drift_cost_weight * float((w_z_f / MAX_WZ) ** 2)
        else:
            # --- Слежение за командой-твистом ---
            # Команда и факт как точки в нормированном пространстве (v/v_max, w/w_max):
            # у машины на круговой развязке это одна цель, а не две независимые
            u = np.array([self._norm_v(self._command[0]), w_cmd_eff / MAX_WZ])
            raw_u = np.array([self._norm_v(self._command[0]), self._command[1] / MAX_WZ])
            a_twist = np.array([self._norm_v(v_x_f), w_z_f / MAX_WZ])

            u_norm = float(np.linalg.norm(u))
            raw_u_norm = float(np.linalg.norm(raw_u))
            if raw_u_norm > 0.05 and u_norm > 0.05:
                # Единое ядро: максимум только когда обе компоненты отработаны
                # одновременно (+ дрейф вбок = 0), то есть робот идёт по дуге v/w.
                # Ошибка поворота весит больше: повороты даются труднее скорости.
                d_twist = u - a_twist
                twist_err2 = (
                    float(d_twist[0] ** 2)
                    + self._yaw_err_weight * float(d_twist[1] ** 2)
                    + (v_y_f / MAX_VX_FORWARD) ** 2
                )
                twist_tracking = self._twist_tracking_weight * float(
                    np.exp(-twist_err2 / self._twist_tracking_sigma2)
                )

                # Шейпинг: проекция факта на командное направление. Дуга и поворот
                # автоматически равноправны со скоростью — никакого перекоса "вперёд".
                # Сверху обрезаем на величине команды: перегонять невыгодно.
                along = float(np.dot(u, a_twist)) / u_norm
                twist_shaping = self._twist_shaping_weight * float(np.clip(along, -1.5, u_norm))
                # Простой: движение вдоль команды меньше 35% требуемого
                twist_idle_cost = self._twist_idle_cost_weight * max(0.0, 0.35 * u_norm - along)
            else:
                # Для внешней команды "стоп" не даём полный максимум ходовой награды:
                # иначе политика может сама раскачать курс и получать большую
                # награду за коррекцию вместо полезной локомоции.
                stop_err2 = (
                    float(a_twist[0] ** 2)
                    + self._yaw_err_weight * float((u[1] - a_twist[1]) ** 2)
                    + (v_y_f / MAX_VX_FORWARD) ** 2
                )
                twist_tracking = self._stop_tracking_weight * float(
                    np.exp(-stop_err2 / self._stop_tracking_sigma2)
                )

        # --- Стабильность ---
        healthy = self.is_healthy
        terminated = self._terminate_when_unhealthy and not healthy
        fall_cost = self._fall_cost if terminated else 0.0
        healthy_reward = self._healthy_reward if healthy else 0.0
        vert_vel_cost = self._vert_vel_cost_weight * v_z ** 2
        roll_pitch_rate = self.data.qvel[3] ** 2 + self.data.qvel[4] ** 2
        angvel_cost = self._angvel_cost_weight * float(roll_pitch_rate)
        # Наклон корпуса: xy-компоненты осей Z всех сегментов (0 = все параллельны полу)
        xmat = self.data.xmat[self._seg_ids]
        flatness_cost = self._flatness_cost_weight * float(np.mean(xmat[:, 2] ** 2 + xmat[:, 5] ** 2))
        height_cost = self._height_cost_weight * float((self.data.qpos[2] - self._height_target) ** 2)

        # --- Перекрёст ног: соседние ступни одной стороны не должны сближаться ---
        feet_l = self.data.geom_xpos[self._left_foot_gids][:, :2]
        feet_r = self.data.geom_xpos[self._right_foot_gids][:, :2]
        gaps = np.concatenate((
            np.linalg.norm(np.diff(feet_l, axis=0), axis=1),
            np.linalg.norm(np.diff(feet_r, axis=0), axis=1),
        ))
        overlap = np.clip(self._leg_clearance - gaps, 0.0, None)
        leg_cross_cost = self._leg_cross_cost_weight * float(np.sum(overlap ** 2))

        # --- Бережём сервоприводы ---
        torque_frac = self.data.actuator_force / SERVO_TORQUE
        torque_cost = self._torque_cost_weight * float(np.sum(np.square(torque_frac)))
        action_rate_cost = self._action_rate_weight * float(
            np.sum(np.square(action - self._last_action))
        )
        # Размахивание суставами (скорости без 6 DoF свободного корня)
        dof_vel_cost = self._dof_vel_cost_weight * float(
            np.sum(np.square(self.data.qvel[6:]))
        )

        if self._reward_mode == "forward":
            reward = (
                reward_forward + reward_speed_track + healthy_reward
                - lateral_cost - yaw_drift_cost - vert_vel_cost - angvel_cost
                - flatness_cost - height_cost - leg_cross_cost - torque_cost
                - action_rate_cost - dof_vel_cost - fall_cost
            )
        else:
            reward = (
                twist_tracking + twist_shaping + healthy_reward
                - twist_idle_cost - vert_vel_cost - angvel_cost - flatness_cost - height_cost
                - leg_cross_cost - torque_cost - action_rate_cost - dof_vel_cost - fall_cost
            )

        self._last_action = action

        # Периодическая смена команды во время обучения
        self._steps_since_resample += 1
        if (
            self._reward_mode == "command"
            and self._auto_command_resample
            and not terminated
            and self._steps_since_resample >= self._command_resample_steps
        ):
            self._command = self._sample_command()
            self._steps_since_resample = 0
            # Новая команда — новая "развязка": целевой курс от текущего,
            # эффективная w плавно нарастёт сама по мере интегрирования
            self._target_yaw = yaw_after
            self._cmd_w_eff = 0.0

        observation = self._get_obs()
        info = {
            "x_position": com_after[0],
            "y_position": com_after[1],
            "distance_from_origin": float(np.linalg.norm(com_after[:2])),
            "v_x_local": v_x_f,
            "v_y_local": v_y_f,
            "w_z": w_z_f,
            "v_x_raw": v_x_local,
            "w_z_raw": w_z,
            "cmd_v_x": self._command[0],
            "cmd_w_z": self._command[1],
            "cmd_w_z_eff": w_cmd_eff,
            "heading_err": heading_err,
            "target_yaw": self._target_yaw,
            "reward_twist_tracking": twist_tracking,
            "reward_twist_shaping": twist_shaping,
            "reward_forward": reward_forward,
            "reward_speed_track": reward_speed_track,
            "reward_survive": healthy_reward,
            "cost_twist_idle": -twist_idle_cost,
            "cost_lateral": -lateral_cost,
            "cost_yaw_drift": -yaw_drift_cost,
            "cost_vert_vel": -vert_vel_cost,
            "cost_angvel": -angvel_cost,
            "cost_flatness": -flatness_cost,
            "cost_height": -height_cost,
            "cost_leg_cross": -leg_cross_cost,
            "cost_torque": -torque_cost,
            "cost_action_rate": -action_rate_cost,
            "cost_dof_vel": -dof_vel_cost,
            "cost_fall": -fall_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        """Собирает вектор наблюдений для политики.

        Только то, что доступно и на реальном роботе: углы/скорости суставов
        (энкодеры серво), высота и ориентация головы (IMU), текущее состояние
        PWM и команда скорости. Глобальные координаты x, y исключены —
        реальный робот не знает, где он на столе, и политика не должна
        привязываться к месту.
        """
        position = self.data.qpos.flat[2:]  # без глобальных x, y
        velocity = self.data.qvel.flat[:]
        # В наблюдения идёт СГЛАЖЕННАЯ команда серво — реальное состояние PWM.
        # Вместо сырой w — эффективная команда из ошибки курса: она уже
        # содержит "сколько осталось довернуть", политике не нужно помнить угол
        command_obs = np.array([self._command[0], self._cmd_w_eff])
        return np.concatenate((position, velocity, self._filtered_action, command_obs))

    def reset_model(self):
        """Начало нового эпизода: поза покоя + небольшой случайный шум.

        Шум в стартовой позе не даёт политике заучить одну-единственную
        последовательность движений — она обязана уметь стартовать из
        слегка разных положений. Все внутренние фильтры и цели обнуляются.
        """
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, self.model.nq)
        qvel = self.init_qvel + noise * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        self._last_action = np.zeros(self.model.nu)
        self._filtered_action = np.zeros(self.model.nu)
        self._steps_since_resample = 0
        self._twist_meas = np.zeros(3)
        self._target_yaw = self._heading()  # целевой курс = фактический на старте
        self._cmd_w_eff = 0.0
        if self._reward_mode == "forward":
            self._command = np.array([MAX_VX_FORWARD, 0.0])
        elif self._auto_command_resample:
            self._command = self._sample_command()
        return self._get_obs()


def make_centipede_env(n_segments: int = 6, render_mode: str | None = None,
                       max_episode_steps: int = 1000, **kwargs):
    """Фабрика среды с ограничением длины эпизода (используется в train.py и play.py)."""
    env = CentipedeEnv(n_segments=n_segments, render_mode=render_mode, **kwargs)
    return TimeLimit(env, max_episode_steps=max_episode_steps)
