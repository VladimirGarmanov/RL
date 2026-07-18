"""
play.py — Управление роботом-многоножкой стрелочками, 3D-вид и HUD в ОДНОМ окне.

Раньше было два окна (3D MuJoCo + пульт pygame) — на macOS они дрались за фокус
клавиатуры, и стрелки не доходили до пульта. Теперь окно одно: MuJoCo рендерит
кадры за кадром (offscreen), pygame показывает их и читает клавиатуру.

Запуск:
    python play.py                — управление стрелочками; если обучены все
                                    4 сети-специалиста (train.py forward/backward/
                                    left/right), каждую стрелку отрабатывает СВОЯ
                                    сеть, иначе — одна универсальная модель
    python play.py --single      — принудительно одна универсальная модель
    python play.py --forward     — обученная модель постоянно идёт вперёд
    python play.py --auto        — робот сам ездит по программе команд
    python play.py --manual      — стрелки без модели: процедурная походка (проверка механики)
    python play.py --random      — случайные действия (посмотреть тело)
    python play.py --model models/checkpoints/ppo_centipede_500000_steps.zip

Режим 4 сетей: стрелка выбирает специалиста (важен приоритет — повороты
старше хода: зажал ← и ↑ — рулит сеть "left" и ведёт дугой, как машина).
Когда все кнопки отпущены, последняя активная сеть держит робота на месте
(каждый специалист учился и команде "стоп").

Робот:
    Стрелка вверх / вниз   — ехать вперёд / назад
    Стрелка влево / вправо — поворачивать
    Пробел                 — стоп (сбросить команду)
    R                      — рестарт эпизода
    Esc / закрыть окно     — выход

Камера:
    Мышь с зажатой левой кнопкой — вращать вокруг точки взгляда
    Колесо мыши                  — приблизить / отдалить
    W A S D                      — панорама по плоскости (отключает следование)
    F                            — включить/выключить следование за роботом
    Home                         — сбросить камеру
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import pygame

from centipede_model import CONTROL_DT
from centipede_env import (
    make_centipede_env,
    MAX_VX_FORWARD,
    MAX_VX_BACKWARD,
    MAX_WZ,
    GAIT_PHASE_LAG,
    REWARD_VERSION,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
SKILLS_DIR = os.path.join(MODELS_DIR, "skills")
SKILLS = ("forward", "backward", "left", "right")
SKILL_RU = {"forward": "прямо", "backward": "назад", "left": "налево", "right": "направо"}

# Размер 3D-кадра и высота HUD-панели под ним
VIEW_W, VIEW_H = 960, 540
HUD_H = 150

# Скорость нарастания команды при зажатой стрелке (плавный разгон)
CMD_RAMP_TIME = 0.4  # секунд от нуля до максимума
RAMP_V = MAX_VX_FORWARD / CMD_RAMP_TIME * CONTROL_DT
RAMP_W = MAX_WZ / CMD_RAMP_TIME * CONTROL_DT

CAM_PAN_SPEED = 0.9  # м/с панорамы на каждый метр дистанции камеры

# Программа для режима --auto: (название, секунд, v_x, w_z)
AUTO_SCRIPT = [
    ("вперёд", 4.0, MAX_VX_FORWARD, 0.0),
    ("стоп", 1.5, 0.0, 0.0),
    ("разворот влево", 3.5, 0.0, MAX_WZ),
    ("дуга направо", 4.0, 0.7 * MAX_VX_FORWARD, -0.5 * MAX_WZ),
    ("назад", 2.5, -MAX_VX_BACKWARD, 0.0),
    ("стоп", 1.5, 0.0, 0.0),
]

# Цвета HUD
C_WHITE = (235, 235, 235)
C_ORANGE = (255, 170, 60)
C_GREEN = (120, 220, 120)
C_RED = (200, 120, 120)
C_DIM = (140, 140, 150)


class SceneWindow:
    """Одно pygame-окно: 3D-кадр MuJoCo сверху, HUD-панель снизу.

    Владеет свободной камерой MuJoCo: следование за роботом (F), панорама
    по плоскости (WASD), вращение мышью, зум колесом, сброс (Home).
    """

    def __init__(self, core_env, caption):
        self.env = core_env  # CentipedeEnv (unwrapped)
        pygame.init()
        self.screen = pygame.display.set_mode((VIEW_W, VIEW_H + HUD_H))
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 19)
        self.cam = None
        self.follow = True
        self.quit_requested = False
        self.reset_requested = False

    # --- Камера ---

    def _ensure_camera(self):
        if self.cam is not None:
            return
        # Первый render создаёт offscreen-вьюер, у него забираем MjvCamera
        self.env.render()
        self.cam = self.env.mujoco_renderer.viewer.cam
        self.reset_camera()

    def reset_camera(self):
        self._ensure_camera()
        cam = self.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.trackbodyid = -1
        cam.fixedcamid = -1
        cam.distance = 2.0
        cam.azimuth = 125.0
        cam.elevation = -22.0
        head = self.env.data.xpos[self.env.head_body_id]
        cam.lookat[:] = (head[0], head[1], 0.06)
        self.follow = True

    def _pan_camera(self, keys):
        dx = (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0)
        dy = (1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0)
        if not (dx or dy):
            return
        self._ensure_camera()
        self.follow = False
        theta = np.radians(self.cam.azimuth)
        fwd = (np.cos(theta), np.sin(theta))       # направление взгляда на плоскости
        right = (np.sin(theta), -np.cos(theta))
        step = CAM_PAN_SPEED * self.cam.distance * CONTROL_DT
        self.cam.lookat[0] += step * (dy * fwd[0] + dx * right[0])
        self.cam.lookat[1] += step * (dy * fwd[1] + dx * right[1])

    # --- События и отрисовка ---

    def process_events(self):
        """Обрабатывает окно/камеру; возвращает снимок клавиатуры для вызывающего."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                elif event.key == pygame.K_r:
                    self.reset_requested = True
                elif event.key == pygame.K_f:
                    self.follow = not self.follow
                elif event.key == pygame.K_HOME:
                    self.reset_camera()
            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                self._ensure_camera()
                self.cam.azimuth -= event.rel[0] * 0.4
                self.cam.elevation = float(
                    np.clip(self.cam.elevation - event.rel[1] * 0.3, -85.0, -5.0))
            elif event.type == pygame.MOUSEWHEEL:
                self._ensure_camera()
                self.cam.distance = float(
                    np.clip(self.cam.distance * (0.92 ** event.y), 0.4, 8.0))

        keys = pygame.key.get_pressed()
        self._pan_camera(keys)
        return keys

    def draw(self, rows):
        self._ensure_camera()
        if self.follow:
            head = self.env.data.xpos[self.env.head_body_id]
            la = self.cam.lookat
            la[0] += 0.12 * (head[0] - la[0])
            la[1] += 0.12 * (head[1] - la[1])
            la[2] = 0.06

        frame = self.env.render()
        surf = pygame.image.frombuffer(
            frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
        self.screen.blit(surf, (0, 0))

        pygame.draw.rect(self.screen, (18, 20, 26), (0, VIEW_H, VIEW_W, HUD_H))
        y = VIEW_H + 10
        for text, color in rows:
            self.screen.blit(self.font.render(text, True, color), (16, y))
            y += 26
        for line in (
            "робот:  стрелки — ехать,  пробел — стоп,  R — рестарт,  Esc — выход",
            "камера: мышь (ЛКМ) — вращать,  колесо — зум,  WASD — панорама,  F — следование,  Home — сброс",
        ):
            self.screen.blit(self.font_small.render(line, True, C_DIM), (16, y))
            y += 21
        pygame.display.flip()

    def close(self):
        pygame.quit()


def approach(value, target, step):
    """Плавно двигает value к target с шагом step."""
    if value < target:
        return min(value + step, target)
    return max(value - step, target)


def update_drive(keys, cmd_vx, cmd_wz):
    """Стрелки -> целевая команда с плавным разгоном; пробел -> мгновенный стоп."""
    if keys[pygame.K_SPACE]:
        return 0.0, 0.0
    target_vx = MAX_VX_FORWARD if keys[pygame.K_UP] else (
        -MAX_VX_BACKWARD if keys[pygame.K_DOWN] else 0.0)
    target_wz = MAX_WZ if keys[pygame.K_LEFT] else (
        -MAX_WZ if keys[pygame.K_RIGHT] else 0.0)
    return approach(cmd_vx, target_vx, RAMP_V), approach(cmd_wz, target_wz, RAMP_W)


def status_rows(cmd_vx, cmd_wz, info, falls, follow, header=None):
    """Собирает строки HUD-панели: команда, фактические скорости, счётчик падений.

    Возвращает список пар (текст, цвет) — их отрисует SceneWindow.draw().
    """
    rows = []
    if header:
        rows.append((header, C_WHITE))
    rows.append((f"команда:  скорость {cmd_vx:+.2f} м/с    поворот {cmd_wz:+.2f} рад/с    "
                 f"(эфф. {info.get('cmd_w_z_eff', 0.0):+.2f}, довернуть {np.degrees(info.get('heading_err', 0.0)):+.0f}°)",
                 C_ORANGE))
    rows.append((f"факт:     скорость {info.get('v_x_local', 0.0):+.2f} м/с    "
                 f"поворот {info.get('w_z', 0.0):+.2f} рад/с", C_GREEN))
    rows.append((f"падений: {falls}      камера: {'следует за роботом' if follow else 'свободная'}", C_RED))
    return rows


def checkpoint_steps(path):
    """Число шагов из имени чекпоинта ('..._500000_steps.zip' -> 500000);
    -1, если имя не подходит под шаблон. Нужно для поиска самого свежего."""
    m = re.search(r"_(\d+)_steps\.zip$", path)
    return int(m.group(1)) if m else -1


def normalized_speed(cmd_vx):
    """Команда скорости -> доля от максимума [-1, 1].
    Пределы вперёд и назад разные, поэтому делители разные."""
    if cmd_vx >= 0:
        return float(np.clip(cmd_vx / MAX_VX_FORWARD, 0.0, 1.0))
    return float(np.clip(cmd_vx / MAX_VX_BACKWARD, -1.0, 0.0))


def manual_cpg_action(n_segments, phase, cmd_vx, cmd_wz):
    """Простая волновая походка, чтобы проверить механику до обучения PPO.

    CPG (central pattern generator) — «генератор ритма», как у настоящих
    многоножек: все ноги качаются синусоидой с фазовым сдвигом вдоль тела
    (бегущая волна), левая и правая стороны в противофазе. Поворот
    добавляется изгибом позвоночника и асимметрией маха ног.

    Вход:  phase — текущая фаза волны (растёт со временем);
           cmd_vx, cmd_wz — команда скорости/поворота.
    Выход: вектор действий той же формы, что у политики (5*n-1 серво).
    """
    speed = normalized_speed(cmd_vx)
    turn = float(np.clip(cmd_wz / MAX_WZ, -1.0, 1.0))
    drive = max(abs(speed), abs(turn))
    if drive < 0.02:
        return np.zeros(5 * n_segments - 1, dtype=np.float32)

    action = []
    direction = 1.0 if speed >= 0 else -1.0
    speed_abs = abs(speed)
    for i in range(n_segments):
        if i > 0:
            spine = 0.42 * turn * np.sin(phase - i * 0.70)
            spine += 0.16 * speed * np.sin(phase - i * 0.45)
            action.append(spine)

        for side in (+1, -1):
            leg_phase = phase + i * GAIT_PHASE_LAG + (0.0 if side > 0 else np.pi)
            swing = np.sin(leg_phase)
            lift = max(0.0, swing)

            hip = 0.62 * direction * speed_abs * np.sin(leg_phase)
            hip += 0.35 * turn * side * np.cos(leg_phase)
            knee = -0.28 * drive + 0.78 * drive * lift
            action.extend((hip, knee))

    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def find_model_and_vecnorm(args):
    """Ищет модель и парную статистику нормализации."""
    model_path = args.model
    if model_path is None:
        final = os.path.join(MODELS_DIR, "ppo_centipede_final.zip")
        if os.path.exists(final):
            model_path = final
        else:
            checkpoints = glob.glob(os.path.join(CHECKPOINTS_DIR, "ppo_centipede_*_steps.zip"))
            if checkpoints:
                model_path = max(checkpoints, key=checkpoint_steps)
    if model_path is None:
        return None, None

    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        # Для чекпоинта берём .pkl с тем же числом шагов, иначе — финальный
        m = re.search(r"_(\d+)_steps\.zip$", model_path)
        if m:
            candidate = os.path.join(
                os.path.dirname(model_path), f"ppo_centipede_vecnormalize_{m.group(1)}_steps.pkl"
            )
            if os.path.exists(candidate):
                vecnorm_path = candidate
        if vecnorm_path is None:
            final_pkl = os.path.join(MODELS_DIR, "vecnormalize_final.pkl")
            if os.path.exists(final_pkl):
                vecnorm_path = final_pkl
            else:
                pkls = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pkl"))
                if pkls:
                    vecnorm_path = max(pkls, key=os.path.getmtime)
    return model_path, vecnorm_path


def read_meta(meta_path):
    """meta.json как dict; пустой dict, если меты нет или она битая."""
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def generalist_meta():
    """Мета универсальной модели (models/meta.json)."""
    return read_meta(os.path.join(MODELS_DIR, "meta.json"))


def default_segments():
    """Число сегментов из models/meta.json (сохраняется при обучении);
    6, если меты нет. Модель и среда обязаны совпадать по размеру тела."""
    return generalist_meta().get("n_segments", 6)


def warn_if_stale(meta, what):
    """Возвращает False для несовместимой модели и печатает причину отказа."""
    if meta.get("reward_version") != REWARD_VERSION:
        print(f"ОШИБКА: {what} обучена старой версией среды "
              f"(meta reward_version={meta.get('reward_version')}, нужна {REWARD_VERSION}) — "
              f"размер/смысл наблюдений несовместим. Переобучите: python train.py")
        return False
    return True


def make_view_env(n_segments, max_episode_steps=1_000_000, reward_mode="command"):
    """Среда для просмотра: offscreen-рендер нужного размера, команды задаём сами.

    reward_mode обязан совпадать с режимом обучения модели: в режиме forward
    эффективная команда поворота всегда нулевая, поэтому статистика наблюдений
    командной модели будет другой. Пол — ровный (как «на столе»);
    в обучении бугры дают только устойчивость, показывать их незачем.
    """
    return make_centipede_env(
        n_segments, render_mode="rgb_array", max_episode_steps=max_episode_steps,
        auto_command_resample=False, reward_mode=reward_mode,
        terrain_roughness=0.0, width=VIEW_W, height=VIEW_H,
    )


def build_policy_stack(model_path, vecnorm_path, n_segments, reward_mode="command"):
    """(core_env для рендера/камеры, venv для нормализованных наблюдений, модель)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = make_view_env(n_segments, reward_mode=reward_mode)
    core = env.unwrapped
    venv = DummyVecEnv([lambda: env])
    if vecnorm_path:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    model = PPO.load(model_path, device="cpu")
    return core, venv, model


def pace(t0):
    """Держит реальное время: досыпает остаток такта управления (0.02 с),
    чтобы симуляция шла с той же скоростью, что и настоящий робот."""
    time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))


def run_keyboard(model_path, vecnorm_path, n_segments, reward_mode="command"):
    """Основной режим: стрелочки + обученная модель."""
    if reward_mode == "forward":
        print("Модель обучена только ходить вперёд (reward-mode forward) — "
              "стрелки поворота/назад работать не будут; для рулёжки обучите "
              "командную модель или специалистов.")
    core, venv, model = build_policy_stack(model_path, vecnorm_path, n_segments, reward_mode)
    win = SceneWindow(core, "Многоножка — стрелки: ехать, мышь/WASD: камера")
    obs = venv.reset()
    cmd_vx, cmd_wz, falls, info = 0.0, 0.0, 0, {}

    while not win.quit_requested:
        t0 = time.time()
        keys = win.process_events()
        if win.reset_requested:
            obs = venv.reset()
            cmd_vx, cmd_wz = 0.0, 0.0
            win.reset_requested = False

        cmd_vx, cmd_wz = update_drive(keys, cmd_vx, cmd_wz)
        core.set_command(cmd_vx, cmd_wz)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        if dones[0]:
            falls += 1
            cmd_vx, cmd_wz = 0.0, 0.0

        win.draw(status_rows(cmd_vx, cmd_wz, info, falls, win.follow))
        pace(t0)
    win.close()
    venv.close()


def find_skill_models():
    """Ищет обученных специалистов в models/skills/<навык>/.

    Возвращает словарь {навык: (модель.zip, vecnormalize.pkl | None)} только
    для навыков с финальной моделью, обученной ТЕКУЩЕЙ версией среды в
    командном режиме. Старые несовместимые модели пропускаются с
    предупреждением: они видели другую физику/наблюдения и в текущей среде
    сразу падают.
    """
    found, stale = {}, []
    for skill in SKILLS:
        base = os.path.join(SKILLS_DIR, skill)
        model_zip = os.path.join(base, "ppo_centipede_final.zip")
        if not os.path.exists(model_zip):
            continue
        meta = read_meta(os.path.join(base, "meta.json"))
        if meta.get("reward_version") != REWARD_VERSION or meta.get("reward_mode") != "command":
            stale.append(skill)
            continue
        vecnorm = os.path.join(base, "vecnormalize_final.pkl")
        found[skill] = (model_zip, vecnorm if os.path.exists(vecnorm) else None)
    if stale:
        print(f"Специалисты [{', '.join(stale)}] обучены старой версией среды — пропускаю. "
              f"Переобучите: python train.py <навык> (нужны reward_version={REWARD_VERSION}, "
              "reward_mode=command в их meta.json).")
    return found


class SkillPolicy:
    """Сеть-специалист: PPO-модель + её личная статистика нормализации.

    У каждого специалиста своя статистика VecNormalize (среднее/дисперсия
    наблюдений накапливались в его обучении), поэтому среда у нас одна,
    «сырая», а наблюдение нормализует та сеть, которая сейчас рулит.
    """

    def __init__(self, model_path, vecnorm_path):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device="cpu")
        self.vecnorm = None
        if vecnorm_path:
            # Из .pkl нужна только статистика наблюдений — venv не нужен
            with open(vecnorm_path, "rb") as f:
                self.vecnorm = pickle.load(f)
            self.vecnorm.training = False

    def act(self, obs):
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def select_skill(keys):
    """Стрелки -> имя сети-специалиста; None, если ни одна не нажата.

    Повороты старше хода: зажал <- и стрелку вверх — рулит сеть поворота
    и ведёт робота дугой (поворот через круг, как у машины).
    """
    if keys[pygame.K_LEFT]:
        return "left"
    if keys[pygame.K_RIGHT]:
        return "right"
    if keys[pygame.K_DOWN]:
        return "backward"
    if keys[pygame.K_UP]:
        return "forward"
    return None


def run_skills(skill_models, n_segments):
    """Режим 4 сетей: стрелка выбирает специалиста, он и рулит роботом.

    Среда одна и та же, переключается только сеть (и каждая нормализует
    наблюдение своей статистикой). При отпущенных кнопках последняя
    активная сеть отрабатывает команду "стоп".
    """
    print("Загружаю специалистов: " + ", ".join(skill_models))
    policies = {name: SkillPolicy(z, p) for name, (z, p) in skill_models.items()}

    env = make_view_env(n_segments)
    core = env.unwrapped
    win = SceneWindow(core, "Многоножка — 4 сети-специалиста (стрелки: ехать)")
    obs, _ = env.reset()
    cmd_vx, cmd_wz, falls, info = 0.0, 0.0, 0, {}
    active = "forward"  # кто держит робота, пока кнопки не нажаты

    while not win.quit_requested:
        t0 = time.time()
        keys = win.process_events()
        if win.reset_requested:
            obs, _ = env.reset()
            cmd_vx, cmd_wz = 0.0, 0.0
            win.reset_requested = False

        pressed = select_skill(keys)
        if pressed:
            active = pressed
        cmd_vx, cmd_wz = update_drive(keys, cmd_vx, cmd_wz)
        core.set_command(cmd_vx, cmd_wz)

        action = policies[active].act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            falls += 1
            obs, _ = env.reset()
            cmd_vx, cmd_wz = 0.0, 0.0

        header = (f"Сеть: {SKILL_RU[active]} [{active}]"
                  + ("" if pressed else "   (кнопки отпущены — держит стоп)"))
        win.draw(status_rows(cmd_vx, cmd_wz, info, falls, win.follow, header=header))
        pace(t0)
    win.close()
    env.close()


def run_manual_keyboard(n_segments):
    """Стрелочки без PPO: сервы двигает простой волновой контроллер (проверка механики)."""
    env = make_view_env(n_segments)
    core = env.unwrapped
    win = SceneWindow(core, "Многоножка manual (без модели) — стрелки: ехать")
    env.reset()
    cmd_vx, cmd_wz, phase, falls, info = 0.0, 0.0, 0.0, 0, {}

    print("Manual-режим: волновая походка без обученной модели — для проверки клавиш и механики.")
    while not win.quit_requested:
        t0 = time.time()
        keys = win.process_events()
        if win.reset_requested:
            env.reset()
            cmd_vx, cmd_wz, phase = 0.0, 0.0, 0.0
            win.reset_requested = False

        cmd_vx, cmd_wz = update_drive(keys, cmd_vx, cmd_wz)
        core.set_command(cmd_vx, cmd_wz)

        drive = max(abs(normalized_speed(cmd_vx)), abs(cmd_wz / MAX_WZ))
        if drive > 0.02:
            phase += CONTROL_DT * (2.5 + 5.5 * drive)
        action = manual_cpg_action(n_segments, phase, cmd_vx, cmd_wz)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            falls += 1
            env.reset()
            cmd_vx, cmd_wz, phase = 0.0, 0.0, 0.0

        win.draw(status_rows(cmd_vx, cmd_wz, info, falls, win.follow, header="Manual CPG (без обучения)"))
        pace(t0)
    win.close()
    env.close()


def run_auto(model_path, vecnorm_path, n_segments, reward_mode="command"):
    """Робот сам выполняет программу AUTO_SCRIPT по кругу; камера — на пользователе."""
    if reward_mode == "forward":
        print("Модель обучена только ходить вперёд — повороты автопрограммы она не отработает.")
    core, venv, model = build_policy_stack(model_path, vecnorm_path, n_segments, reward_mode)
    win = SceneWindow(core, "Многоножка — автопрограмма")
    obs = venv.reset()
    falls, info = 0, {}
    step_idx, t_in_step = 0, 0.0

    while not win.quit_requested:
        t0 = time.time()
        win.process_events()
        if win.reset_requested:
            obs = venv.reset()
            step_idx, t_in_step = 0, 0.0
            win.reset_requested = False

        name, duration, v_x, w_z = AUTO_SCRIPT[step_idx]
        core.set_command(v_x, w_z)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        if dones[0]:
            falls += 1

        t_in_step += CONTROL_DT
        if t_in_step >= duration:
            step_idx = (step_idx + 1) % len(AUTO_SCRIPT)
            t_in_step = 0.0

        win.draw(status_rows(v_x, w_z, info, falls, win.follow,
                             header=f"Автопрограмма: {name}  ({duration - t_in_step:.1f} с)"))
        pace(t0)
    win.close()
    venv.close()


def run_forward(model_path, vecnorm_path, n_segments, reward_mode="forward"):
    """Простой просмотр forward-only модели: постоянная команда идти вперёд."""
    core, venv, model = build_policy_stack(model_path, vecnorm_path, n_segments, reward_mode)
    win = SceneWindow(core, "Многоножка — постоянный ход вперёд")
    obs = venv.reset()
    falls, info = 0, {}

    while not win.quit_requested:
        t0 = time.time()
        win.process_events()
        if win.reset_requested:
            obs = venv.reset()
            win.reset_requested = False

        core.set_command(MAX_VX_FORWARD, 0.0)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        if dones[0]:
            falls += 1

        win.draw(status_rows(MAX_VX_FORWARD, 0.0, info, falls, win.follow,
                             header="Постоянная команда: вперёд"))
        pace(t0)
    win.close()
    venv.close()


def run_random(n_segments, episodes):
    """Случайные целевые углы серво — посмотреть тело без обучения."""
    env = make_view_env(n_segments, max_episode_steps=500)
    core = env.unwrapped
    win = SceneWindow(core, "Многоножка — случайные действия")
    env.reset()
    ep, total, info = 1, 0.0, {}

    while not win.quit_requested and ep <= episodes:
        t0 = time.time()
        win.process_events()
        if win.reset_requested:
            env.reset()
            total = 0.0
            win.reset_requested = False

        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        if terminated or truncated:
            print(f"Эпизод {ep}/{episodes}: награда {total:.1f}")
            ep += 1
            total = 0.0
            env.reset()

        win.draw([
            (f"Случайные действия — эпизод {ep}/{episodes}, награда {total:.1f}", C_WHITE),
            (f"факт: скорость {info.get('v_x_local', 0.0):+.2f} м/с", C_GREEN),
            (f"камера: {'следует' if win.follow else 'свободная'}", C_RED),
        ])
        pace(t0)
    win.close()
    env.close()


def main():
    """Разбирает аргументы командной строки и запускает нужный режим:
    --random / --manual / --auto / по умолчанию — стрелочки с моделью."""
    parser = argparse.ArgumentParser(description="Просмотр и управление роботом-многоножкой")
    parser.add_argument("--model", default=None, help="путь к .zip модели")
    parser.add_argument("--vecnorm", default=None, help="путь к .pkl нормализации")
    parser.add_argument("--random", action="store_true", help="случайные действия без модели")
    parser.add_argument("--manual", action="store_true", help="стрелки без модели: процедурная походка")
    parser.add_argument("--forward", action="store_true", help="постоянная команда вперёд для forward-only модели")
    parser.add_argument("--auto", action="store_true", help="автопрограмма команд вместо стрелочек")
    parser.add_argument("--single", action="store_true",
                        help="одна универсальная модель даже при обученных специалистах")
    parser.add_argument("--episodes", type=int, default=5, help="эпизодов в режиме --random")
    parser.add_argument("--segments", type=int, default=None,
                        help="число сегментов (по умолчанию из models/meta.json)")
    args = parser.parse_args()

    n_segments = args.segments if args.segments is not None else default_segments()

    if args.random:
        run_random(n_segments, args.episodes)
        return

    if args.manual:
        run_manual_keyboard(n_segments)
        return

    # 4 сети-специалиста: включаются сами, если обучены все четыре
    # (и пользователь не попросил --single / конкретную --model)
    if not args.single and not args.auto and not args.forward and args.model is None:
        skill_models = find_skill_models()
        if len(skill_models) == len(SKILLS):
            run_skills(skill_models, n_segments)
            return
        if skill_models:
            missing = [s for s in SKILLS if s not in skill_models]
            print(f"Специалисты обучены не все (не хватает: {', '.join(missing)}) — "
                  f"использую универсальную модель.")

    model_path, vecnorm_path = find_model_and_vecnorm(args)
    if model_path is None:
        print("Обученная модель не найдена в models/.")
        print("Запускаю manual-режим: стрелки будут двигать процедурную походку без PPO.")
        run_manual_keyboard(n_segments)
        return

    meta = generalist_meta()
    if not warn_if_stale(meta, "универсальная модель"):
        return
    reward_mode = meta.get("reward_mode") or "command"

    print(f"Модель:       {model_path}")
    print(f"Нормализация: {vecnorm_path or 'НЕ НАЙДЕНА (движения будут неадекватными!)'}")
    print(f"Сегментов:    {n_segments}, режим награды: {reward_mode}")

    if args.forward:
        run_forward(model_path, vecnorm_path, n_segments, reward_mode)
    elif args.auto:
        run_auto(model_path, vecnorm_path, n_segments, reward_mode)
    else:
        run_keyboard(model_path, vecnorm_path, n_segments, reward_mode)


if __name__ == "__main__":
    main()
