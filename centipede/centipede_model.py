"""
centipede_model.py — Процедурная генерация модели РОБОТА-многоножки в формате MJCF (MuJoCo XML).

Модель повторяет конструкцию, которую реально можно собрать. Список компонентов
на 6 сегментов (все размеры в модели — реальные габариты этих деталей):

    29 x сервопривод MG996R          40.7 x 19.7 x 42.9 мм, 55 г, 1.0 Н·м, 0.17 с/60°
     6 x шасси-пластина сегмента     110 x 84 мм (акрил/карбон), с АКБ и платой ~200 г
    12 x бедро: алюм. трубка/скоба   55 мм между осями серво
    12 x голень: алюм. трубка        ~95 мм до ступни
    12 x резиновая ступня-шарик      d=22 мм
     5 x пассивный шарнир наклона    между сегментами, с пружиной (подвеска)

Кинематика:
    - каждая нога = 2 серво: бедро (мах вперёд/назад) + колено (подъём стопы);
    - между сегментами серво поворота (руление) + пассивный подпружиненный наклон;
    - серво позиционные: команда = целевой угол, момент ограничен SERVO_TORQUE,
      скорость ограничена естественно через SERVO_KV
      (макс. скорость ~ TORQUE/KV ~ 6.7 рад/с ~ 0.16 с/60°, как в даташите MG996R).

Управление — 50 Гц, как стандартный PWM хобби-сервоприводов.
Итог: 6 сегментов, 12 ног, 29 серво, ~3.3 кг, ~0.75 м.

Запуск напрямую печатает статистику модели:  python centipede_model.py
"""

import os

# --- Шасси сегмента (половинные размеры бокса, метры) ---
CHASSIS_X = 0.055     # вдоль тела: пластина 110 мм
CHASSIS_Y = 0.042     # поперёк: 84 мм
CHASSIS_Z = 0.016     # толщина стека: пластина + АКБ + плата = 32 мм
CHASSIS_MASS = 0.20   # рама + аккумулятор + электроника + спинной серво (кг)
SEG_SPACING = 0.128   # расстояние между центрами сегментов
SPAWN_HEIGHT = 0.10   # начальная высота шасси (стопы касаются пола)

# --- Сервопривод MG996R: реальные габариты 40.7 x 19.7 x 42.9 мм ---
SERVO_HALF_L = 0.0204  # половина длины корпуса
SERVO_HALF_W = 0.0099  # половина ширины
SERVO_HALF_H = 0.0215  # половина высоты (вдоль вала)
SERVO_TORQUE = 1.0     # Н·м, предел момента (MG996R @ 6В: ~0.9-1.1)
SERVO_KP = 12.0        # жёсткость позиционного регулятора, Н·м/рад
SERVO_KV = 0.15        # демпфирование => предел скорости ~ TORQUE/KV = 6.7 рад/с
SERVO_MASS = 0.055     # кг

# --- Нога ---
HIP_Y = 0.052         # вынос оси бедра от центра шасси вбок
FEMUR_LEN = 0.055     # бедро: 55 мм между осями hip- и knee-серво
TIBIA_OUT = 0.035     # голень: вынос вбок
TIBIA_DOWN = 0.085    # голень: вниз до ступни (~95 мм по трубке)
FOOT_R = 0.011        # резиновая ступня-шарик d=22 мм
FEMUR_MASS = 0.02
TIBIA_MASS = 0.02
FOOT_MASS = 0.005

# --- Диапазоны суставов (градусы) ---
HIP_RANGE = 35          # мах бедра вперёд/назад
KNEE_RANGE = 45         # подъём/опускание стопы
SPINE_YAW_RANGE = 32    # изгиб позвоночника в стороны (активный, руление)
SPINE_PITCH_RANGE = 12  # наклон между сегментами (пассивная подвеска)

# Управление: dt физики * FRAME_SKIP = период команд сервоприводам
PHYSICS_TIMESTEP = 0.005
FRAME_SKIP = 4
CONTROL_DT = PHYSICS_TIMESTEP * FRAME_SKIP  # 0.02 c -> 50 Гц, как PWM серво

# --- Шероховатая поверхность (heightfield) ---
# Сетка высот поверх пола: пологие бугры высотой до terrain_roughness (задаётся
# в среде, по умолчанию 8 мм). Сами высоты заполняет CentipedeEnv при создании —
# у каждого параллельного процесса обучения свой случайный рельеф.
TERRAIN_HALF = 5.0    # полуразмер поля неровностей, м (10 x 10 м)
TERRAIN_RES = 128     # узлов сетки на сторону (ячейка ~8 см)

# --- Цвета (RGBA) ---
PLASTIC = "0.22 0.24 0.27 1"       # тёмный пластик шасси
PLASTIC_HEAD = "0.30 0.32 0.36 1"  # голова чуть светлее
ACCENT = "0.95 0.50 0.10 1"        # оранжевая пластина-маркер головы
SERVO_BLACK = "0.10 0.10 0.11 1"   # корпуса сервоприводов
ALUMINUM = "0.72 0.74 0.78 1"      # трубки ног
RUBBER = "0.05 0.05 0.05 1"        # ступни
TABLE_EDGE = "0.24 0.22 0.20 1"
MARK_BLUE = "0.10 0.32 0.62 1"
MARK_ORANGE = "0.95 0.45 0.10 1"
MARK_WHITE = "0.86 0.88 0.84 1"
RIB_RGBA = "0.36 0.33 0.29 1"

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(_HERE, "assets")


def _leg_xml(prefix: str, sgn: int) -> str:
    """XML одной ноги. sgn = +1 для левой (+Y), -1 для правой (-Y).

    Оси подобраны так, что одинаковая команда значит одно и то же для обеих
    сторон: +1 = нога вперёд (бедро) / стопа вверх (колено).
    """
    return f"""
          <body name="{prefix}_hip_body" pos="0 {sgn * HIP_Y:g} 0">
            <joint name="{prefix}_hip" class="leg" axis="0 0 {-sgn}" range="-{HIP_RANGE} {HIP_RANGE}"/>
            <geom name="{prefix}_hip_servo" class="visual" type="box" size="{SERVO_HALF_L:g} {SERVO_HALF_W:g} {SERVO_HALF_H:g}" rgba="{SERVO_BLACK}" mass="{SERVO_MASS}"/>
            <geom name="{prefix}_femur" type="capsule" fromto="0 0 0 0 {sgn * FEMUR_LEN:g} 0" size="0.008" rgba="{ALUMINUM}" mass="{FEMUR_MASS}"/>
            <body name="{prefix}_knee_body" pos="0 {sgn * FEMUR_LEN:g} 0">
              <joint name="{prefix}_knee" class="leg" axis="{sgn} 0 0" range="-{KNEE_RANGE} {KNEE_RANGE}"/>
              <geom name="{prefix}_knee_servo" class="visual" type="box" size="{SERVO_HALF_W:g} {SERVO_HALF_L:g} {SERVO_HALF_H:g}" rgba="{SERVO_BLACK}" mass="{SERVO_MASS}"/>
              <geom name="{prefix}_tibia" type="capsule" fromto="0 0 0 0 {sgn * TIBIA_OUT:g} {-TIBIA_DOWN:g}" size="0.006" rgba="{ALUMINUM}" mass="{TIBIA_MASS}"/>
              <geom name="{prefix}_foot" class="foot" type="sphere" pos="0 {sgn * TIBIA_OUT:g} {-TIBIA_DOWN:g}" size="{FOOT_R}" rgba="{RUBBER}" mass="{FOOT_MASS}"/>
            </body>
          </body>"""


def _table_scene_xml() -> str:
    """Визуальная сцена вокруг робота: стол, линейка и низкие боковые ребра."""
    marks = []
    for idx, x in enumerate(range(-7, 8)):
        rgba = MARK_ORANGE if x == 0 else (MARK_BLUE if idx % 2 == 0 else MARK_WHITE)
        marks.append(
            f'<geom name="table_x_mark_{idx}" class="visual" type="box" '
            f'pos="{x * 0.5:g} 0 0.0015" size="0.008 2.8 0.0015" rgba="{rgba}"/>'
        )
    for idx, y in enumerate((-2.0, -1.0, 1.0, 2.0)):
        marks.append(
            f'<geom name="table_y_mark_{idx}" class="visual" type="box" '
            f'pos="0 {y:g} 0.0017" size="4.0 0.010 0.0015" rgba="{MARK_WHITE}"/>'
        )
    for idx, x in enumerate((-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)):
        marks.append(
            f'<geom name="side_rib_left_{idx}" type="box" pos="{x:g} 1.55 0.003" '
            f'size="0.16 0.025 0.003" rgba="{RIB_RGBA}" friction="1.2 0.1 0.02"/>'
        )
        marks.append(
            f'<geom name="side_rib_right_{idx}" type="box" pos="{x:g} -1.55 0.003" '
            f'size="0.16 0.025 0.003" rgba="{RIB_RGBA}" friction="1.2 0.1 0.02"/>'
        )
    return "\n    ".join(marks)


def build_centipede_xml(n_segments: int = 6) -> str:
    """Собирает полный MJCF-документ робота из n_segments сегментов."""
    if n_segments < 2:
        raise ValueError("Многоножке нужно минимум 2 сегмента")

    # Тело собираем от хвоста к голове: каждый следующий сегмент оборачивает предыдущий
    body_tree = ""
    for i in range(n_segments - 1, -1, -1):
        is_head = i == 0
        legs = _leg_xml(f"seg{i}_left", +1) + _leg_xml(f"seg{i}_right", -1)

        if is_head:
            pos = f"0 0 {SPAWN_HEIGHT:g}"
            joints = '<freejoint name="root"/>'
            extra = (f'\n          <geom name="head_plate" class="visual" type="box" pos="0.025 0 {CHASSIS_Z + 0.004:g}" '
                     f'size="0.025 0.03 0.004" rgba="{ACCENT}" mass="0.005"/>')
            chassis_color = PLASTIC_HEAD
        else:
            pos = f"{-SEG_SPACING:g} 0 0"
            jp = SEG_SPACING / 2  # сустав на середине промежутка между сегментами
            joints = (
                f'<joint name="spine{i}_yaw" class="spine_yaw" pos="{jp:g} 0 0" axis="0 0 1" range="-{SPINE_YAW_RANGE} {SPINE_YAW_RANGE}"/>\n'
                f'          <joint name="spine{i}_pitch" class="spine_pitch" pos="{jp:g} 0 0" axis="0 1 0" range="-{SPINE_PITCH_RANGE} {SPINE_PITCH_RANGE}"/>'
            )
            # Корпус спинного сервопривода над стыком сегментов
            extra = (f'\n          <geom name="spine{i}_servo" class="visual" type="box" pos="{jp:g} 0 {CHASSIS_Z:g}" '
                     f'size="{SERVO_HALF_W:g} {SERVO_HALF_L:g} {SERVO_HALF_H:g}" rgba="{SERVO_BLACK}" mass="{SERVO_MASS}"/>')
            chassis_color = PLASTIC

        body_tree = f"""
        <body name="seg{i}" pos="{pos}">
          {joints}
          <geom name="seg{i}_chassis" type="box" size="{CHASSIS_X:g} {CHASSIS_Y:g} {CHASSIS_Z:g}" rgba="{chassis_color}" mass="{CHASSIS_MASS}"/>{extra}{legs}{body_tree}
        </body>"""

    # Сервоприводы — от головы к хвосту; порядок задаёт размерность действия среды.
    # Пассивные pitch-суставы позвоночника приводов не имеют (это подвеска).
    servos = []
    for i in range(n_segments):
        if i > 0:
            servos.append(f'<position name="spine{i}_yaw" joint="spine{i}_yaw"/>')
        for side in ("left", "right"):
            servos.append(f'<position name="seg{i}_{side}_hip" joint="seg{i}_{side}_hip"/>')
            servos.append(f'<position name="seg{i}_{side}_knee" joint="seg{i}_{side}_knee"/>')
    servo_block = "\n    ".join(servos)
    table_scene = _table_scene_xml()

    return f"""<mujoco model="centipede-robot-{n_segments}">
  <compiler angle="degree"/>
  <option timestep="{PHYSICS_TIMESTEP}"/>

  <default>
    <joint armature="0.01" damping="0.03"/>
    <geom condim="3" friction="1.0 0.1 0.01"/>
    <position kp="{SERVO_KP}" kv="{SERVO_KV}" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}" inheritrange="1.0"/>
    <default class="leg">
      <joint damping="0.05"/>
    </default>
    <default class="spine_yaw">
      <joint damping="0.05"/>
    </default>
    <default class="spine_pitch">
      <joint stiffness="5" damping="0.3"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0"/>
    </default>
    <default class="foot">
      <geom friction="1.4 0.1 0.02"/>
    </default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.65 0.95" rgb2="0.10 0.15 0.30" width="256" height="256"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.22 0.30 0.38" rgb2="0.30 0.40 0.48" width="512" height="512"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="24 24" reflectance="0.15"/>
    <material name="table_mat" rgba="0.54 0.50 0.43 1" reflectance="0.08"/>
    <hfield name="terrain" nrow="{TERRAIN_RES}" ncol="{TERRAIN_RES}" size="{TERRAIN_HALF} {TERRAIN_HALF} 1.0 0.02"/>
  </asset>

  <worldbody>
    <light directional="true" pos="0 0 8" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.2 0.2 0.2"/>
    <geom name="terrain" type="hfield" hfield="terrain" material="grid_mat" friction="1.0 0.1 0.01"/>
    <geom name="floor" type="plane" size="50 50 0.5" pos="0 0 -0.005" material="grid_mat" friction="1.0 0.1 0.01"/>
    <geom name="tabletop" class="visual" type="box" pos="0 0 -0.014" size="4.5 3.0 0.012" material="table_mat"/>
    <geom name="table_front_edge" class="visual" type="box" pos="0 -3.02 -0.012" size="4.55 0.035 0.035" rgba="{TABLE_EDGE}"/>
    <geom name="table_back_edge" class="visual" type="box" pos="0 3.02 -0.012" size="4.55 0.035 0.035" rgba="{TABLE_EDGE}"/>
    <geom name="table_left_edge" class="visual" type="box" pos="-4.52 0 -0.012" size="0.035 3.0 0.035" rgba="{TABLE_EDGE}"/>
    <geom name="table_right_edge" class="visual" type="box" pos="4.52 0 -0.012" size="0.035 3.0 0.035" rgba="{TABLE_EDGE}"/>
    {table_scene}
    {body_tree}
  </worldbody>

  <actuator>
    {servo_block}
  </actuator>
</mujoco>
"""


def ensure_xml(n_segments: int = 6) -> str:
    """Записывает XML в assets/ и возвращает абсолютный путь к файлу.

    Запись атомарная (temp-файл + rename): параллельные процессы обучения
    создают среды одновременно, и никто не должен прочитать недописанный файл.
    """
    os.makedirs(ASSETS_DIR, exist_ok=True)
    xml_path = os.path.join(ASSETS_DIR, f"centipede_{n_segments}seg.xml")
    tmp_path = f"{xml_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        f.write(build_centipede_xml(n_segments))
    os.replace(tmp_path, xml_path)
    return xml_path


if __name__ == "__main__":
    import mujoco
    import numpy as np

    model_path = ensure_xml(6)
    m = mujoco.MjModel.from_xml_path(model_path)
    print(f"Модель записана: {model_path}")
    print(f"Тел: {m.nbody}, суставов (DoF): {m.nv}, сервоприводов: {m.nu}")
    print(f"Общая масса: {sum(m.body_mass):.2f} кг, длина ~{SEG_SPACING * 5 + CHASSIS_X * 2:.2f} м")
    print(f"Частота управления: {1 / CONTROL_DT:.0f} Гц")
    print("Диапазоны команд серво (рад):")
    print(np.round(m.actuator_ctrlrange[:5], 3))
