"""
track.py — Гоночная трасса через Catmull-Rom сплайн.

Трасса строится из 16 контрольных точек и интерполируется плавно.
Замкнутый контур без самопересечений: главная прямая + T1 (широкая дуга
вправо) + задняя прямая + T2 (шпилька вправо вверху) + верхняя прямая
+ T3 (широкая дуга влево) + возврат к старту.

Маска дороги: pygame.mask для O(1) проверки is_on_road(x, y).
"""

import pygame
import math
import numpy as np

WORLD_WIDTH  = 3000
WORLD_HEIGHT = 2000
SCREEN_W     = 1000   # размер экрана (для culling в draw)
SCREEN_H     = 700
MINIMAP_W    = 200
MINIMAP_H    = 133    # int(2000/3000*200)

# backward-compat aliases (car_env.py не импортирует их, но на всякий случай)
WINDOW_WIDTH  = WORLD_WIDTH
WINDOW_HEIGHT = WORLD_HEIGHT

# Цвета
COLOR_GRASS      = (30, 120, 30)
COLOR_DIRT       = (140, 100, 60)    # отбойник/зона вылета
COLOR_ROAD       = (75, 75, 75)
COLOR_KERB_RED   = (200, 30, 30)     # поребрик красный
COLOR_KERB_WHITE = (230, 230, 230)   # поребрик белый
COLOR_LINE       = (255, 255, 255)   # разметка
COLOR_STARTLINE  = (255, 200, 0)     # линия старта

ROAD_WIDTH = 82    # ширина полотна (в пикселях, базовая)


def generate_random_track(rng, center=(1500, 1000)):
    """
    Замкнутая трасса БЕЗ самопересечений через ряд Фурье в полярных координатах.

    r(θ) = 1 + Σ Aₖ·cos(k·θ + φₖ),  Σ|Aₖ| < 1  →  r(θ) > 0 всегда →
    кривая звездообразна → самопересечений нет.

    Гармоники и повороты:
        k=2,3  — общая форма (большие дуги, мало поворотов)
        k=4,5  — средние повороты (4-5 выступов на круг)
        k=6..10 — мелкие шиканы и шпильки (до 10 выступов)
    Чем выше k при заметной амплитуде — тем больше поворотов на трассе.
    """
    N = 28   # больше точек → Catmull-Rom точнее передаёт тугие повороты

    # Мир 3000×2000 → масштабируем базовый эллипс в ~3× от старых значений
    base_rx = float(rng.integers(870, 1110))
    base_ry = float(rng.integers(540, 735))

    # k=2,3 — урезаны: не даём им доминировать и делать просто овал
    # k=4..10 — основные "поворотообразующие" гармоники
    # Σ max_frac = 0.10+0.09+0.09+0.08+0.07+0.06+0.05+0.04+0.03 = 0.61 < 1.0
    harmonics = [2,    3,    4,    5,    6,    7,    8,    9,    10  ]
    max_fracs = [0.10, 0.09, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    min_fracs = [0.00, 0.00, 0.04, 0.03, 0.02, 0.02, 0.01, 0.00, 0.00]

    amps   = [float(rng.uniform(min_fracs[j], max_fracs[j])) for j in range(len(harmonics))]
    phases = [float(rng.uniform(0.0, 2.0 * math.pi)) for _ in harmonics]

    margin = 200
    waypoints = []
    for i in range(N):
        theta = 2.0 * math.pi * i / N
        r = 1.0 + sum(amps[j] * math.cos(harmonics[j] * theta + phases[j])
                      for j in range(len(harmonics)))
        x = center[0] + base_rx * r * math.cos(theta)
        y = center[1] + base_ry * r * math.sin(theta)
        waypoints.append((
            int(np.clip(x, margin, WORLD_WIDTH  - margin)),
            int(np.clip(y, margin, WORLD_HEIGHT - margin)),
        ))

    return waypoints

# F1-style circuit in WORLD (3000×2000).
# Inspired by Spa-Francorchamps layout: long pit straight, slow T1 hairpin,
# fast uphill S-curve, Kemmel straight, chicane, tight hairpin, final chicane.
F1_CIRCUIT_WAYPOINTS = [
    # PIT STRAIGHT (going right →, longest section ~2100 px)
    (300,  1800),   #  0  Start / Finish
    (750,  1810),   #  1
    (1250, 1810),   #  2
    (1750, 1805),   #  3
    (2150, 1790),   #  4
    (2400, 1765),   #  5  Braking zone
    # T1-T2: La Source — slow right hairpin
    (2560, 1700),   #  6
    (2660, 1590),   #  7
    (2710, 1450),   #  8  Apex
    (2670, 1310),   #  9  Exit
    (2560, 1200),   # 10
    # Eau Rouge / Raidillon — fast uphill S-curve
    (2430, 1080),   # 11
    (2290, 960),    # 12
    (2150, 840),    # 13
    (2065, 715),    # 14
    (2065, 595),    # 15  Apex
    (2125, 490),    # 16  Exit onto Kemmel
    # Kemmel straight (going left ←)
    (1960, 440),    # 17
    (1770, 415),    # 18
    (1570, 415),    # 19
    # Les Combes chicane (right-left-right)
    (1380, 440),    # 20  Entry
    (1225, 485),    # 21  Right
    (1065, 450),    # 22  Left
    (915,  480),    # 23  Right
    (785,  525),    # 24  Exit
    # T8: medium left
    (650,  625),    # 25
    (540,  745),    # 26
    # T9: Rivage — tight left hairpin
    (415,  875),    # 27  Entry
    (315,  1025),   # 28  Apex
    (325,  1175),   # 29  Exit
    # Blanchimont — fast sweeper going down-right
    (400,  1295),   # 30
    (510,  1395),   # 31
    (645,  1475),   # 32
    # T12: medium right
    (760,  1570),   # 33
    (830,  1655),   # 34
    # Bus stop chicane (final sector)
    (790,  1722),   # 35  Left
    (665,  1758),   # 36
    (540,  1778),   # 37  Right
    (420,  1795),   # 38  Exit onto pit straight
    # closes back to (300, 1800)
]

# Legacy small circuit (1000×700 world, kept for reference)
CIRCUIT_WAYPOINTS = [
    (155, 588),    # 0  — линия старта / главная прямая
    (370, 600),    # 1
    (590, 590),    # 2
    (745, 558),    # 3  — вход T1
    (848, 488),    # 4  — T1
    (892, 395),    # 5  — T1 апекс
    (875, 295),    # 6  — правая сторона
    (820, 218),    # 7  — вход верхнего правого угла
    (720, 172),    # 8  — T2 (шпилька вправо)
    (575, 148),    # 9  — верхняя прямая
    (425, 148),    # 10
    (290, 162),    # 11
    (188, 220),    # 12 — вход T3
    (130, 315),    # 13 — T3 апекс (широкая дуга влево)
    (138, 420),    # 14 — выход T3
    (162, 512),    # 15 — возврат к старту
]


def _segment_intersection(p1, p2, p3, p4):
    """Точка пересечения отрезков (p1,p2) и (p3,p4), или None."""
    d1x = p2[0] - p1[0]; d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]; d2y = p4[1] - p3[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / denom
    u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / denom
    if 0.0 < t < 1.0 and 0.0 < u < 1.0:
        return (p1[0] + t * d1x, p1[1] + t * d1y)
    return None


def _fix_polyline_crossings(pts):
    """
    Убирает самопересечения из открытой ломаной.
    Когда отрезок i→i+1 пересекает j→j+1, точки i+1…j заменяются
    точкой пересечения — «шип» превращается во внешний сглаженный контур.
    """
    result = list(pts)
    i = 0
    while i < len(result) - 2:
        found = False
        for j in range(i + 2, len(result) - 1):
            cross = _segment_intersection(result[i], result[i + 1],
                                          result[j], result[j + 1])
            if cross is not None:
                result = result[:i + 1] + [cross] + result[j + 1:]
                found = True
                break
        if not found:
            i += 1
    return result


def _catmull_rom(p0, p1, p2, p3, t: float):
    """Одна точка Catmull-Rom сплайна между p1 и p2 при параметре t ∈ [0,1)."""
    t2, t3 = t * t, t * t * t
    return (
        0.5 * (2*p1[0] + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3),
        0.5 * (2*p1[1] + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3),
    )


def _build_centerline(waypoints, steps_per_segment: int = 50):
    """
    Строим плавную центральную линию через Catmull-Rom сплайн.
    Замкнутая кривая: последняя точка соединяется с первой.
    """
    n = len(waypoints)
    points = []
    for i in range(n):
        p0 = waypoints[(i - 1) % n]
        p1 = waypoints[i]
        p2 = waypoints[(i + 1) % n]
        p3 = waypoints[(i + 2) % n]
        for s in range(steps_per_segment):
            t = s / steps_per_segment
            points.append(_catmull_rom(p0, p1, p2, p3, t))
    return points


def _build_figure8_centerline(center=(1500, 1000), rx=950, ry=480, n=400):
    """
    Центральная линия трассы-восьмёрки через кривую Лиссажу:
        x(t) = cx + rx·sin(t)
        y(t) = cy + ry·sin(2t)
    Кривая самопересекается в центре при t=0 и t=π.
    rx=330 и ry=185 дают восьмёрку 150..830 по x, 155..525 по y.
    """
    cx, cy = center
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        x = cx + rx * math.sin(t)
        y = cy + ry * math.sin(2.0 * t)
        pts.append((x, y))
    return pts


class Track:
    """
    Гоночная трасса.

    Публичный API:
        draw(screen)
        draw_with_active_checkpoint(screen, idx)
        is_on_road(x, y) -> bool
        get_start_position() -> (x, y, angle)
        get_checkpoints() -> list[(x, y, angle)]
        get_direction_at(x, y) -> float   # угол трассы в рад (для reward)
        get_progress_reward(x, y) -> float
    """

    def __init__(self, waypoints=None, road_width=None, figure8=False):
        """Строит трассу.

        waypoints  — контрольные точки контура (None = встроенная F1-трасса);
        road_width — ширина полотна в пикселях (None = стандартная);
        figure8    — True = трасса-«восьмёрка» с перекрёстком (waypoints
                     игнорируются, центр строится кривой Лиссажу).

        При создании сразу: строится плавная центральная линия, рисуется
        картинка трассы, готовится маска дороги (для быстрых проверок
        «на дороге ли точка») и расставляются 24 чекпоинта.
        """
        self._figure8   = figure8
        self.road_width = road_width if road_width is not None else ROAD_WIDTH

        if figure8:
            self._waypoints = None
            self.centerline = _build_figure8_centerline()
        else:
            self._waypoints = waypoints if waypoints is not None else F1_CIRCUIT_WAYPOINTS
            self.centerline = _build_centerline(self._waypoints)

        # numpy-массив для быстрого поиска ближайшей точки (O(N) но векторизованный)
        self._cl_np = np.array(self.centerline, dtype=np.float32)

        # Поверхность трассы (размер мира)
        self.surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
        self._draw_to_surface()

        # Маска: белые пиксели = дорога
        self.road_mask = self._build_road_mask()

        # Чекпоинты
        self.checkpoints = self._build_checkpoints(num=24)

    # ------------------------------------------------------------------
    # Построение
    # ------------------------------------------------------------------

    def _stroke_centerline(self, surface, color, radius):
        """
        Рисует центральную линию толстой кистью: линии между точками + кружки на стыках.
        Работает корректно для самопересекающихся трасс (восьмёрка).
        """
        pts = self.centerline
        n   = len(pts)
        r2  = radius * 2
        for i in range(n):
            p1 = (int(pts[i][0]),           int(pts[i][1]))
            p2 = (int(pts[(i + 1) % n][0]), int(pts[(i + 1) % n][1]))
            pygame.draw.line(surface, color, p1, p2, r2)
            pygame.draw.circle(surface, color, p1, radius)

    def _left_right_offsets(self):
        """
        Вычисляем левый и правый края дороги для каждой точки центральной линии.
        Возвращает (left_pts, right_pts).
        """
        pts = self.centerline
        n = len(pts)
        left, right = [], []
        hw = self.road_width / 2
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[(i + 1) % n]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length   # нормаль
            left.append( (pts[i][0] + nx * hw, pts[i][1] + ny * hw))
            right.append((pts[i][0] - nx * hw, pts[i][1] - ny * hw))
        left  = _fix_polyline_crossings(left)
        right = _fix_polyline_crossings(right)
        return left, right

    def _draw_to_surface(self):
        """Рисует всю трассу один раз в self.surface (потом только blit'ится):
        трава -> грунтовая обочина -> асфальт -> поребрики -> разметка -> старт."""
        self.surface.fill(COLOR_GRASS)
        hw = int(self.road_width / 2)

        # Stroke-based rendering works correctly for all track shapes,
        # including cases where polygon winding would fill the wrong region.
        self._stroke_centerline(self.surface, COLOR_DIRT, hw + 18)
        self._stroke_centerline(self.surface, COLOR_ROAD, hw)

        if self._figure8:
            self._draw_kerbs_figure8()
        else:
            left, right = self._left_right_offsets()
            self._draw_kerbs(left, right)
            pygame.draw.lines(self.surface, COLOR_LINE, True,
                              [(int(x), int(y)) for x, y in left],  2)
            pygame.draw.lines(self.surface, COLOR_LINE, True,
                              [(int(x), int(y)) for x, y in right], 2)

        self._draw_center_dashes()
        self._draw_start_line()

    def _draw_kerbs_figure8(self):
        """Поребрики для восьмёрки — цветные точки вдоль краёв."""
        left, right = self._left_right_offsets()
        step = 6
        for i in range(0, min(len(left), len(right)), step):
            color = COLOR_KERB_RED if (i // step) % 2 == 0 else COLOR_KERB_WHITE
            pygame.draw.circle(self.surface, color,
                               (int(left[i][0]),  int(left[i][1])),  4)
            pygame.draw.circle(self.surface, color,
                               (int(right[i][0]), int(right[i][1])), 4)

    def _draw_kerbs(self, left, right):
        """Поребрики — чередующиеся красно-белые полосы по краям."""
        n = len(left)
        kerb_len = 12   # пикселей на каждый сегмент поребрика
        for i in range(0, n, kerb_len * 2):
            seg_l = [(int(x), int(y)) for x, y in left[i:i+kerb_len]]
            seg_r = [(int(x), int(y)) for x, y in right[i:i+kerb_len]]
            if len(seg_l) >= 2 and len(seg_r) >= 2:
                pygame.draw.lines(self.surface, COLOR_KERB_RED,   False, seg_l, 5)
                pygame.draw.lines(self.surface, COLOR_KERB_RED,   False, seg_r, 5)
            seg_l2 = [(int(x), int(y)) for x, y in left[i+kerb_len:i+kerb_len*2]]
            seg_r2 = [(int(x), int(y)) for x, y in right[i+kerb_len:i+kerb_len*2]]
            if len(seg_l2) >= 2 and len(seg_r2) >= 2:
                pygame.draw.lines(self.surface, COLOR_KERB_WHITE, False, seg_l2, 5)
                pygame.draw.lines(self.surface, COLOR_KERB_WHITE, False, seg_r2, 5)

    def _draw_center_dashes(self):
        """Пунктирная жёлтая разметка центральной линии."""
        pts = self.centerline
        n = len(pts)
        dash = 18
        for i in range(0, n, dash * 2):
            seg = [(int(x), int(y)) for x, y in pts[i:i+dash]]
            if len(seg) >= 2:
                pygame.draw.lines(self.surface, (255, 200, 0), False, seg, 1)

    def _draw_start_line(self):
        """Шахматная линия старта."""
        if self._figure8:
            n   = len(self.centerline)
            idx = (n // 24) // 2   # середина между CP0 и CP1 — сразу после перекрёстка
            sx, sy = self.centerline[idx]
            nx, ny = self.centerline[(idx + 1) % n]
        else:
            sx, sy = self._waypoints[0]
            nx, ny = self._waypoints[1]
        dx, dy = nx - sx, ny - sy
        length = math.hypot(dx, dy) or 1.0
        # Нормаль — поперёк трассы
        perp_x, perp_y = -dy / length, dx / length

        hw = self.road_width / 2
        sq = 8  # размер клетки
        num_sq = int(self.road_width / sq)
        for i in range(num_sq):
            t = -hw + i * sq
            x1 = int(sx + perp_x * t)
            y1 = int(sy + perp_y * t)
            x2 = int(sx + perp_x * (t + sq))
            y2 = int(sy + perp_y * (t + sq))
            # Вдоль трассы — 2 ряда клеток
            hx_n, hy_n = dx / length, dy / length
            for row in range(2):
                rx = int(x1 + hx_n * row * sq)
                ry = int(y1 + hy_n * row * sq)
                color = COLOR_STARTLINE if (i + row) % 2 == 0 else (30, 30, 30)
                pygame.draw.rect(self.surface, color, (rx - sq//2, ry - sq//2, sq, sq))

    def _build_road_mask(self) -> pygame.mask.Mask:
        """
        Маска дороги для O(1) is_on_road.
        Покрывает асфальт + грунтовую зону вылета (dirt).
        Stroke-based — всегда корректна независимо от формы трассы.
        """
        mask_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        mask_surf.fill((0, 0, 0))
        mask_surf.set_colorkey((0, 0, 0))
        dirt_hw = int(self.road_width / 2) + 18
        self._stroke_centerline(mask_surf, (255, 255, 255), dirt_hw)
        return pygame.mask.from_surface(mask_surf)

    def _build_checkpoints(self, num: int):
        """num равноудалённых чекпоинтов вдоль трассы."""
        pts = self.centerline
        n   = len(pts)
        step = n // num
        checkpoints = []
        for i in range(num):
            idx      = (i * step) % n
            next_idx = (idx + 5) % n
            px, py   = pts[idx]
            nx, ny   = pts[next_idx]
            angle    = math.degrees(math.atan2(ny - py, nx - px))
            checkpoints.append((px, py, angle))
        return checkpoints

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def draw(self, screen: pygame.Surface, cam_x: float = 0, cam_y: float = 0):
        """Рисует трассу и кружки чекпоинтов (красный = старт, голубые — остальные).
        Чекпоинты за пределами экрана не рисуются (culling)."""
        screen.blit(self.surface, (-int(cam_x), -int(cam_y)))
        for i, (cx, cy, _) in enumerate(self.checkpoints):
            sx, sy = int(cx - cam_x), int(cy - cam_y)
            if -20 <= sx <= SCREEN_W + 20 and -20 <= sy <= SCREEN_H + 20:
                color = (255, 50, 50) if i == 0 else (0, 200, 255)
                pygame.draw.circle(screen, color, (sx, sy), 7, 2)

    def draw_with_active_checkpoint(self, screen: pygame.Surface, active_idx: int,
                                    cam_x: float = 0, cam_y: float = 0):
        """То же, что draw(), но текущий целевой чекпоинт выделен жёлтым
        и крупнее — видно, куда агент должен ехать."""
        screen.blit(self.surface, (-int(cam_x), -int(cam_y)))
        n = len(self.checkpoints)
        for i, (cx, cy, _) in enumerate(self.checkpoints):
            sx, sy = int(cx - cam_x), int(cy - cam_y)
            if -20 <= sx <= SCREEN_W + 20 and -20 <= sy <= SCREEN_H + 20:
                if i == active_idx % n:
                    color, r = (255, 255, 0), 11
                elif i == 0:
                    color, r = (255, 50, 50), 9
                else:
                    color, r = (0, 200, 255), 7
                pygame.draw.circle(screen, color, (sx, sy), r, 2)

    def is_on_road(self, x: float, y: float, margin: int = 0) -> bool:
        """Находится ли точка (x, y) на дороге.

        Основная проверка — один взгляд в заранее построенную маску (O(1)).
        margin > 0 добавляет допуск: точка считается «на дороге», если
        дорога есть в круге радиуса margin вокруг неё (нужно, чтобы
        белая линия разметки на краю не давала ложный вылет).
        """
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= WORLD_WIDTH or iy < 0 or iy >= WORLD_HEIGHT:
            return False
        if self.road_mask.get_at((ix, iy)):
            return True

        if margin <= 0:
            return False

        margin = int(margin)
        for dy in range(-margin, margin + 1):
            for dx in range(-margin, margin + 1):
                if dx * dx + dy * dy > margin * margin:
                    continue
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < WINDOW_WIDTH and 0 <= ny < WINDOW_HEIGHT:
                    if self.road_mask.get_at((nx, ny)):
                        return True
        return False

    def get_start_position(self):
        """Стартовая позиция: для восьмёрки — сразу после перекрёстка (CP0),
        чтобы следующий CP1 был впереди по ходу движения."""
        if self._figure8:
            n   = len(self.centerline)
            idx = (n // 24) // 2   # середина между CP0 (idx=0) и CP1 (idx=n//24)
            sx, sy = self.centerline[idx]
            nx, ny = self.centerline[(idx + 5) % n]
            angle = math.degrees(math.atan2(ny - sy, nx - sx))
            return sx, sy, angle
        sx, sy = self._waypoints[0]
        nx, ny = self._waypoints[1]
        angle = math.degrees(math.atan2(ny - sy, nx - sx))
        return sx, sy, angle

    def get_checkpoints(self):
        """Список чекпоинтов [(x, y, угол трассы в градусах), ...]."""
        return self.checkpoints

    def get_direction_at(self, x: float, y: float) -> float:
        """
        Возвращает направление трассы (угол в радианах) в точке ближайшей
        к (x, y) на центральной линии.

        Используется в reward: проекция скорости на направление трассы.
        Векторизованный numpy-поиск — быстро даже на 600+ точках.
        """
        dists = np.sum((self._cl_np - [x, y]) ** 2, axis=1)
        idx = int(np.argmin(dists))
        n = len(self.centerline)
        nx, ny = self.centerline[(idx + 1) % n]
        px, py = self.centerline[idx]
        return math.atan2(ny - py, nx - px)

    def get_progress_reward(self, x: float, y: float) -> float:
        """Близость к центру трассы (0..1). 1 = точно по центру."""
        dists = np.sum((self._cl_np - [x, y]) ** 2, axis=1)
        min_dist = float(np.sqrt(np.min(dists)))
        return max(0.0, 1.0 - min_dist / (self.road_width / 2))
