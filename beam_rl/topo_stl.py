"""Генерация 3D-модели органической балки: best_topology.npy -> beam_organic.stl.

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ
--------------------
Берёт карту материала (сетку из 0 и 1), которую «вырезал» RL-агент
(topo_train.py), и превращает её в гладкую печатаемую 3D-модель:

  1. Половина балки зеркалится в полную (оптимизировали половину —
     задача симметрична относительно середины пролёта).
  2. Ступенчатая бинарная сетка превращается в ГЛАДКИЙ контур алгоритмом
     marching squares: значения плотности усредняются в узлы сетки, и
     граница проводится там, где узловое поле пересекает уровень ISO —
     получаются плавные диагонали вместо пиксельных ступенек.
  3. Плоский контур экструдируется (вытягивается) на толщину THICK_MM.

Балка печатается плашмя: каждый слой печати одинаковый, ни нависаний,
ни поддержек. Заодно выполняется МКЭ-проверка прочности итоговой
структуры (напряжения фон Мизеса, прогиб, запас прочности).

Запуск: python3 topo_stl.py  (нужен best_topology.npy от topo_train.py)
"""

import collections
import pathlib
import struct

import numpy as np

from topo_env import (SPAN_MM, HEIGHT_MM, THICK_MM, ELEM_MM, DESIGN_LOAD_N,
                      SIGMA_ULT_MPA, DENSITY_G_MM3, von_mises)
from make_stl import triangulate, write_stl, check_mesh

ISO = 0.45   # уровень изолинии: где узловое поле пересекает это значение,
             # там проходит граница детали (0.45 слегка «толстит» контур)


# ---- Зачистка полной (уже зеркалированной) сетки ------------------------------
def cleanup(d, seeds):
    """Оставляет только связный кусок структуры, достижимый из точек seeds.

    То же, что largest_component в topo_env, но для полной балки:
    после зеркалирования могли остаться изолированные острова материала —
    они висят в воздухе и не должны попасть в печать. Обход в ширину (BFS)
    по 4-соседству.
    """
    keep = np.zeros_like(d, dtype=bool)
    q = collections.deque(s for s in seeds if d[s])
    for s in q:
        keep[s] = True
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if 0 <= a < d.shape[0] and 0 <= b < d.shape[1] and d[a, b] \
                    and not keep[a, b]:
                keep[a, b] = True
                q.append((a, b))
    return d * keep


def node_field(d):
    """Переводит клеточные плотности в узловое поле (значения в углах клеток).

    Значение в узле = среднее плотностей примыкающих клеток (у внутреннего
    узла их 4, у краевого меньше — за краем считаем «ничего», отсюда
    массив-счётчик cnt). Это сглаживает пиксельные углы: узел между
    заполненной и пустой клеткой получает 0.5, и изолиния пройдёт
    посередине — граница станет плавной.

    Вход:  d (nx, ny) — клетки.  Выход: phi (nx+1, ny+1) — узлы.
    """
    nx, ny = d.shape
    pad = np.zeros((nx + 2, ny + 2))
    pad[1:-1, 1:-1] = d
    cnt = np.zeros((nx + 2, ny + 2))
    cnt[1:-1, 1:-1] = 1.0
    phi = np.zeros((nx + 1, ny + 1))
    w = np.zeros((nx + 1, ny + 1))
    # Каждый узел собирает вклад от 4 соседних клеток (сдвиги di, dj).
    for di in (0, 1):
        for dj in (0, 1):
            phi += pad[di:di + nx + 1, dj:dj + ny + 1]
            w += cnt[di:di + nx + 1, dj:dj + ny + 1]
    return phi / np.maximum(w, 1.0)


# ---- Marching squares с заливкой ----------------------------------------------
def cell_polys(v, corners):
    """Возвращает многоугольники «внутренней» части одной клетки.

    Классический marching squares: смотрим, какие из 4 углов клетки
    внутри детали (значение >= ISO), и строим соответствующий кусок
    заливки. В отличие от обычного варианта (только линия границы),
    здесь возвращается сам закрашенный многоугольник — из него потом
    делаются треугольники крышек.

    Вход:  v — 4 значения узлового поля в углах (обход CCW);
           corners — 4 координаты углов (тот же порядок).
    Выход: список многоугольников (обычно 0 или 1, при «седле» — 2).
    """
    inside = [x >= ISO for x in v]
    if not any(inside):
        return []                 # клетка целиком снаружи — пусто
    if all(inside):
        return [list(corners)]    # целиком внутри — весь квадрат

    def crossing(k):
        """Точка пересечения изолинии с ребром k (линейная интерполяция)."""
        a, b = v[k], v[(k + 1) % 4]
        t = (ISO - a) / (b - a)
        p0, p1 = corners[k], corners[(k + 1) % 4]
        return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))

    # Неоднозначный случай «седло»: два ПРОТИВОПОЛОЖНЫХ угла внутри.
    # Разрешаем по среднему значению клетки: если оно ниже ISO, углы
    # не соединены — два отдельных треугольничка.
    if inside == [True, False, True, False] and sum(v) / 4.0 < ISO:
        return [[corners[0], crossing(0), crossing(3)],
                [corners[2], crossing(2), crossing(1)]]
    if inside == [False, True, False, True] and sum(v) / 4.0 < ISO:
        return [[corners[1], crossing(1), crossing(0)],
                [corners[3], crossing(3), crossing(2)]]

    # Общий случай: обходим углы по кругу, добавляя внутренние углы и
    # точки пересечения изолинии с рёбрами — получается один многоугольник.
    poly = []
    for k in range(4):
        if inside[k]:
            poly.append(corners[k])
        if inside[k] != inside[(k + 1) % 4]:
            poly.append(crossing(k))
    return [poly]


def build_organic_mesh(density):
    """Строит замкнутый 3D-меш: гладкий 2D-контур + экструзия на THICK_MM.

    Этапы:
      1. Узловое поле -> marching squares по каждой клетке -> набор
         2D-треугольников заливки (tris2d).
      2. Верхняя крышка (z=THICK_MM) и нижняя (z=0) из этих треугольников.
      3. Боковые стенки: находим ГРАНИЧНЫЕ рёбра (ребро без парного
         обратного ребра — граница заливки) и вытягиваем каждое в
         вертикальный прямоугольник = 2 треугольника.

    Выход: (треугольники (N, 3, 3), площадь заливки в мм^2 — для
    контрольной проверки объёма).
    """
    phi = node_field(density)
    nx, ny = density.shape
    tris2d = []
    for i in range(nx):
        for j in range(ny):
            # 4 узловых значения и 4 координаты углов клетки (обход CCW).
            v = [phi[i, j], phi[i + 1, j], phi[i + 1, j + 1], phi[i, j + 1]]
            c = [(i * ELEM_MM, j * ELEM_MM), ((i + 1) * ELEM_MM, j * ELEM_MM),
                 ((i + 1) * ELEM_MM, (j + 1) * ELEM_MM),
                 (i * ELEM_MM, (j + 1) * ELEM_MM)]
            for poly in cell_polys(v, c):
                if len(poly) < 3:
                    continue
                # Многоугольник заливки -> треугольники (ушная триангуляция
                # из make_stl; она возвращает индексы, переводим в координаты).
                pts = np.array(poly)
                for (a, b, k) in triangulate(pts):
                    tris2d.append((tuple(pts[a]), tuple(pts[b]), tuple(pts[k])))

    # Ищем граничные рёбра. У внутреннего ребра обязательно есть
    # «обратное» ребро соседнего треугольника (B->A к нашему A->B);
    # у ребра на границе заливки обратного нет.
    def key(p):
        return (round(p[0], 6), round(p[1], 6))   # округление против ошибок float

    edge_set = set()
    for t in tris2d:
        for e in ((0, 1), (1, 2), (2, 0)):
            edge_set.add((key(t[e[0]]), key(t[e[1]])))

    tris = []
    # Крышки: верхняя с нормалью +Z (прямой порядок вершин), нижняя с
    # нормалью -Z (обратный порядок).
    for (a, b, c) in tris2d:
        tris.append(((*a, THICK_MM), (*b, THICK_MM), (*c, THICK_MM)))  # верх +Z
        tris.append(((*a, 0.0), (*c, 0.0), (*b, 0.0)))                 # низ -Z
    # Боковые стенки из граничных рёбер.
    for t in tris2d:
        for e in ((0, 1), (1, 2), (2, 0)):
            a, b = t[e[0]], t[e[1]]
            if (key(b), key(a)) in edge_set:
                continue                     # есть обратное ребро -> внутреннее
            a0, b0 = (*a, 0.0), (*b, 0.0)
            a1, b1 = (*a, THICK_MM), (*b, THICK_MM)
            tris.append((a0, b0, b1))
            tris.append((a0, b1, a1))

    # Площадь заливки (сумма площадей 2D-треугольников) — для проверки:
    # объём меша обязан равняться площадь * толщина.
    area = sum(abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
               for a, b, c in tris2d) / 2.0
    return np.array(tris, dtype=np.float64), area


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    half = np.load(here / "best_topology.npy")   # карта половины балки (60x24)
    nxh, ny = half.shape

    # МКЭ-проверка прочности — на той же полумодели, что и оптимизировали
    # (там уже настроены опоры и нагрузка).
    vm, defl = von_mises(half)
    smax = vm.max()                                       # пик напряжений, МПа
    sf = SIGMA_ULT_MPA / smax                             # фактический запас прочности
    # Какой груз можно повесить, сохраняя запас x3.
    safe_load_kg = DESIGN_LOAD_N * (SIGMA_ULT_MPA / 3.0) / smax / 9.81

    # Зеркалим половину в полную балку ([::-1] — отражение по X) и чистим
    # возможные изолированные куски (семена: точка груза и обе опоры).
    full = np.concatenate([half[::-1, :], half], axis=0)
    full = cleanup(full, [(full.shape[0] // 2, ny - 1), (0, 0),
                          (full.shape[0] - 1, 0)])

    # Строим меш, проверяем водонепроницаемость и объём, пишем STL.
    tris, area = build_organic_mesh(full)
    vol = check_mesh(tris, area * THICK_MM)
    out = here / "beam_organic.stl"
    write_stl(out, tris)

    # Сводка: габариты, масса, прочность.
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    print(f"STL: {out}")
    print(f"Треугольников: {len(tris)}, меш водонепроницаем, объём {vol/1000:.1f} см3")
    print(f"Габариты: {hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f} x {hi[2]-lo[2]:.0f} мм "
          f"(пролёт x высота x толщина)")
    print(f"Масса (100% PLA): {vol*DENSITY_G_MM3:.1f} г "
          f"(сплошная была бы {SPAN_MM*HEIGHT_MM*THICK_MM*DENSITY_G_MM3:.0f} г)")
    print(f"При 400 Н (40 кг): sigma_max {smax:.1f} МПа, прогиб {defl:.2f} мм, "
          f"запас прочности x{sf:.1f}")
    print(f"Нагрузка с запасом x3: ~{safe_load_kg:.0f} кг")
