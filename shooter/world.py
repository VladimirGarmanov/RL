"""
world.py — Карта, препятствия, столкновения.

Мир полностью вмещается в экран: SCREEN_W = PX_W = 1200, SCREEN_H = PX_H = 750.
Камера не нужна, всегда передавай cam_x=0, cam_y=0.
"""

import pygame
import numpy as np
import math

TILE      = 30
WORLD_W   = 40
WORLD_H   = 25
PX_W      = WORLD_W * TILE   # 1200 px
PX_H      = WORLD_H * TILE   # 750 px
SCREEN_W  = PX_W
SCREEN_H  = PX_H

EMPTY = 0
ROCK  = 1


def _row(spans, width=WORLD_W):
    """Строка тайлов: spans = [(start_col, count), ...]"""
    r = ['.'] * width
    for start, count in spans:
        for c in range(start, min(start + count, width)):
            r[c] = 'R'
    return ''.join(r)


_E = '.' * WORLD_W

# Симметричная боевая арена.
# Агент спавнится в верхнем-левом углу, бот — в нижнем-правом.
_COMBAT_MAP = [
    _E,                         # 0
    _row([(1,3),(36,3)]),       # 1: угловые камни
    _E,                         # 2
    _row([(3,2),(35,2)]),       # 3
    _row([(3,2),(35,2)]),       # 4
    _E,                         # 5
    _row([(6,4),(30,4)]),       # 6
    _row([(6,4),(30,4)]),       # 7
    _E,                         # 8
    _row([(1,2),(37,2)]),       # 9
    _row([(1,2),(37,2)]),       # 10
    _E,                         # 11
    _row([(14,4),(22,4)]),      # 12: центр
    _E,                         # 13
    _row([(1,2),(37,2)]),       # 14
    _row([(1,2),(37,2)]),       # 15
    _E,                         # 16
    _row([(6,4),(30,4)]),       # 17
    _row([(6,4),(30,4)]),       # 18
    _E,                         # 19
    _row([(3,2),(35,2)]),       # 20
    _row([(3,2),(35,2)]),       # 21
    _E,                         # 22
    _row([(1,3),(36,3)]),       # 23
    _E,                         # 24
]


def _parse_rows(rows):
    grid = np.zeros((WORLD_H, WORLD_W), dtype=np.uint8)
    for r, line in enumerate(rows[:WORLD_H]):
        for c, ch in enumerate(line[:WORLD_W]):
            grid[r, c] = ROCK if ch == 'R' else EMPTY
    return grid


MAPS = [_parse_rows(_COMBAT_MAP)]


class World:
    """Тайловая карта арены: сетка WORLD_H x WORLD_W клеток по TILE пикселей.

    Каждая клетка — либо пусто (EMPTY), либо камень (ROCK). Все проверки
    столкновений сводятся к вопросу «какой тайл в этой точке?» — это
    дёшево и достаточно для аркадной физики.
    """

    def __init__(self, map_idx: int = 0):
        """Загружает карту и обводит её сплошной каменной рамкой по периметру
        (чтобы никто не выехал и не выстрелил за пределы мира)."""
        self.grid = MAPS[map_idx % len(MAPS)].copy()
        self.grid[0, :]  = ROCK
        self.grid[-1, :] = ROCK
        self.grid[:, 0]  = ROCK
        self.grid[:, -1] = ROCK

        self._surface = None   # создаётся лениво при первом draw()

    def _bake(self):
        """Отрисовывает карту один раз в закешированную поверхность:
        шахматная трава, сетка, камни с псевдо-3D фасками. Дальше draw()
        просто копирует готовую картинку — быстро."""
        for r in range(WORLD_H):
            for c in range(WORLD_W):
                x, y = c * TILE, r * TILE
                col = (32, 90, 32) if (r + c) % 2 == 0 else (28, 78, 28)
                pygame.draw.rect(self._surface, col, (x, y, TILE, TILE))

        for r in range(WORLD_H + 1):
            pygame.draw.line(self._surface, (24, 68, 24),
                             (0, r * TILE), (PX_W, r * TILE))
        for c in range(WORLD_W + 1):
            pygame.draw.line(self._surface, (24, 68, 24),
                             (c * TILE, 0), (c * TILE, PX_H))

        for r in range(WORLD_H):
            for c in range(WORLD_W):
                if self.grid[r, c] == ROCK:
                    x, y = c * TILE, r * TILE
                    pygame.draw.rect(self._surface, (105, 85, 68), (x, y, TILE, TILE))
                    pygame.draw.rect(self._surface, (95, 78, 62), (x+2, y+2, TILE-4, TILE-4))
                    pygame.draw.line(self._surface, (148, 125, 100),
                                     (x+1, y+1), (x+TILE-2, y+1), 2)
                    pygame.draw.line(self._surface, (148, 125, 100),
                                     (x+1, y+1), (x+1, y+TILE-2), 2)
                    pygame.draw.line(self._surface, (60, 48, 38),
                                     (x+1, y+TILE-2), (x+TILE-2, y+TILE-2), 2)
                    pygame.draw.line(self._surface, (60, 48, 38),
                                     (x+TILE-2, y+1), (x+TILE-2, y+TILE-2), 2)
                    pygame.draw.line(self._surface, (80, 65, 52),
                                     (x+6, y+5), (x+TILE-8, y+TILE-7), 1)

    def draw(self, screen: pygame.Surface, cam_x: float = 0, cam_y: float = 0):
        """Рисует карту (лениво запекая её при первом вызове)."""
        if self._surface is None:
            self._surface = pygame.Surface((PX_W, PX_H))
            self._bake()
        screen.blit(self._surface, (-int(cam_x), -int(cam_y)))

    def tile_at(self, px: float, py: float) -> int:
        """Тип тайла в пиксельной точке (px, py).
        За пределами карты — ROCK (мир как будто окружён камнем)."""
        c = int(px // TILE)
        r = int(py // TILE)
        if 0 <= r < WORLD_H and 0 <= c < WORLD_W:
            return int(self.grid[r, c])
        return ROCK

    def is_solid(self, px: float, py: float) -> bool:
        """Есть ли стена в этой точке (для пуль, лучей, линии огня)."""
        return self.tile_at(px, py) == ROCK

    def circle_collides(self, px: float, py: float, radius: float) -> bool:
        """Задевает ли круг (игрок) стену: проверяем 9 точек — центр,
        4 стороны и 4 диагонали на расстоянии radius."""
        for dx in (-radius, 0, radius):
            for dy in (-radius, 0, radius):
                if self.is_solid(px + dx, py + dy):
                    return True
        return False

    def slide_resolve(self, x, y, vx, vy, radius):
        """Движение со «скольжением» вдоль стен.

        Оси проверяются независимо: если движение по X упирается в стену —
        гасится только X-компонента, Y продолжает работать (и наоборот).
        Так игрок не «липнет» к стене, а скользит вдоль неё.

        Возвращает (новый x, новый y, новый vx, новый vy, был ли контакт).
        """
        nx, ny = x + vx, y + vy
        hit = False
        if self.circle_collides(nx, y, radius):
            vx = 0.0
            nx = x
            hit = True
        if self.circle_collides(x, ny, radius):
            vy = 0.0
            ny = y
            hit = True
        return nx, ny, vx, vy, hit

    def raycast(self, ox, oy, angle, max_dist=400.0, step=4.0):
        """Луч-дальномер: идёт из точки (ox, oy) под углом angle шагами
        по step пикселей и возвращает расстояние до первой стены
        (или max_dist, если стены не встретилось). Основа сенсоров игрока."""
        dx, dy = math.cos(angle), math.sin(angle)
        dist = 0.0
        while dist < max_dist:
            dist += step
            if self.is_solid(ox + dx * dist, oy + dy * dist):
                return dist
        return max_dist
