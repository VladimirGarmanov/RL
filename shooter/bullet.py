"""
bullet.py — Пуля с AA-графикой и светящимся хвостом.
"""

import pygame
import pygame.gfxdraw
import math

BULLET_SPEED  = 13.0
BULLET_RADIUS = 4
MAX_LIFETIME  = 140


class Bullet:
    def __init__(self, x: float, y: float, angle: float,
                 trail_color: tuple = (255, 140, 0)):
        rad      = math.radians(angle)
        self.x   = x
        self.y   = y
        self.vx  = math.cos(rad) * BULLET_SPEED
        self.vy  = math.sin(rad) * BULLET_SPEED
        self.alive       = True
        self.lifetime    = 0
        self._trail: list[tuple[float, float]] = []
        self._trail_color = trail_color

    def update(self, world) -> bool:
        if not self.alive:
            return False
        self._trail.append((self.x, self.y))
        if len(self._trail) > 8:
            self._trail.pop(0)
        self.x += self.vx
        self.y += self.vy
        self.lifetime += 1
        if world.is_solid(self.x, self.y) or self.lifetime >= MAX_LIFETIME:
            self.alive = False
            return True
        return False

    def hits(self, cx: float, cy: float, radius: float) -> bool:
        return math.hypot(self.x - cx, self.y - cy) < radius + BULLET_RADIUS

    def draw(self, screen: pygame.Surface, cam_x: float = 0, cam_y: float = 0):
        if not self.alive:
            return

        # Светящийся хвост
        tc = self._trail_color
        n = len(self._trail)
        for i, (tx, ty) in enumerate(self._trail):
            progress = (i + 1) / n
            r  = max(1, int(BULLET_RADIUS * progress))
            cr = min(255, int(tc[0] * progress))
            cg = min(255, int(tc[1] * progress))
            cb = min(255, int(tc[2] * progress))
            sx, sy = int(tx - cam_x), int(ty - cam_y)
            pygame.gfxdraw.filled_circle(screen, sx, sy, r, (cr, cg, cb))

        # Ядро пули
        sx, sy = int(self.x - cam_x), int(self.y - cam_y)
        pygame.gfxdraw.filled_circle(screen, sx, sy, BULLET_RADIUS + 1, (255, 180, 0))
        pygame.gfxdraw.aacircle(screen, sx, sy, BULLET_RADIUS + 1, (255, 180, 0))
        pygame.gfxdraw.filled_circle(screen, sx, sy, BULLET_RADIUS - 1, (255, 245, 180))
