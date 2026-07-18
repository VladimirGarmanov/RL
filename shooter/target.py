"""
target.py — Мишень с пульсирующими кольцами (AA-графика).
"""

import pygame
import pygame.gfxdraw
import math

TARGET_RADIUS = 18


class Target:
    """Неподвижная мишень (использовалась в ранних версиях среды для
    обучения чистой стрельбе; в текущем self-play бою не участвует)."""

    def __init__(self, x: float, y: float):
        self.x     = x
        self.y     = y
        self.alive = True
        self._tick = 0     # для анимации пульса

    def draw(self, screen: pygame.Surface, cam_x: float = 0, cam_y: float = 0):
        """Рисует мишень: концентрические кольца с плавной пульсацией
        (синус по счётчику кадров) + прицельные линии."""
        if not self.alive:
            return

        self._tick += 1
        pulse = int(math.sin(self._tick * 0.07) * 3)

        sx, sy = int(self.x - cam_x), int(self.y - cam_y)
        r = TARGET_RADIUS + pulse

        # Тень
        sh = pygame.Surface((r*2+16, r*2+16), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(sh, r+8, r+8, r+4, (0, 0, 0, 50))
        screen.blit(sh, (sx - r - 8, sy - r - 4))

        # Внешнее кольцо (тёмное)
        pygame.gfxdraw.aacircle(screen, sx, sy, r + 2, (160, 20, 20))
        pygame.gfxdraw.aacircle(screen, sx, sy, r + 3, (100, 10, 10))

        # Заливка
        pygame.gfxdraw.filled_circle(screen, sx, sy, r, (200, 40, 40))
        pygame.gfxdraw.aacircle(screen, sx, sy, r, (220, 50, 50))

        # Среднее кольцо (белое)
        pygame.gfxdraw.aacircle(screen, sx, sy, r - 5, (255, 220, 220))
        pygame.gfxdraw.filled_circle(screen, sx, sy, r - 6, (210, 45, 45))

        # Внутренний круг (красный)
        pygame.gfxdraw.filled_circle(screen, sx, sy, r - 11, (235, 55, 55))
        pygame.gfxdraw.aacircle(screen, sx, sy, r - 11, (255, 80, 80))

        # Яблочко
        pygame.gfxdraw.filled_circle(screen, sx, sy, 5, (255, 240, 240))
        pygame.gfxdraw.aacircle(screen, sx, sy, 5, (255, 255, 255))

        # Прицельные линии
        pygame.draw.aaline(screen, (255, 180, 180), (sx - r - 3, sy), (sx + r + 3, sy))
        pygame.draw.aaline(screen, (255, 180, 180), (sx, sy - r - 3), (sx, sy + r + 3))
