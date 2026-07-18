"""
player.py — Игрок: кружок с пушкой.

Механика:
  SPIN   — стоит, вращается с постоянной угловой скоростью.
  MOVING — удерживаешь ПРОБЕЛ: едет вперёд в зафиксированном направлении,
            НЕ крутится. Отпустил пробел — останавливается, возвращается в SPIN.

Actions:
  0 — ничего
  1 — MOVE удерживается
  2 — SHOOT

color_scheme:
  'blue' — агент (синий/зелёный, жёлтые пули)
  'red'  — бот   (красный/оранжевый, красные пули)
"""

import pygame
import pygame.gfxdraw
import math

RADIUS        = 14
GUN_LEN       = 24
ANGULAR_SPEED = 3.0
MOVE_SPEED    = 4.5
MAX_HP        = 3

SPIN   = 0
MOVING = 1


def _aa_circle(surf, color, cx, cy, r):
    """Сглаженный залитый круг: заливка + антиалиасный контур поверх."""
    if r < 1:
        return
    pygame.gfxdraw.filled_circle(surf, cx, cy, r, color)
    pygame.gfxdraw.aacircle(surf, cx, cy, r, color)


class Player:
    """Игрок-«танчик»: круг с пушкой, два состояния (SPIN / MOVING).

    Одинаковый класс для агента и бота — различаются только цветом.
    Вся «сложность» управления в том, что пушка либо крутится
    (стоя на месте), либо зафиксирована (в движении) — попадать можно,
    только правильно выбирая моменты.
    """

    def __init__(self, x: float, y: float, angle: float = 0.0,
                 color_scheme: str = 'blue'):
        self.x     = x
        self.y     = y
        self.angle = angle
        self.vx    = 0.0
        self.vy    = 0.0
        self.state = SPIN
        self.alive = True
        self.hp    = MAX_HP
        self.max_hp = MAX_HP

        self._color_scheme  = color_scheme
        self._locked_angle  = angle
        self.shoot_cooldown = 0
        self._flash         = 0   # кадры белой вспышки при попадании

    # ------------------------------------------------------------------
    # Физика
    # ------------------------------------------------------------------

    def update(self, action: int, world) -> list:
        """Один тик игрока.

        action: 0 = ничего (стоим, пушка крутится), 1 = MOVE удерживается
        (едем в направлении, зафиксированном в момент нажатия), 2 = SHOOT.

        Логика состояний: первое нажатие MOVE в состоянии SPIN «замораживает»
        текущий угол пушки и переводит в MOVING; движение идёт строго по
        замороженному углу; врезался в стену или отпустил MOVE — обратно
        в SPIN, и пушка снова крутится.

        Возвращает список созданных за тик пуль (пустой или из одной).
        """
        from bullet import Bullet
        bullets = []

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self._flash > 0:
            self._flash -= 1

        move_held = (action == 1)
        shoot     = (action == 2)

        if move_held:
            if self.state == SPIN:
                self._locked_angle = self.angle
                self.state = MOVING

            self.angle = self._locked_angle
            rad = math.radians(self._locked_angle)
            self.vx = math.cos(rad) * MOVE_SPEED
            self.vy = math.sin(rad) * MOVE_SPEED

            nx, ny, self.vx, self.vy, hit = world.slide_resolve(
                self.x, self.y, self.vx, self.vy, RADIUS
            )
            self.x, self.y = nx, ny

            if hit:
                self.state = SPIN
                self.vx = self.vy = 0.0

        else:
            if self.state == MOVING:
                self.state = SPIN
                self.vx = self.vy = 0.0
            self.angle = (self.angle + ANGULAR_SPEED) % 360.0

        if shoot and self.shoot_cooldown == 0:
            rad = math.radians(self.angle)
            bx  = self.x + math.cos(rad) * (RADIUS + 6)
            by  = self.y + math.sin(rad) * (RADIUS + 6)
            trail = (220, 30, 30) if self._color_scheme == 'red' else (255, 140, 0)
            bullets.append(Bullet(bx, by, self.angle, trail))
            self.shoot_cooldown = 10

        return bullets

    def hit(self) -> bool:
        """Получить 1 урон: минус HP, белая вспышка на 8 кадров.
        Возвращает True, если игрок ещё жив."""
        self.hp -= 1
        self._flash = 8
        self.alive = self.hp > 0
        return self.alive

    def reset(self, x: float, y: float, angle: float = 0.0):
        """Возвращает игрока к стартовому состоянию (новый бой)."""
        self.x, self.y, self.angle = x, y, angle
        self._locked_angle  = angle
        self.vx = self.vy   = 0.0
        self.state          = SPIN
        self.alive          = True
        self.hp             = self.max_hp
        self.shoot_cooldown = 0
        self._flash         = 0

    # ------------------------------------------------------------------
    # Сенсоры
    # ------------------------------------------------------------------

    def cast_rays(self, world, n_rays: int = 8, max_dist: float = 300.0):
        """n_rays лучей-дальномеров равномерно по кругу (первый — по пушке).

        Возвращает нормированные расстояния до стен [0..1] — часть
        наблюдений RL-агента. Лучи привязаны к углу пушки, поэтому
        «картинка» вращается вместе с игроком.
        """
        dists = []
        for i in range(n_rays):
            angle_rad = math.radians(self.angle + 360.0 * i / n_rays)
            d = world.raycast(self.x, self.y, angle_rad, max_dist)
            dists.append(d / max_dist)
        return dists

    # ------------------------------------------------------------------
    # Отрисовка
    # ------------------------------------------------------------------

    def draw(self, screen: pygame.Surface, cam_x: float = 0, cam_y: float = 0):
        """Рисует игрока: тень, ствол, корпус, блик, HP-бар.

        Цвет кодирует состояние: синий/красный = SPIN (пушка крутится),
        зелёный/оранжевый = MOVING (едет); белая вспышка = только что
        получил урон.
        """
        sx, sy = int(self.x - cam_x), int(self.y - cam_y)
        rad    = math.radians(self.angle)

        # Тень
        sh = pygame.Surface((RADIUS*2+12, RADIUS*2+12), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(sh, RADIUS+6, RADIUS+6, RADIUS+3, (0, 0, 0, 55))
        screen.blit(sh, (sx - RADIUS - 3, sy - RADIUS + 3))

        # Ствол
        gx = int(sx + math.cos(rad) * (RADIUS - 2))
        gy = int(sy + math.sin(rad) * (RADIUS - 2))
        ex = int(sx + math.cos(rad) * GUN_LEN)
        ey = int(sy + math.sin(rad) * GUN_LEN)
        pygame.draw.line(screen, (120, 100, 30), (gx, gy), (ex, ey), 8)
        pygame.draw.line(screen, (210, 185, 60), (gx, gy), (ex, ey), 5)
        pygame.draw.line(screen, (255, 235, 110), (gx, gy), (ex, ey), 2)
        _aa_circle(screen, (220, 190, 55), ex, ey, 5)
        _aa_circle(screen, (255, 240, 120), ex, ey, 3)

        # Цвет тела
        if self._flash > 0:
            body_col  = (255, 255, 255)
            inner_col = (255, 255, 220)
            rim_col   = (200, 200, 200)
        elif self._color_scheme == 'red':
            body_col  = (215, 55, 45) if self.state == SPIN else (215, 120, 40)
            inner_col = (240, 80, 65) if self.state == SPIN else (240, 155, 60)
            rim_col   = (110, 18, 12) if self.state == SPIN else (110, 60, 10)
        else:
            body_col  = (50, 130, 215) if self.state == SPIN else (40, 195, 110)
            inner_col = (80, 160, 240) if self.state == SPIN else (70, 225, 145)
            rim_col   = (20,  55, 120) if self.state == SPIN else (15, 100,  55)

        _aa_circle(screen, rim_col,   sx, sy, RADIUS + 2)
        _aa_circle(screen, body_col,  sx, sy, RADIUS)
        _aa_circle(screen, inner_col, sx, sy, RADIUS - 4)

        # Блик
        hl = pygame.Surface((14, 14), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(hl, 5, 5, 4, (255, 255, 255, 110))
        screen.blit(hl, (sx - RADIUS//2 - 5, sy - RADIUS//2 - 5))

        # HP-бар
        bar_w  = RADIUS * 2 + 10
        bar_h  = 4
        filled = max(0, int(bar_w * self.hp / self.max_hp))
        bx     = sx - bar_w // 2
        by     = sy - RADIUS - 11
        pygame.draw.rect(screen, (70, 0, 0),    (bx, by, bar_w, bar_h))
        hp_col = (220, 60, 60) if self._color_scheme == 'red' else (60, 200, 80)
        pygame.draw.rect(screen, hp_col,         (bx, by, filled, bar_h))
        pygame.draw.rect(screen, (200, 200, 200),(bx, by, bar_w, bar_h), 1)
