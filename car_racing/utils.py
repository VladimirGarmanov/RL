"""
utils.py — Вспомогательные функции для отрисовки UI.

На Python 3.14 pygame.font сломан (circular import в самом pygame).
Используем pygame.freetype как fallback через make_font().
"""

import pygame


class _FreetypeFontWrapper:
    """
    Обёртка вокруг pygame.freetype.Font, имитирующая API pygame.font.Font.
    Нужна только как fallback при сломанном pygame.font.
    """

    def __init__(self, ft_font):
        self._font = ft_font

    def render(self, text: str, antialias: bool, color) -> pygame.Surface:
        surf, _ = self._font.render(text, color)
        return surf

    def size(self, text: str):
        rect = self._font.get_rect(text)
        return rect.width, rect.height


def make_font(name: str = "monospace", size: int = 18):
    """
    Создаёт шрифт. Сначала пробует pygame.font, при ошибке — pygame.freetype.
    Возвращает объект с методом .render(text, antialias, color) -> Surface.
    """
    try:
        if not pygame.font.get_init():
            pygame.font.init()
        return pygame.font.SysFont(name, size)
    except Exception:
        pass

    # Fallback: pygame.freetype работает на Python 3.14
    import pygame.freetype
    if not pygame.freetype.get_init():
        pygame.freetype.init()
    ft = pygame.freetype.SysFont(name, size)
    return _FreetypeFontWrapper(ft)


def draw_text(screen: pygame.Surface, text: str, x: int, y: int,
              font, color=(255, 255, 255)):
    """Рисует текст на экране."""
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))


def draw_hud(screen: pygame.Surface, font,
             speed: float, checkpoint: int, total_checkpoints: int,
             reward: float, step: int, mode: str = ""):
    """
    Рисует HUD (heads-up display) в левом верхнем углу.
    """
    hud_surface = pygame.Surface((220, 150), pygame.SRCALPHA)
    hud_surface.fill((0, 0, 0, 160))
    screen.blit(hud_surface, (5, 5))

    lines = [
        f"Mode:       {mode}",
        f"Speed:      {speed:.2f}",
        f"Checkpoint: {checkpoint}/{total_checkpoints}",
        f"Reward:     {reward:+.3f}",
        f"Step:       {step}",
    ]

    for i, line in enumerate(lines):
        draw_text(screen, line, 12, 12 + i * 26, font, color=(220, 220, 220))


def draw_controls_help(screen: pygame.Surface, font,
                       window_width: int, window_height: int):
    """
    Рисует подсказку по управлению в правом нижнем углу.
    """
    lines = [
        "UP  Gas       DOWN  Brake",
        "LEFT  Left    RIGHT Right",
        "R  Reset      Esc  Quit",
    ]

    small_font = make_font("monospace", 14)
    padding = 8
    line_h = 18
    box_w = 220
    box_h = len(lines) * line_h + padding * 2

    bx = window_width - box_w - 10
    by = window_height - box_h - 10

    bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 140))
    screen.blit(bg, (bx, by))

    for i, line in enumerate(lines):
        draw_text(screen, line, bx + padding, by + padding + i * line_h,
                  small_font, color=(180, 180, 180))
