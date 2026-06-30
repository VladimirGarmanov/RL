"""
main.py — Ручной режим управления машинкой.

Запуск: python main.py

Управление:
    ↑  — газ
    ↓  — тормоз / задний ход
    ←  — поворот влево
    →  — поворот вправо
    R  — сброс позиции
    Esc — выход
"""

import sys
import math
import pygame

from track import Track, WINDOW_WIDTH, WINDOW_HEIGHT
from car import Car
from utils import draw_hud, draw_controls_help, make_font

FPS = 60


def get_action_from_keys(keys) -> int:
    gas   = keys[pygame.K_UP]
    brake = keys[pygame.K_DOWN]
    left  = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]

    if gas and left:  return 2
    if gas and right: return 3
    if gas:           return 1
    if brake:         return 4
    if left:          return 5
    if right:         return 6
    return 0


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("RL Car Racing — Manual Mode")
    clock = pygame.time.Clock()

    font       = make_font("monospace", 18)
    msg_font   = make_font("monospace", 26)
    small_font = make_font("monospace", 14)

    track = Track()
    sx, sy, sangle = track.get_start_position()
    car = Car(sx, sy, sangle)

    step           = 0
    checkpoint_idx = 1
    last_reward    = 0.0
    total_reward   = 0.0
    on_road        = True

    checkpoints  = track.get_checkpoints()
    n_checkpoints = len(checkpoints)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    car.reset(sx, sy, sangle)
                    step           = 0
                    checkpoint_idx = 1
                    last_reward    = 0.0
                    total_reward   = 0.0
                    on_road        = True

        keys   = pygame.key.get_pressed()
        action = get_action_from_keys(keys)
        car.update(action)
        step += 1

        # Collision: проверяем все углы машинки
        on_road = not car.is_any_corner_off_road(track)

        # Простой reward для отображения
        if on_road:
            last_reward = max(car.long_speed * 0.05, 0.0)
        else:
            last_reward = -10.0
        total_reward += last_reward

        # Чекпоинты
        if n_checkpoints > 0 and on_road:
            tx, ty, _ = checkpoints[checkpoint_idx % n_checkpoints]
            if math.hypot(car.x - tx, car.y - ty) < 52:
                checkpoint_idx = (checkpoint_idx + 1) % n_checkpoints
                total_reward  += 5.0

        # Обновляем лучи для отрисовки
        car.cast_rays(track)

        # Рисуем
        track.draw_with_active_checkpoint(screen, checkpoint_idx % n_checkpoints)
        car.draw(screen, draw_rays=True)

        # Оверлей при выезде с дороги
        if not on_road:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 35))
            screen.blit(overlay, (0, 0))
            txt = msg_font.render("OFF ROAD!  Press R to reset", True, (255, 60, 60))
            screen.blit(txt, (WINDOW_WIDTH//2 - txt.get_width()//2, WINDOW_HEIGHT//2 - 16))

        # HUD
        draw_hud(
            screen=screen,
            font=font,
            speed=car.long_speed,
            checkpoint=checkpoint_idx % n_checkpoints,
            total_checkpoints=n_checkpoints,
            reward=last_reward,
            step=step,
            mode="MANUAL",
        )

        # Полоска дрейфа
        from car import OBS_MAX_LAT_SPEED
        lat    = abs(car.lat_speed)
        bar_w  = int(min(lat / OBS_MAX_LAT_SPEED, 1.0) * 140)
        pygame.draw.rect(screen, (40, 40, 40),    (12, 168, 140, 12))
        if bar_w > 0:
            clr = (80, 150, 255) if lat < 2.0 else (255, 80, 80)
            pygame.draw.rect(screen, clr, (12, 168, bar_w, 12))
        pygame.draw.rect(screen, (180, 180, 180), (12, 168, 140, 12), 1)
        dtxt = small_font.render(f"Drift {lat:.1f}", True, (200, 200, 200))
        screen.blit(dtxt, (158, 167))

        # Общий reward
        tot_txt = small_font.render(f"Total reward: {total_reward:.1f}", True, (200, 255, 200))
        screen.blit(tot_txt, (12, 186))

        draw_controls_help(screen, font, WINDOW_WIDTH, WINDOW_HEIGHT)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
