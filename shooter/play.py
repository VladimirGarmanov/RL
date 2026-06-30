"""
play.py — Ручная игра или просмотр обученной модели.

Запуск (руками):  python play.py --human
Запуск (модель):  python play.py
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from shooter_env import ShooterEnv

_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "models", "ppo_shooter_v2")


def format_result(info: dict, terminated: bool, truncated: bool,
                  human: bool = False) -> str:
    result = info.get("result", "")
    if result == "win":
        return "ПОБЕДА!" if human else "WIN"
    if result == "lose":
        return "ПОРАЖЕНИЕ" if human else "LOSE"
    if result == "timeout_win":
        return "ПОБЕДА ПО HP" if human else "TIMEOUT_WIN"
    if result == "timeout_loss":
        return "ПОРАЖЕНИЕ ПО HP" if human else "TIMEOUT_LOSS"
    if result == "timeout_draw" or truncated:
        return "НИЧЬЯ" if human else "DRAW"
    if terminated:
        return "ПОБЕДА!" if info.get("win") and human else ("WIN" if info.get("win") else "LOSE")
    return "НИЧЬЯ" if human else "DRAW"


def play_human():
    env = ShooterEnv(render_mode="human")
    obs, _ = env.reset()

    print("=== RL SHOOTER — Агент vs Бот ===")
    print("  ПРОБЕЛ (держать) — двигаться вперёд (без вращения)")
    print("  ENTER            — выстрел")
    print("  ESC              — выход")
    print("  ТЫ — синий  |  БОТ — красный")

    env.render()

    running = True
    while running:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    action = 2

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and action == 0:
            action = 1

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            result = format_result(info, terminated, truncated, human=True)
            print(f"{result} | Попаданий: {info['agent_hits']} | Получено: {info['bot_hits']} | Шагов: {info['steps']}")
            obs, _ = env.reset()

    env.close()


def play_model():
    from stable_baselines3 import PPO

    model_file = MODEL_PATH + ".zip"
    if not os.path.exists(model_file):
        print(f"Модель не найдена: {model_file}")
        print("Сначала запусти: python train.py")
        sys.exit(1)

    model = PPO.load(MODEL_PATH)
    env   = ShooterEnv(render_mode="human")
    obs, _ = env.reset()
    env.render()

    episode = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                return

        if terminated or truncated:
            episode += 1
            result = format_result(info, terminated, truncated)
            print(f"Эп {episode}: {result} | agent_hits={info['agent_hits']} bot_hits={info['bot_hits']} steps={info['steps']}")
            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args()

    if args.human:
        play_human()
    else:
        play_model()
