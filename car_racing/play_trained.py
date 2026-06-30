"""
play_trained.py — Запуск обученной модели с визуализацией.

Запуск: python play_trained.py

Загружает сохранённую модель PPO и показывает,
как агент едет по трассе в реальном времени.

Нажми Esc или закрой окно, чтобы выйти.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from stable_baselines3 import PPO

from car_env import CarRacingEnv

_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "models", "ppo_car_racing")
NUM_EPISODES = 10   # сколько эпизодов показать (0 = бесконечно)
TRACK_VARIANT = None  # 0 = стандартная, 1 = восьмёрка, None = случайные из пула
RANDOM_START_PROBABILITY = 0.0


def main():
    model_file = MODEL_PATH + ".zip"

    if not os.path.exists(model_file):
        print(f"Модель не найдена: {model_file}")
        print("Сначала запусти обучение: python train.py")
        sys.exit(1)

    print(f"Загружаем модель: {model_file}")
    model = PPO.load(MODEL_PATH)

    # Создаём среду с визуализацией.
    # Для просмотра фиксируем стандартную трассу и старт, чтобы результат был понятным.
    env = CarRacingEnv(
        render_mode="human",
        render_substeps=3,
        track_variant=TRACK_VARIANT,
        random_start_probability=RANDOM_START_PROBABILITY,
        terminate_on_lap=True,
    )

    episode = 0
    total_episodes = NUM_EPISODES if NUM_EPISODES > 0 else float("inf")

    while episode < total_episodes:
        episode += 1
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0
        step = 0

        print(f"\n--- Эпизод {episode} ---")

        while not (terminated or truncated):
            # Агент выбирает действие детерминированно (без случайного исследования)
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            step += 1

            env.render()

            # Проверяем, не закрыл ли пользователь окно
            if env.screen is None:
                print("Окно закрыто.")
                env.close()
                sys.exit(0)

        terminal_reason = info.get("terminal_reason", "offroad")
        if not terminated:
            reason = f"лимит {step} шагов"
        elif terminal_reason == "missed_checkpoint":
            reason = "пропустил чекпоинт"
        elif terminal_reason == "wrong_checkpoint":
            reason = "не тот чекпоинт"
        elif terminal_reason == "lap_completed":
            reason = "круг завершён"
        else:
            reason = "выехал с дороги"

        checkpoints_passed = info.get("checkpoints_passed", 0)
        print(f"Эпизод {episode} завершён ({reason}). "
              f"Reward: {ep_reward:.2f}, Шагов: {step}, "
              f"Чекпоинтов: {checkpoints_passed}, "
              f"Скорость: {info.get('speed', 0):.1f}")

    print("\nГотово!")
    env.close()


if __name__ == "__main__":
    main()
