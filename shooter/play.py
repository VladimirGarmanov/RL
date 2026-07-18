"""
play.py — Запуск матча с настраиваемыми игроками.

Первый аргумент = синий (агент), второй = красный (бот).
Без аргументов — ai vs bot.

Примеры:
  python play.py human bot   # ты vs скриптовый бот
  python play.py human ai    # ты vs обученный RL агент
  python play.py ai bot      # RL агент vs скриптовый бот  [по умолчанию]
  python play.py ai ai       # RL агент vs RL агент (смотрим)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from shooter_env import ShooterEnv

_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "models", "ppo_shooter_v2")

PLAYER_CHOICES = ["human", "ai", "bot"]


def load_model():
    """Загружает обученную PPO-модель; если её нет — понятная ошибка и выход."""
    from stable_baselines3 import PPO
    model_file = MODEL_PATH + ".zip"
    if not os.path.exists(model_file):
        print(f"Модель не найдена: {model_file}")
        print("Сначала запусти: python train.py")
        sys.exit(1)
    return PPO.load(MODEL_PATH)


def format_result(info: dict, terminated: bool, truncated: bool, human: bool = False) -> str:
    """Итог эпизода в читаемую строку (по-русски для человека,
    короткие коды для матчей ботов)."""
    result = info.get("result", "")
    if result == "win":
        return "ПОБЕДА!" if human else "WIN"
    if result == "lose":
        return "ПОРАЖЕНИЕ" if human else "LOSE"
    if result == "timeout_win":
        return "ПОБЕДА ПО HP" if human else "TIMEOUT_WIN"
    if result == "timeout_loss":
        return "ПОРАЖЕНИЕ ПО HP" if human else "TIMEOUT_LOSS"
    if result in ("timeout_draw",) or truncated:
        return "НИЧЬЯ" if human else "DRAW"
    if terminated:
        win = info.get("win", False)
        return ("ПОБЕДА!" if human else "WIN") if win else ("ПОРАЖЕНИЕ" if human else "LOSE")
    return "НИЧЬЯ" if human else "DRAW"


def run(p1: str, p2: str):
    """Главный цикл матча (60 FPS, бесконечные эпизоды до Esc).

    p1 — кто управляет синим:   'human' (клавиатура) | 'ai' (RL-модель);
    p2 — кто управляет красным: 'bot' (скрипт) | 'ai' (RL-модель).

    Красным управляет сама среда (через SelfPlayBot), поэтому для
    p2 == 'ai' достаточно подложить модель в env.update_bot_model().
    """
    need_model = (p1 == "ai") or (p2 == "ai")
    model = load_model() if need_model else None

    env = ShooterEnv(render_mode="human")

    if p2 == "ai" and model is not None:
        env.update_bot_model(model)

    obs, _ = env.reset()
    env.render()

    labels = {
        ("human", "bot"): "Ты (синий)  vs  Скриптовый бот (красный)",
        ("human", "ai"):  "Ты (синий)  vs  RL агент (красный)",
        ("ai",    "bot"): "RL агент (синий)  vs  Скриптовый бот (красный)",
        ("ai",    "ai"):  "RL агент (синий)  vs  RL агент (красный)",
    }
    print(f"\n=== {labels.get((p1, p2), f'{p1} vs {p2}')} ===")
    if p1 == "human":
        print("  ПРОБЕЛ (держать) — двигаться вперёд")
        print("  ENTER            — выстрел")
        print("  ESC              — выход")

    human_p1 = (p1 == "human")
    episode   = 0
    running   = True

    while running:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif human_p1 and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    action = 2  # выстрел

        if human_p1:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] and action == 0:
                action = 1
        else:  # ai
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            episode += 1
            result = format_result(info, terminated, truncated, human=human_p1)
            print(f"Эп {episode}: {result} | "
                  f"синий={info['agent_hits']} попаданий | "
                  f"красный={info['bot_hits']} попаданий | "
                  f"шагов={info['steps']}")
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запуск матча. Первый аргумент = синий, второй = красный.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python play.py human bot   # ты vs скриптовый бот\n"
            "  python play.py human ai    # ты vs RL агент\n"
            "  python play.py ai bot      # RL агент vs скриптовый бот\n"
            "  python play.py ai ai       # RL агент vs RL агент"
        ),
    )
    parser.add_argument(
        "p1", nargs="?", default="ai", choices=PLAYER_CHOICES,
        metavar="P1",
        help="Синий игрок: human | ai (по умолчанию: ai)",
    )
    parser.add_argument(
        "p2", nargs="?", default="bot", choices=PLAYER_CHOICES,
        metavar="P2",
        help="Красный игрок: ai | bot (по умолчанию: bot)",
    )
    args = parser.parse_args()

    if args.p1 == "bot":
        parser.error("p1 не может быть 'bot' — 'bot' доступен только для красного (p2)")

    run(args.p1, args.p2)
