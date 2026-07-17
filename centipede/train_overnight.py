#!/usr/bin/env python
"""
Обучение универсальной модели на ночь: одна сеть в командном режиме,
18 млн шагов (обычно 15–20 часов на CPU с 8 параллельными средами).

Запуск:
    python train_overnight.py           — обучить с нуля
    python train_overnight.py --resume  — продолжить сохранённый чекпоинт

После завершения:
    python play.py                      — смотреть результат, стрелки на полный контроль
    python play.py --auto               — автопрограмма команд
"""

import sys
import subprocess

if __name__ == "__main__":
    # Переопределяем параметры train.py для ночного прогона
    resume = "--resume" in sys.argv

    # Одна универсальная сеть, командный режим (не 4 специалиста)
    cmd = [
        sys.executable, "train.py",
        "--reward-mode", "command",
        "--steps", "18000000",  # 12M базовых, +50% на полировку = 18M
    ]

    if resume:
        cmd.append("--resume")

    print("=" * 70)
    print("ОБУЧЕНИЕ УНИВЕРСАЛЬНОЙ МОДЕЛИ НА НОЧЬ")
    print("=" * 70)
    print(f"Режим: {' resume' if resume else 'fresh start'}")
    print(f"Шагов: 18,000,000 (примерно 15-20 часов)")
    print(f"Сеть: [512, 256], батч 1024, rollout 16384")
    print(f"Результат: models/ppo_centipede_final.zip")
    print("=" * 70)
    print()

    sys.exit(subprocess.call(cmd))
