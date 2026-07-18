#!/usr/bin/env python
"""
Обучение универсальной модели: 40 млн шагов в командном режиме
(примерно 18-24 часа на 8-ядерном CPU — поставил на ночь и день, готово).

Запуск:
    python train_overnight.py           — обучить с нуля
    python train_overnight.py --resume  — продолжить сохранённый чекпоинт

После завершения:
    python play.py                      — смотреть результат, плавная ходьба
    python play.py --auto               — автопрограмма команд

Плавность обеспечивают штрафы в среде (action_rate, dof_vel, НЧ-фильтр),
поэтому душить оптимизатор сотнями миллионов шагов больше не нужно.
"""

import sys
import subprocess

if __name__ == "__main__":
    # Переопределяем параметры train.py для максимального обучения
    resume = "--resume" in sys.argv

    # Одна универсальная сеть, командный режим
    cmd = [
        sys.executable, "train.py",
        "--reward-mode", "command",
        "--steps", "40000000",  # 40М шагов: стандартные гиперпараметры учатся быстро
    ]

    if resume:
        cmd.append("--resume")

    print("=" * 80)
    print("ОБУЧЕНИЕ УНИВЕРСАЛЬНОЙ МОДЕЛИ — КОМАНДНЫЙ РЕЖИМ")
    print("=" * 80)
    print(f"Режим: {'resume' if resume else 'fresh start'}")
    print(f"Шагов: 40,000,000 (примерно 18-24 часа на CPU с 8 ядрами)")
    print(f"Сеть: [512, 256], батч 1024, rollout 16384")
    print(f"Оптимизатор: lr 3e-4 -> 3e-5 (linear), clip=0.2, target_kl=0.05")
    print(f"Плавность: штрафы среды action_rate/dof_vel + НЧ-фильтр действий")
    print(f"Результат: models/ppo_centipede_final.zip")
    print()
    print("После обучения робот будет:")
    print("  ✓ Очень плавно ходить (никаких рывков)")
    print("  ✓ Активно балансировать даже на месте")
    print("  ✓ Реагировать на все стрелки: вперёд↑ назад↓ влево← вправо→ стоп")
    print()
    print("РЕКОМЕНДАЦИЯ: запустите в экране (screen / tmux) или в фоне (&)")
    print("=" * 80)
    print()

    sys.exit(subprocess.call(cmd))
