"""
train.py — Обучение агента PPO алгоритмом из Stable-Baselines3.

Запуск: python train.py

После обучения модель сохраняется в models/ppo_car_racing.zip

Параметры обучения можно менять прямо в этом файле:
    TOTAL_TIMESTEPS — сколько шагов обучать (200k — базовое, 1M — хорошее)
    N_STEPS         — длина rollout-буфера
    BATCH_SIZE      — размер мини-батча
    LEARNING_RATE   — скорость обучения
    ENT_COEF        — коэффициент энтропии (больше = больше исследования)
    RESUME_TRAINING — дообучать сохранённую модель вместо старта с нуля
"""

import os
import sys

# Добавляем текущую директорию в путь, чтобы импорты работали из PyCharm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from car_env import CarRacingEnv

# --- Параметры обучения ---
TOTAL_TIMESTEPS = 2_000_000
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(_HERE, "models", "ppo_car_racing")
CHECKPOINT_FREQ = 25_000    # сохраняем промежуточные модели каждые N шагов
RESUME_TRAINING = False      # True = дообучать MODEL_SAVE_PATH.zip, если файл существует

# Гиперпараметры PPO (можно менять для улучшения обучения)
PPO_PARAMS = {
    "n_steps": 4096,        # шагов за один rollout (больше = стабильнее градиент)
    "batch_size": 128,      # размер мини-батча
    "n_epochs": 10,         # эпох обновления за один rollout
    "learning_rate": 2e-4,  # чуть ниже — стабильнее с большой сетью
    "gamma": 0.99,          # дисконт-фактор
    "gae_lambda": 0.95,     # lambda для GAE
    "clip_range": 0.2,      # клиппинг PPO
    "ent_coef": 0.05,       # энтропийная регуляризация (поощряет исследование)
    "vf_coef": 0.5,         # коэффициент value function loss
    "max_grad_norm": 0.5,   # клиппинг градиентов
    "verbose": 1,
    # Архитектура нейросети: два слоя вместо дефолтных [64,64]
    "policy_kwargs": {
        "net_arch": [256, 128],   # Actor и Critic оба используют эту структуру
    },
}


class ProgressCallback(BaseCallback):
    """
    Callback для вывода статистики обучения.
    Выводит лучший средний reward каждые PRINT_FREQ шагов.
    """

    PRINT_FREQ = 10_000

    def __init__(self):
        super().__init__()
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.PRINT_FREQ:
            self._last_print = self.num_timesteps

            # Достаём статистику из логов
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                mean_r = sum(rewards) / len(rewards)
                mean_l = sum(lengths) / len(lengths)
                print(f"[{self.num_timesteps:>8} steps] "
                      f"mean_reward={mean_r:+.2f}  mean_ep_len={mean_l:.0f}")
            else:
                print(f"[{self.num_timesteps:>8} steps] collecting data...")

        return True  # продолжаем обучение


def main():
    print("=" * 60)
    print("RL Car Racing — PPO Training")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}.zip")
    print(f"Resume existing model: {RESUME_TRAINING}")
    print("=" * 60)

    # Создаём директорию для моделей
    os.makedirs(os.path.join(_HERE, "models"), exist_ok=True)

    # Создаём среду БЕЗ визуализации (render_mode=None — быстрое обучение)
    print("\n[1/4] Creating environment...")
    env = CarRacingEnv(render_mode=None)

    # Проверяем, что среда соответствует Gymnasium API
    print("[2/4] Checking environment with SB3 check_env...")
    check_env(env, warn=True)
    print("      Environment OK!")

    # Создаём новую модель или дообучаем сохранённую.
    print("[3/4] Creating PPO model...")
    model_file = f"{MODEL_SAVE_PATH}.zip"
    if RESUME_TRAINING and os.path.exists(model_file):
        print(f"      Loading existing model: {model_file}")
        model = PPO.load(MODEL_SAVE_PATH, env=env)
        reset_num_timesteps = False
    else:
        print("      Starting from scratch.")
        model = PPO("MlpPolicy", env, **PPO_PARAMS)
        reset_num_timesteps = True
    print(f"      Policy network: {model.policy}")

    # Callbacks: прогресс + промежуточные сохранения
    progress_cb = ProgressCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(_HERE, "models"),
        name_prefix="ppo_checkpoint",
        verbose=1,
    )

    # Запускаем обучение
    print(f"\n[4/4] Training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("      (нажми Ctrl+C для остановки, модель сохранится автоматически)\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[progress_cb, checkpoint_cb],
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем.")

    # Сохраняем финальную модель
    model.save(MODEL_SAVE_PATH)
    print(f"\nМодель сохранена: {MODEL_SAVE_PATH}.zip")
    print("Запусти play_trained.py, чтобы посмотреть результат!")

    env.close()


if __name__ == "__main__":
    main()
