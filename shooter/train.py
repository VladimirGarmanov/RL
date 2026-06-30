"""
train.py — Self-play обучение агента.

Как это работает:
  1. Агент (синий) учится с PPO.
  2. Каждые SELFPLAY_UPDATE_FREQ шагов, бот (красный) получает копию
     текущих весов агента → начинает играть умнее.
  3. Агент снова адаптируется к более умному боту. Цикл.

  Это называется self-play — именно так обучали AlphaGo, OpenAI Five, AlphaStar.

Запуск с нуля:      python train.py
Продолжить обучение: поставить RESUME_TRAINING = True
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from shooter_env import ShooterEnv

_HERE           = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(_HERE, "models", "ppo_shooter_v2")
TOTAL_TIMESTEPS      = 10_000_000  # нужно больше для self-play
RESUME_TRAINING      = True        # продолжаем v2
SELFPLAY_UPDATE_FREQ = 200_000     # каждые 200k шагов обновляем бота
EXPLORATION_ENT_COEF = 0.05        # больше исследования: чаще пробует стрелять/двигаться


PPO_PARAMS = {
    "n_steps":       2048,
    "batch_size":    64,
    "n_epochs":      10,
    "learning_rate": 2e-4,          # чуть меньше — уже есть база
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "ent_coef":      EXPLORATION_ENT_COEF,
    "clip_range":    0.2,
    "verbose":       1,
    "policy_kwargs": {
        "net_arch": [256, 256, 128],
    },
}


class SelfPlayCallback(BaseCallback):
    """
    Каждые SELFPLAY_UPDATE_FREQ шагов копируем веса агента в бота.
    Бот становится всё умнее → агент вынужден адаптироваться.
    """

    def __init__(self, update_freq: int = SELFPLAY_UPDATE_FREQ):
        super().__init__()
        self.update_freq  = update_freq
        self._last_update = 0
        self._generation  = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_freq:
            self.training_env.env_method("update_bot_model", self.model)
            self._last_update = self.num_timesteps
            self._generation += 1
            print(f"\n[Self-Play] Поколение {self._generation} — бот обновлён "
                  f"на шаге {self.num_timesteps}")
        return True


class ProgressCallback(BaseCallback):
    PRINT_FREQ = 20_000

    def __init__(self):
        super().__init__()
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.PRINT_FREQ:
            self._last_print = self.num_timesteps
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                mean_r  = sum(rewards) / len(rewards)
                print(f"[{self.num_timesteps:>9}] mean_reward={mean_r:+.1f}")
        return True


def main():
    os.makedirs(os.path.join(_HERE, "models"), exist_ok=True)

    env = ShooterEnv(render_mode=None)
    check_env(env, warn=True)

    model_file = MODEL_SAVE_PATH + ".zip"
    if RESUME_TRAINING and os.path.exists(model_file):
        print(f"Загружаем модель: {model_file}")
        model = PPO.load(MODEL_SAVE_PATH, env=env)
        model.ent_coef = EXPLORATION_ENT_COEF
        # Сразу даём боту начальный уровень из загруженной модели
        env.update_bot_model(model)
        print("Бот инициализирован копией загруженной модели.")
    else:
        print("Обучение с нуля.")
        model = PPO("MlpPolicy", env, **PPO_PARAMS)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(_HERE, "models"),
        name_prefix="selfplay_ckpt",
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[ProgressCallback(), SelfPlayCallback(), checkpoint_cb],
            progress_bar=True,
            reset_num_timesteps=not RESUME_TRAINING,
        )
    except KeyboardInterrupt:
        print("\nПрервано.")

    model.save(MODEL_SAVE_PATH)
    print(f"Модель сохранена: {MODEL_SAVE_PATH}.zip")
    env.close()


if __name__ == "__main__":
    main()
