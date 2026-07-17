"""
train.py — Обучение многоножки ходьбе алгоритмом PPO (Stable-Baselines3).

Запуск:
    python train.py             — простая модель: идти вперёд
    python train.py --reward-mode command — универсал по командам скорости (v/w)
    python train.py forward     — сеть-специалист "прямо"
    python train.py backward    — сеть-специалист "назад"
    python train.py left        — сеть-специалист "поворот налево" (на месте и дугой)
    python train.py right       — сеть-специалист "поворот направо"
    python train.py --steps 3000000 left   — переопределить длину обучения

Специалисты ВСЕГДА учатся в командном режиме: каждый видит команды только
своего навыка (+ команду "стоп") и обязан их отрабатывать — в play.py их
переключают стрелочки, и все четыре сети должны понимать наблюдения одинаково.
Режим forward для специалиста не имеет смысла (там награда игнорирует команду
и все четыре сети выучили бы одно и то же: идти вперёд).

Специалисты обучаются ТЁПЛЫМ СТАРТОМ от универсальной командной модели
(models/ppo_centipede_final.zip), если её meta.json имеет ту же reward_version
и reward_mode=command: она уже умеет ходить, специалисту остаётся отточить
свой навык — поэтому шагов нужно в разы меньше. Без совместимого универсала
специалист учится с нуля — это дольше (SKILL_SCRATCH_TIMESTEPS).
Старые несовместимые модели не подхватываются.
Каждый специалист живёт в своей папке models/skills/<навык>/ со своей
статистикой нормализации. Их по очереди запускает play.py по стрелочкам.

Обучение идёт в N_ENVS параллельных симуляциях. Наблюдения и награды
нормализуются (VecNormalize) — статистика нормализации сохраняется рядом
с моделью и обязательна для воспроизведения в play.py.

Результаты (для универсала; специалисты — то же внутри models/skills/<навык>/):
    models/ppo_centipede_final.zip   — финальная модель
    models/vecnormalize_final.pkl    — статистика нормализации
    models/meta.json                 — параметры (число сегментов и т.п.)
    models/checkpoints/              — промежуточные модели каждые CHECKPOINT_FREQ шагов

Параметры обучения можно менять прямо в этом файле:
    TOTAL_TIMESTEPS         — простая ходьба вперёд (reward-mode forward)
    COMMAND_TIMESTEPS       — универсал по командам скорости
    SKILL_TIMESTEPS         — специалист с тёплым стартом от универсала
    SKILL_SCRATCH_TIMESTEPS — специалист с нуля (совместимого универсала нет)
    N_SEGMENTS      — размер многоножки (менять вместе с переобучением с нуля)
    RESUME_TRAINING — дообучать совместимую сохранённую модель вместо старта с нуля
"""

import argparse
import json
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from centipede_env import make_centipede_env, REWARD_VERSION

# --- Параметры обучения ---
N_SEGMENTS = 6               # сегментов тела (ног = в 2 раза больше)
TOTAL_TIMESTEPS = 3_000_000   # простая ходьба вперёд обучается заметно быстрее
COMMAND_TIMESTEPS = 12_000_000       # универсал по командам: задача сильно сложнее
SKILL_TIMESTEPS = 4_000_000          # специалист с тёплым стартом от универсала
SKILL_SCRATCH_TIMESTEPS = 8_000_000  # специалист с нуля (универсала нет)
N_ENVS = 8                   # параллельных симуляций
MAX_EPISODE_STEPS = 1000     # 50 секунд симуляции на эпизод
CHECKPOINT_FREQ = 100_000    # суммарных шагов между чекпоинтами
SEED = 42
REWARD_MODE = "forward"      # default: простая ходьба вперёд без поворотов/стопа
RESUME_TRAINING = False      # по умолчанию старт с нуля: старые v1-модели падали

SKILLS = ("forward", "backward", "left", "right")

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")
GENERALIST_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_centipede_final")
GENERALIST_VECNORM_PATH = os.path.join(MODELS_DIR, "vecnormalize_final.pkl")

# Гиперпараметры PPO (стандартные для локомоции).
# PPO (Proximal Policy Optimization) — рабочая лошадка современного RL:
# как REINFORCE, повышает вероятность удачных действий, но обновляет
# политику осторожно (clip_range не даёт шагу увести её слишком далеко)
# и использует критика (оценщика ценности состояний) для снижения шума.
PPO_PARAMS = {
    "n_steps": 2048,          # шагов на среду за rollout (8 сред -> 16384 на обновление)
    "batch_size": 1024,       # крупные батчи => менее шумные, более плавные обновления
    "n_epochs": 5,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,        # останавливает слишком резкие PPO-обновления
    "use_sde": True,          # более гладкая exploration для непрерывных сервокоманд
    "sde_sample_freq": 4,
    "verbose": 1,
    "device": "cpu",          # MLP-политика на CPU быстрее, чем на MPS/GPU
    "policy_kwargs": {
        "net_arch": [512, 256],
        "log_std_init": -2.0,  # меньше стартовых конвульсий на весь ход серво
    },
}

# При догрузке сохранённой модели (resume / тёплый старт) PPO.load восстанавливает
# и её СТАРЫЕ гиперпараметры. Перекрываем те, что определяют стабильность обучения:
# модель, сохранённую со старым агрессивным lr/clip, иначе снова разнесёт.
LOAD_OVERRIDES = {
    key: PPO_PARAMS[key]
    for key in ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
                "gae_lambda", "clip_range", "ent_coef", "vf_coef",
                "max_grad_norm", "target_kl")
}


def skill_paths(skill: str):
    """Пути артефактов обучения для заданного навыка.

    "all" — универсальная модель в models/ (как раньше);
    специалисты — каждый в своей папке models/skills/<навык>/.
    Возвращает (models_dir, checkpoints_dir, final_model_path, vecnorm_path, meta_path).
    """
    base = MODELS_DIR if skill == "all" else os.path.join(MODELS_DIR, "skills", skill)
    return (
        base,
        os.path.join(base, "checkpoints"),
        os.path.join(base, "ppo_centipede_final"),
        os.path.join(base, "vecnormalize_final.pkl"),
        os.path.join(base, "meta.json"),
    )


def checkpoint_steps(path):
    """Достаёт число шагов из имени чекпоинта.

    Например 'ppo_centipede_500000_steps.zip' -> 500000. Нужно, чтобы
    среди чекпоинтов найти самый свежий. Если имя не подходит под
    шаблон — возвращает -1 (такой файл проиграет любому нормальному).
    """
    m = re.search(r"_(\d+)_steps\.zip$", path)
    return int(m.group(1)) if m else -1


def find_resume_paths(final_model_path, vecnorm_path, checkpoints_dir):
    """Возвращает пару (модель, VecNormalize) для продолжения обучения.

    Приоритет: финальная модель, если её нет — самый свежий чекпоинт.
    VecNormalize (.pkl) — статистика нормализации наблюдений; модель
    обучалась на нормализованных данных, поэтому продолжать без неё нельзя.
    """
    final_zip = final_model_path + ".zip"
    if os.path.exists(final_zip):
        return final_model_path, vecnorm_path if os.path.exists(vecnorm_path) else None

    checkpoints = glob.glob(os.path.join(checkpoints_dir, "ppo_centipede_*_steps.zip"))
    if not checkpoints:
        return None, None

    model_path = max(checkpoints, key=checkpoint_steps)
    steps = checkpoint_steps(model_path)
    ckpt_vecnorm = os.path.join(
        checkpoints_dir, f"ppo_centipede_vecnormalize_{steps}_steps.pkl"
    )
    return model_path, ckpt_vecnorm if os.path.exists(ckpt_vecnorm) else None


def saved_meta(meta_path):
    """Метаданные сохранённого набора артефактов; пустой dict для старых моделей."""
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def is_compatible_meta(meta_path, reward_mode):
    """Можно ли продолжать эту модель с текущей средой."""
    meta = saved_meta(meta_path)
    return (
        meta.get("reward_version") == REWARD_VERSION
        and meta.get("reward_mode") == reward_mode
    )


def main():
    """Полный цикл обучения: создать среды -> (загрузить/создать модель) ->
    учить нужное число шагов -> сохранить модель + нормализацию + мету."""
    parser = argparse.ArgumentParser(description="Обучение многоножки (PPO)")
    parser.add_argument("skill", nargs="?", default="all",
                        choices=("all",) + SKILLS,
                        help="куда сохранять навык: all = основная модель в models/")
    parser.add_argument("--steps", type=int, default=None,
                        help="переопределить число шагов обучения")
    parser.add_argument("--reward-mode", choices=("forward", "command"), default=REWARD_MODE,
                        help="forward = только идти вперёд; command = команды скорости/поворота")
    parser.add_argument("--resume", action="store_true", default=RESUME_TRAINING,
                        help="продолжить только совместимую модель/checkpoint")
    parser.add_argument("--fresh", action="store_false", dest="resume",
                        help="стартовать с нуля, игнорируя сохранённые модели")
    args = parser.parse_args()

    skill = args.skill
    if skill != "all" and args.reward_mode != "command":
        # Специалист по определению отрабатывает команды своего навыка;
        # в режиме forward награда команду игнорирует, и все 4 сети
        # выучили бы одно и то же — просто идти вперёд.
        print(f"Специалист [{skill}] обучается только в командном режиме — включаю --reward-mode command")
        args.reward_mode = "command"
    models_dir, checkpoints_dir, final_model_path, vecnorm_path, meta_path = skill_paths(skill)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # N_ENVS параллельных копий среды в отдельных процессах (SubprocVecEnv):
    # PPO собирает опыт со всех сразу — в N_ENVS раз быстрее по времени.
    env = make_vec_env(
        make_centipede_env,
        n_envs=N_ENVS,
        seed=SEED,
        env_kwargs={
            "n_segments": N_SEGMENTS,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "command_profile": skill if args.reward_mode == "command" else "forward",
            "reward_mode": args.reward_mode,
            "auto_command_resample": args.reward_mode == "command",
            # Лёгкие бугры дают устойчивость к толчкам, но мешают плавности;
            # 5 мм — компромисс (play.py показывает робота на ровном полу)
            "terrain_roughness": 0.0 if args.reward_mode == "forward" else 0.005,
        },
        vec_env_cls=SubprocVecEnv,
    )

    # VecNormalize приводит наблюдения и награды к единому масштабу
    # (скользящие среднее и дисперсия) — без этого нейросети тяжело:
    # углы ~0.5 рад и скорости ~5 рад/с живут в разных масштабах.
    saved = saved_meta(meta_path)
    can_resume = args.resume and is_compatible_meta(meta_path, args.reward_mode)
    if args.resume and not can_resume:
        print(
            f"Сохранённая модель [{skill}] несовместима "
            f"(нужно reward_mode={args.reward_mode}, reward_version={REWARD_VERSION}; "
            f"в meta.json: mode={saved.get('reward_mode')}, version={saved.get('reward_version')}); "
            "стартуем с нуля."
        )
    resume_model_path, resume_vecnorm_path = (
        find_resume_paths(final_model_path, vecnorm_path, checkpoints_dir)
        if can_resume else (None, None)
    )
    resume = resume_model_path is not None

    # Тёплый старт специалиста: своей модели ещё нет, но есть универсал —
    # он уже умеет ходить, специалисту остаётся отточить один навык.
    warm_start = (
        not resume
        and skill != "all"
        and is_compatible_meta(os.path.join(MODELS_DIR, "meta.json"), args.reward_mode)
        and os.path.exists(GENERALIST_MODEL_PATH + ".zip")
    )

    # Длина обучения зависит от задачи: простая ходьба вперёд — быстро,
    # командный универсал — долго, специалист — смотря есть ли тёплый старт.
    if args.steps:
        total_timesteps = args.steps
    elif skill == "all":
        total_timesteps = TOTAL_TIMESTEPS if args.reward_mode == "forward" else COMMAND_TIMESTEPS
    elif resume or warm_start:
        total_timesteps = SKILL_TIMESTEPS
    else:
        total_timesteps = SKILL_SCRATCH_TIMESTEPS

    if resume and resume_vecnorm_path:
        env = VecNormalize.load(resume_vecnorm_path, env)
        env.training = True   # продолжаем обновлять статистику нормализации
    elif warm_start and os.path.exists(GENERALIST_VECNORM_PATH):
        env = VecNormalize.load(GENERALIST_VECNORM_PATH, env)
        env.training = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=PPO_PARAMS["gamma"])

    if resume:
        print(f"Продолжаем обучение [{skill}]: {resume_model_path}")
        print(f"Нормализация: {resume_vecnorm_path or 'создана заново'}")
        model = PPO.load(resume_model_path, env=env, device="cpu",
                         custom_objects=LOAD_OVERRIDES)
    elif warm_start:
        print(f"Тёплый старт [{skill}] от универсала: {GENERALIST_MODEL_PATH}.zip")
        model = PPO.load(GENERALIST_MODEL_PATH, env=env, device="cpu",
                         custom_objects=LOAD_OVERRIDES)
    else:
        if skill != "all":
            print(f"Совместимого командного универсала нет — специалист [{skill}] учится с нуля, "
                  f"{total_timesteps:,} шагов (быстрее: сначала обучить универсала "
                  f"`python train.py --reward-mode command`, тогда специалистам хватит {SKILL_TIMESTEPS:,})")
        model = PPO("MlpPolicy", env, seed=SEED, **PPO_PARAMS)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=checkpoints_dir,
        name_prefix="ppo_centipede",
        save_vecnormalize=True,
    )

    print(f"Многоножка: {N_SEGMENTS} сегментов, {N_SEGMENTS * 2} ног, навык: {skill}")
    print(f"Режим награды: {args.reward_mode}")
    print(f"Наблюдения: {env.observation_space.shape}, действия: {env.action_space.shape}")
    print(f"Обучение: {total_timesteps:,} шагов в {N_ENVS} параллельных средах\n")

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=not resume,
    )
    elapsed = time.time() - start

    model.save(final_model_path)
    env.save(vecnorm_path)
    with open(meta_path, "w") as f:
        json.dump({
            "n_segments": N_SEGMENTS,
            "skill": skill,
            "reward_mode": args.reward_mode,
            "reward_version": REWARD_VERSION,
            "total_timesteps": total_timesteps,
            "resumed_from": resume_model_path or (GENERALIST_MODEL_PATH if warm_start else None),
        }, f)

    print(f"\nГотово за {elapsed / 60:.1f} мин")
    print(f"Модель:       {final_model_path}.zip")
    print(f"Нормализация: {vecnorm_path}")
    print("Смотреть результат: python play.py")


if __name__ == "__main__":
    main()
