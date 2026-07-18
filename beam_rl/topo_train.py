"""Обучение REINFORCE для топологической оптимизации балки.

КАК ЭТО РАБОТАЕТ (кратко для тех, кто не знаком с RL)
-----------------------------------------------------
В отличие от одношаговых задач (train.py, prop_rl), здесь настоящий
многошаговый эпизод: агент 25 шагов подряд «вырезает» материал из балки
(среда CarveEnv из topo_env.py), и только В КОНЦЕ узнаёт, хороша ли
получившаяся структура. Это классическая задача с отложенной наградой.

Политика — маленькая нейросеть (MLP): по состоянию (доля оставшегося
материала, номер шага) выдаёт средние гауссова распределения над двумя
действиями: (доля удаления на этом шаге, температура резьбы).

REINFORCE для эпизодов: вероятность всей траектории = произведение
вероятностей шагов, поэтому лог-вероятности шагов суммируются, и весь
эпизод усиливается/ослабляется целиком по его финальной награде
(advantage = награда эпизода минус средняя по батчу эпизодов).

Лучшая найденная структура сохраняется в best_topology.npy — из неё
topo_stl.py построит печатаемую 3D-модель.

Запуск: python3 topo_train.py
"""

import pathlib

import numpy as np
import torch
import torch.nn as nn

from topo_env import CarveEnv, solve, TARGET_VOL

# ---- Гиперпараметры -----------------------------------------------------------
UPDATES = 40     # шагов обучения (каждый = батч эпизодов + градиентный шаг)
BATCH_EP = 6     # эпизодов в батче (каждый эпизод = 25 МКЭ-расчётов, дорого!)
LR = 5e-3        # learning rate
SEED = 0         # фиксируем случайность для воспроизводимости

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


class Policy(nn.Module):
    """Политика: нейросеть 2 -> 32 -> 2 (вход: наблюдение, выход: средние действий).

    log_std — обучаемый разброс действий, общий для всех состояний
    (nn.Parameter = тензор, который оптимизатор тоже будет обновлять).
    Старт -1.5 даёт std ~0.22 — умеренное исследование.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 2))
        self.log_std = nn.Parameter(torch.full((2,), -1.5))

    def dist(self, obs):
        """Строит гауссово распределение действий для данного наблюдения.
        clamp на std: не даём разбросу ни схлопнуться, ни взорваться."""
        mean = self.net(torch.as_tensor(obs, dtype=torch.float32))
        return torch.distributions.Normal(mean, self.log_std.exp().clamp(0.03, 0.7))


def to_action(raw):
    """Переводит «сырой» выход политики в физические параметры шага резьбы.

    sigmoid сжимает произвольные числа в (0, 1), затем линейно
    масштабируем: доля удаления 1..7% клеток за шаг, температура 0.03..0.6.

    torch.no_grad(): здесь градиенты не нужны — REINFORCE обучается через
    лог-вероятности сырых действий, а не через само преобразование.
    """
    with torch.no_grad():
        s = torch.sigmoid(raw)
    frac = 0.01 + 0.06 * float(s[0])
    temp = 0.03 + 0.57 * float(s[1])
    return frac, temp


def run_episode(policy, env, sample=True):
    """Прогоняет один полный эпизод резьбы (25 шагов).

    sample=True  — действия сэмплируются случайно (режим обучения);
    sample=False — берётся среднее распределения (детерминированный
                   режим, чтобы оценить обученную политику без шума).

    Возвращает: (сумма лог-вероятностей всех шагов — нужна REINFORCE,
    финальная награда, info последнего шага).
    """
    obs = env.reset()
    logps, done = [], False
    while not done:
        d = policy.dist(obs)
        raw = d.sample() if sample else d.mean.detach()
        # log_prob суммируем по 2 компонентам действия; список logps
        # соберёт все шаги эпизода.
        logps.append(d.log_prob(raw).sum())
        obs, r, done, info = env.step(*to_action(raw))
    # Сумма по шагам = лог-вероятность всей траектории.
    return torch.stack(logps).sum(), r, info


policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=LR)
env = CarveEnv(rng)
best = {"reward": -np.inf}   # лучшая структура за всё обучение

for up in range(UPDATES):
    # Собираем батч эпизодов.
    logps, rewards = [], []
    for _ in range(BATCH_EP):
        lp, r, info = run_episode(policy, env)
        logps.append(lp)
        rewards.append(r)
        # Запоминаем лучшую структуру (копию карты материала!).
        if r > best["reward"]:
            best = {"reward": r, "density": env.density.copy(),
                    "compliance": info["compliance"], "vol": info["vol"]}

    # Шаг REINFORCE по батчу эпизодов: advantage = награда эпизода минус
    # средняя по батчу; хорошие эпизоды становятся вероятнее целиком.
    r_t = torch.as_tensor(rewards, dtype=torch.float32)
    adv = r_t - r_t.mean()
    loss = -(adv * torch.stack(logps)).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"upd {up:2d} | reward {r_t.mean():7.2f} +- {r_t.std():5.2f} | "
          f"best {best['reward']:6.2f} (C/C0 {best['compliance']/env.c0:.2f}, "
          f"vol {best['vol']:.2f})")

# Финальный детерминированный прогон обученной политики (без случайности) —
# вдруг она стабильно выдаёт структуру лучше случайно пойманной.
_, r, info = run_episode(policy, env, sample=False)
print(f"детерминированная политика: reward {r:.2f}, "
      f"C/C0 {info['compliance']/env.c0:.2f}, vol {info['vol']:.2f}")
if r > best["reward"]:
    best = {"reward": r, "density": env.density.copy(),
            "compliance": info["compliance"], "vol": info["vol"]}

# Сохраняем карту материала лучшей структуры (numpy-массив 60x24 из 0 и 1).
here = pathlib.Path(__file__).parent
np.save(here / "best_topology.npy", best["density"])

# Итоговая сводка: сравниваем жёсткость с исходной сплошной балкой.
c0, _ = solve(np.ones_like(best["density"]))
print(f"\nИтог: материал {best['vol']*100:.1f}% (цель {TARGET_VOL*100:.0f}%), "
      f"жёсткость {c0/best['compliance']*100:.0f}% от сплошной балки")
print(f"Сохранено: {here / 'best_topology.npy'}")

# ASCII-превью половины балки прямо в консоль ('#' = материал, '.' = пустота;
# полная балка получается зеркалированием в середине пролёта).
d = best["density"]
print("\nСтруктура (слева середина пролёта, справа опора; верх балки сверху):")
for iy in range(d.shape[1] - 1, -1, -1):
    print("".join("#" if d[ix, iy] else "." for ix in range(d.shape[0])))
