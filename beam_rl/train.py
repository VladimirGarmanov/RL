"""Обучение REINFORCE с гауссовой политикой: подбор сечения двутавровой балки.

КАК ЭТО РАБОТАЕТ (кратко для тех, кто не знаком с RL)
-----------------------------------------------------
Политика агента — нормальное распределение N(mu, sigma) над 4 «сырыми»
действиями; среда (beam_env.py) сжимает их tanh'ом в физические размеры
сечения. Обучаемые параметры — mu (средние) и log_std (логарифмы разбросов).

Алгоритм REINFORCE на каждой итерации:
  1. Сэмплируем батч из 256 случайных действий из текущего распределения.
  2. Считаем награду каждого (через сопромат в beam_env.evaluate).
  3. Сдвигаем распределение так, чтобы действия с наградой выше средней
     по батчу стали вероятнее, а хуже средней — менее вероятны.
Распределение постепенно «стягивается» к лёгким и прочным балкам.

Награда: -масса + большие штрафы за нарушение прочности/прогиба.
Лучшее допустимое (feasible = все ограничения выполнены) решение
сохраняется в best_beam.json — из него make_stl.py построит 3D-модель.

Запуск: python3 train.py
"""

import json
import pathlib

import numpy as np
import torch

from beam_env import (action_to_params, evaluate, analyze,
                      SIGMA_ALLOW_MPA, DEFLECTION_LIMIT_MM,
                      SPAN_MM, DESIGN_LOAD_N, SIGMA_ULT_MPA)

# ---- Гиперпараметры обучения ------------------------------------------------
ITERS = 1500    # итераций обучения (шагов градиентного спуска)
BATCH = 256     # дизайнов пробуем на каждой итерации
LR = 0.03       # learning rate — размер шага оптимизатора
SEED = 0        # фиксируем случайность для воспроизводимости

torch.manual_seed(SEED)
np.random.seed(SEED)

# Параметры политики: mu — средние 4 действий (старт: нули = середины
# диапазонов после tanh); log_std — логарифмы разбросов (log, чтобы
# std всегда оставался положительным). requires_grad=True включает
# автоматическое дифференцирование PyTorch по этим тензорам.
mu = torch.zeros(4, requires_grad=True)
log_std = torch.zeros(4, requires_grad=True)
opt = torch.optim.Adam([mu, log_std], lr=LR)

best = {"mass_g": float("inf")}   # лучшее допустимое решение за всё обучение

for it in range(ITERS):
    # std ограничиваем: не меньше 0.02 (иначе поиск замирает слишком рано)
    # и не больше 2.0 (иначе поиск — чистый шум).
    std = log_std.exp().clamp(0.02, 2.0)

    # Сэмплируем батч действий и считаем их лог-вероятности.
    # log_prob суммируем по 4 компонентам: вероятность вектора =
    # произведение вероятностей компонент = сумма логарифмов.
    dist = torch.distributions.Normal(mu, std)
    actions = dist.sample((BATCH,))
    logp = dist.log_prob(actions).sum(dim=1)

    # Прогоняем через физику (numpy — градиенты через физику не нужны,
    # REINFORCE обучается только через вероятности действий).
    params = action_to_params(actions.numpy())
    rewards, feasible, sigma, defl, mass = evaluate(params)

    # Запоминаем самое лёгкое из допустимых решений батча.
    if feasible.any():
        # Трюк: недопустимым решениям подставляем массу inf, чтобы argmin
        # выбрал самое лёгкое ИЗ допустимых.
        i = np.where(feasible, mass, np.inf).argmin()
        if mass[i] < best["mass_g"]:
            best = {
                "H_mm": round(float(params[i, 0]), 2),
                "B_mm": round(float(params[i, 1]), 2),
                "tf_mm": round(float(params[i, 2]), 2),
                "tw_mm": round(float(params[i, 3]), 2),
                "mass_g": round(float(mass[i]), 2),
                "sigma_mpa": round(float(sigma[i]), 2),
                "deflection_mm": round(float(defl[i]), 3),
            }

    # ---- Шаг REINFORCE ----
    r = torch.as_tensor(rewards, dtype=torch.float32)

    # Advantage: награда минус средняя по батчу (baseline). Важно не
    # абсолютное значение награды, а лучше ли дизайн остальных в батче —
    # это сильно снижает шум градиента.
    advantage = r - r.mean()

    # Потеря REINFORCE: минимизация -(logp * advantage) повышает
    # вероятность действий с advantage > 0. Второй член — бонус за
    # энтропию: мешает распределению схлопнуться преждевременно.
    loss = -(logp * advantage).mean() - 1e-3 * dist.entropy().sum()

    # Стандартная тройка PyTorch: обнулить градиенты, посчитать, шагнуть.
    opt.zero_grad()
    loss.backward()
    opt.step()

    if it % 150 == 0 or it == ITERS - 1:
        print(f"iter {it:4d} | reward {rewards.mean():7.3f} | "
              f"feasible {feasible.mean()*100:5.1f}% | "
              f"best mass {best.get('mass_g', float('nan'))} g")

# После обучения смотрим, что выдаёт детерминированная политика —
# просто среднее mu без случайности (куда «сошлось» распределение).
final_params = action_to_params(mu.detach().numpy()[None, :])[0]
f_sigma, f_defl, f_mass = analyze(final_params[None, :])
print("\nПолитика (детерм.): H=%.2f B=%.2f tf=%.2f tw=%.2f | "
      "масса %.1f г, sigma %.2f МПа, прогиб %.3f мм"
      % (*final_params, f_mass[0], f_sigma[0], f_defl[0]))

# Если ни одно решение не выполнило ограничения — что-то сломано.
assert "H_mm" in best, "не найдено ни одного допустимого решения"

# Дописываем контекст задачи, чтобы JSON был самодостаточным.
best["safety_factor"] = round(SIGMA_ULT_MPA / best["sigma_mpa"], 1)  # фактический запас
best["span_mm"] = SPAN_MM
best["design_load_n"] = DESIGN_LOAD_N
best["sigma_allow_mpa"] = round(SIGMA_ALLOW_MPA, 2)
best["deflection_limit_mm"] = DEFLECTION_LIMIT_MM

out = pathlib.Path(__file__).parent / "best_beam.json"
out.write_text(json.dumps(best, indent=2))
print(f"\nЛучшая балка: {best}")
print(f"Сохранено в {out}")
