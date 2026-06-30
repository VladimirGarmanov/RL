# RL Projects

Два проекта с Reinforcement Learning на Python — от нуля до self-play.

## Проекты

### 🚗 [car_racing/](car_racing/) — F1-трасса
Машинка учится ехать по Spa-inspired трассе. PPO, 7 лучей-сенсоров, скользящая камера.

```bash
cd car_racing
python train.py       # обучение
python play_trained.py  # смотреть результат
```

### 🔫 [shooter/](shooter/) — Self-Play шутер
Два агента воюют друг против друга. Каждые 200k шагов бот получает копию весов агента и становится умнее.

```bash
cd shooter
python train.py       # обучение (self-play)
python play.py        # смотреть модель
python play.py --human  # сыграть самому
```

## Установка

```bash
git clone <repo>
cd RL
pip install -r requirements.txt
```

## Стек

- **Python 3.10+**
- **pygame 2.x** — рендеринг
- **gymnasium** — RL-среда (OpenAI Gym API)
- **stable-baselines3** — алгоритм PPO
- **PyTorch** + **numpy**
