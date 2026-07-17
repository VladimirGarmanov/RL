# Docker: Centipede

Образ содержит Python, системные библиотеки для MuJoCo, Stable-Baselines3,
Torch, Gymnasium, pygame и код проекта. Обученные модели не запекаются в образ:
папку `centipede/models` нужно монтировать как volume.

## Сборка

Из корня репозитория:

```bash
docker build -t centipede-rl .
```

## Быстрая Проверка

Сгенерировать MuJoCo XML и проверить импорты:

```bash
docker run --rm centipede-rl python centipede_model.py
```

Проверить создание среды без окна:

```bash
docker run --rm centipede-rl python -c "from centipede_env import make_centipede_env; env=make_centipede_env(max_episode_steps=5); obs,_=env.reset(); print(obs.shape, env.action_space.shape); env.close()"
```

## Обучение

Монтируем локальную папку моделей, чтобы чекпоинты не пропадали после удаления
контейнера:

```bash
mkdir -p centipede/models
docker run --rm -it \
  -v "$PWD/centipede/models:/app/centipede/models" \
  centipede-rl \
  python train.py
```

Универсальная политика по командам `(v_x, w_z)`:

```bash
docker run --rm -it \
  -v "$PWD/centipede/models:/app/centipede/models" \
  centipede-rl \
  python train.py --reward-mode command
```

Специалисты:

```bash
docker run --rm -it \
  -v "$PWD/centipede/models:/app/centipede/models" \
  centipede-rl \
  python train.py forward

docker run --rm -it \
  -v "$PWD/centipede/models:/app/centipede/models" \
  centipede-rl \
  python train.py backward

docker run --rm -it \
  -v "$PWD/centipede/models:/app/centipede/models" \
  centipede-rl \
  python train.py left

docker run --rm -it \
  -v "$PWD/centipede/models:/app/centipede/models" \
  centipede-rl \
  python train.py right
```

Продолжить совместимое обучение:

```bash
docker run --rm -it \
  -v "$PWD/centipede/models:/app/centipede/models" \
  centipede-rl \
  python train.py --resume
```

## Просмотр / Manual

GUI из Docker зависит от ОС. Для обучения окно не нужно. На macOS с Docker
Desktop самый надёжный вариант: обучать в Docker, а `play.py` запускать локально
в host-venv.

Если есть настроенный X server, можно пробросить display:

```bash
docker run --rm -it \
  -e DISPLAY="$DISPLAY" \
  -e MUJOCO_GL=glfw \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/centipede/models:/app/centipede/models" \
  centipede-rl \
  python play.py --manual
```

Headless random/manual-проверки без окна потребуют отдельного тестового скрипта.
Текущий `play.py` намеренно открывает pygame-окно.
