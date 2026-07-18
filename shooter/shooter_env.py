"""
shooter_env.py — Gymnasium-среда: агент против бота (скриптового или RL-копии).

Self-play: каждые N шагов вызывается update_bot_model(model),
и бот начинает играть копией текущей политики агента.

Observation (32 значения):
    rays[0..7]       — 8 лучей до стен [0..1]
    sin(angle), cos(angle) — направление пушки
    state            — 0=SPIN 1=MOVING
    vx_norm, vy_norm — скорость
    hp_norm          — HP [0..1]
    shoot_cd_norm    — cooldown выстрела [0..1]
    sin(dir_to_opp)  — направление на противника (относительно пушки)
    cos(dir_to_opp)
    dist_norm        — расстояние до противника [0..1]
    sin(opp_angle), cos(opp_angle) — куда смотрит противник
    opp_state
    opp_hp_norm
    opp_vx_norm, opp_vy_norm
    incoming bullet: risk, time_norm, sin/cos направления на пулю, escape_side
    lead aim: sin/cos ошибки до предиктивного выстрела, line_clear

Actions: 0=noop  1=MOVE  2=SHOOT

Reward:
    попадание / убийство / победа
    штраф за полученный урон / смерть
    anti-camping: штраф за долгое стояние, плюс за полезное движение/стрейф
    tactical positioning: не упираться в стены, открывать линию огня
    shaping за уход с траектории входящей пули
    shaping за предиктивный выстрел с учётом времени долёта
    маленький штраф за время и нахождение под угрозой
"""

import math
import os
import sys

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world  import World, TILE, PX_W, PX_H, SCREEN_W, SCREEN_H
from player import Player, RADIUS, MAX_HP
from bullet import Bullet, BULLET_RADIUS, BULLET_SPEED
from bot    import SelfPlayBot

MAX_STEPS = 3000
N_RAYS    = 8
MAX_DIST  = 350.0
MAX_SPEED = 6.0
FPS       = 60

PLAYER_SHOOT_COOLDOWN = 10.0

STEP_PENALTY          = -0.005
HIT_REWARD            = 6.0
KILL_REWARD           = 14.0
WIN_REWARD            = 18.0
GOT_HIT_PENALTY       = -5.0
DEATH_PENALTY         = -16.0
LOSE_PENALTY          = -18.0
TIMEOUT_LEAD_REWARD   = 4.0
TIMEOUT_BEHIND_PENALTY = -6.0
TIMEOUT_DRAW_PENALTY  = -8.0

THREAT_HORIZON_STEPS  = 45.0
THREAT_MARGIN         = 10.0
DANGER_RADIUS         = RADIUS + BULLET_RADIUS + THREAT_MARGIN
DODGE_PROGRESS_REWARD = 1.8
DODGE_CLEAR_REWARD    = 0.8
BAD_DODGE_PENALTY     = 0.12
STAND_IN_FIRE_PENALTY = 0.08
DANGER_RISK_PENALTY   = 0.035

MOVE_REWARD           = 0.018
STRAFE_REWARD         = 0.018
DISTANCE_PROGRESS_COEFF = 0.035
STILL_GRACE_STEPS     = 28
STILL_PENALTY         = 0.018
STILL_PENALTY_CAP     = 0.08
IDEAL_COMBAT_DIST     = 380.0
COMBAT_BAND_WIDTH     = 180.0

WALL_CLOSE_THRESH     = 0.18
WALL_HIT_PENALTY      = 0.45
WALL_RISK_PENALTY     = 0.045
WALL_CLEAR_PROGRESS_REWARD = 0.08
LOS_CLEAR_REWARD      = 0.045
LOS_OPEN_REWARD       = 0.45
LOS_BLOCKED_PENALTY   = 0.025
LOS_DIST_LIMIT        = 760.0
AGGRESSIVE_ADVANCE_COEFF = 0.010

SHOT_ATTEMPT_REWARD   = 0.12
OPEN_SHOT_REWARD      = 0.12
GOOD_OPPORTUNITY_SHOT_REWARD = 0.35
MISSED_SHOT_OPPORTUNITY_PENALTY = 0.04
SHOT_OPPORTUNITY_QUALITY = 0.62
GOOD_SHOT_REWARD      = 1.2
BAD_SHOT_PENALTY      = -0.20
BLOCKED_SHOT_PENALTY  = -0.40
COOLDOWN_SHOT_PENALTY = -0.04
AIM_TRACK_REWARD      = 0.014
LEAD_FULL_ERROR_DEG   = 30.0
LEAD_TIME_CAP         = 90.0

_AGENT_SPAWNS = [
    (2 * TILE + TILE//2,  2 * TILE + TILE//2),
    (2 * TILE + TILE//2,  4 * TILE + TILE//2),
    (4 * TILE + TILE//2,  2 * TILE + TILE//2),
]
_BOT_SPAWNS = [
    (37 * TILE + TILE//2, 22 * TILE + TILE//2),
    (37 * TILE + TILE//2, 20 * TILE + TILE//2),
    (35 * TILE + TILE//2, 22 * TILE + TILE//2),
]

_N_OBS = N_RAYS + 7 + 9 + 5 + 3   # 32


def _angle_diff_deg(a: float, b: float) -> float:
    """Кратчайшая разница углов a-b со знаком, в градусах (диапазон [-180, 180])."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


class ShooterEnv(gym.Env):
    """Среда gymnasium «дуэль два танчика на арене со стенами».

    Для тех, кто не знаком с RL: среда — «игра» со стандартным интерфейсом
    reset()/step(). Агент на каждом шаге получает наблюдение (32 числа:
    лучи до стен, своё состояние, состояние противника, входящая пуля,
    подсказка прицеливания), выбирает одно из 3 действий и получает
    награду. Алгоритм PPO (train.py) учится максимизировать сумму наград.

    Управление намеренно минималистичное (Discrete(3)):
      0 = ничего (пушка крутится сама — режим SPIN),
      1 = MOVE — ехать туда, куда сейчас смотрит пушка,
      2 = SHOOT — выстрелить туда, куда смотрит пушка.
    Вся тактика (когда повернуться, когда ехать, когда стрелять) —
    из выбора МОМЕНТА нажатия, это и делает задачу интересной.

    Self-play: противник (бот) начинает со скриптового поведения, а
    по мере обучения ему периодически подкладывается копия текущей
    политики агента (update_bot_model) — агент играет против себя
    вчерашнего и не может выучить эксплойт против фиксированного бота.
    """

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_N_OBS,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.world = None
        self.agent = None
        self.bot   = None
        self._bot_ctrl = SelfPlayBot()   # начинает со скриптового поведения

        self.agent_bullets: list[Bullet] = []
        self.bot_bullets:   list[Bullet] = []

        self._step_count  = 0
        self._last_reward = 0.0
        self._agent_hits  = 0
        self._bot_hits    = 0
        self._last_reward_parts = {}
        self._agent_still_steps = 0

        self.screen = None
        self.clock  = None
        self.font   = None

    # ------------------------------------------------------------------
    # Self-play: вызывается SelfPlayCallback из train.py
    # ------------------------------------------------------------------

    def update_bot_model(self, model):
        """Подкладывает боту копию текущей политики агента (self-play).
        Вызывается SelfPlayCallback из train.py каждые N шагов обучения."""
        self._bot_ctrl.set_model(model)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Новый бой: агент и бот на противоположных углах карты,
        случайные стартовые углы пушек, пустые списки пуль."""
        super().reset(seed=seed)

        self.world = World(map_idx=0)

        idx    = int(self.np_random.integers(0, len(_AGENT_SPAWNS)))
        ax, ay = _AGENT_SPAWNS[idx]
        bx, by = _BOT_SPAWNS[idx]

        self.agent = Player(ax, ay,
                            angle=float(self.np_random.integers(0, 360)),
                            color_scheme='blue')
        self.bot   = Player(bx, by,
                            angle=float(self.np_random.integers(0, 360)),
                            color_scheme='red')

        self.agent_bullets = []
        self.bot_bullets   = []
        self._step_count   = 0
        self._last_reward  = 0.0
        self._agent_hits   = 0
        self._bot_hits     = 0
        self._last_reward_parts = {}
        self._agent_still_steps = 0

        return self._get_obs(), {}

    def step(self, action: int):
        """Один шаг боя (1/60 секунды).

        Порядок внутри шага:
          1. Запоминаем «до»: дистанции, линию огня, угрозу от пуль —
             многие награды считаются как разница «до/после».
          2. Двигаем агента по его действию, затем бота по его действию
             (бот получает СВОЁ наблюдение — среда симметрична).
          3. Начисляем шейпинг-награды (движение, тактика, качество выстрела).
          4. Двигаем все пули, проверяем попадания в обе стороны.
          5. Награды за уклонение, попадания, победу/поражение/таймаут.

        Возвращает (obs, reward, terminated, truncated, info); в
        info["reward_parts"] — разбивка награды по слагаемым для отладки.
        """
        self._step_count += 1
        reward = STEP_PENALTY
        reward_parts = {"time": STEP_PENALTY}

        pre_x, pre_y = self.agent.x, self.agent.y
        pre_dist = self._distance_between(self.agent, self.bot)
        pre_dist_error = self._combat_distance_error(self.agent, self.bot)
        pre_los_clear = self._line_clear(self.agent.x, self.agent.y,
                                         self.bot.x, self.bot.y)
        pre_wall_risk = self._wall_risk(self.agent)
        pre_threat = self._incoming_threat(self.agent, self.bot_bullets)

        # Обновляем агента
        new_agent_bullets = self.agent.update(action, self.world)
        self.agent_bullets.extend(new_agent_bullets)
        move_dx = self.agent.x - pre_x
        move_dy = self.agent.y - pre_y
        move_len = math.hypot(move_dx, move_dy)
        agent_moved = move_len > 0.2
        if agent_moved:
            self._agent_still_steps = 0
        else:
            self._agent_still_steps += 1

        # Получаем действие бота (из его собственной наблюдаемости)
        bot_obs     = self._get_obs_for(self.bot, self.agent, self.agent_bullets)
        bot_action  = self._bot_ctrl.get_action(self.bot, self.agent, self.world, bot_obs)
        new_bot_bullets = self.bot.update(bot_action, self.world)
        self.bot_bullets.extend(new_bot_bullets)

        movement_reward = self._movement_reward(
            pre_dist_error=pre_dist_error,
            move_dx=move_dx,
            move_dy=move_dy,
            moved=agent_moved,
        )
        if movement_reward:
            reward += movement_reward
            reward_parts["movement"] = movement_reward

        tactical_reward = self._tactical_position_reward(
            action=action,
            pre_dist=pre_dist,
            pre_los_clear=pre_los_clear,
            pre_wall_risk=pre_wall_risk,
            moved=agent_moved,
        )
        if tactical_reward:
            reward += tactical_reward
            reward_parts["tactical"] = tactical_reward

        aim_eval = self._shot_quality(self.agent, self.bot)
        shot_opportunity = (
            self.bot.alive
            and self.agent.shoot_cooldown == 0
            and aim_eval["line_clear"]
            and aim_eval["quality"] >= SHOT_OPPORTUNITY_QUALITY
        )

        # Reward за качество выстрела агента.
        if action == 2:
            if new_agent_bullets:
                shot_eval = self._shot_quality(self.agent, self.bot, new_agent_bullets[0])
                shot_reward = (
                    SHOT_ATTEMPT_REWARD
                    + BAD_SHOT_PENALTY
                    + (GOOD_SHOT_REWARD - BAD_SHOT_PENALTY) * shot_eval["quality"]
                )
                if shot_eval["line_clear"]:
                    shot_reward += OPEN_SHOT_REWARD
                    if shot_eval["quality"] >= SHOT_OPPORTUNITY_QUALITY:
                        shot_reward += GOOD_OPPORTUNITY_SHOT_REWARD
                else:
                    shot_reward += BLOCKED_SHOT_PENALTY
                reward += shot_reward
                reward_parts["shot_quality"] = shot_reward
            else:
                reward += COOLDOWN_SHOT_PENALTY
                reward_parts["shot_cooldown"] = COOLDOWN_SHOT_PENALTY
        elif shot_opportunity:
            reward -= MISSED_SHOT_OPPORTUNITY_PENALTY
            reward_parts["missed_shot_opportunity"] = -MISSED_SHOT_OPPORTUNITY_PENALTY

        # Двигаем пули агента, проверяем попадание в бота
        for b in self.agent_bullets:
            b.update(self.world)
        bot_killed = False
        for b in self.agent_bullets:
            if b.alive and self.bot.alive and b.hits(self.bot.x, self.bot.y, RADIUS):
                b.alive = False
                killed  = not self.bot.hit()
                bot_killed = killed
                hit_reward = KILL_REWARD if killed else HIT_REWARD
                reward += hit_reward
                reward_parts["hit_bot"] = reward_parts.get("hit_bot", 0.0) + hit_reward
                self._agent_hits += 1

        # Двигаем пули бота, проверяем попадание в агента
        for b in self.bot_bullets:
            b.update(self.world)
        agent_was_hit = False
        agent_killed = False
        for b in self.bot_bullets:
            if b.alive and self.agent.alive and b.hits(self.agent.x, self.agent.y, RADIUS):
                b.alive = False
                killed  = not self.agent.hit()
                agent_was_hit = True
                agent_killed = killed
                hit_penalty = DEATH_PENALTY if killed else GOT_HIT_PENALTY
                reward += hit_penalty
                reward_parts["hit_agent"] = reward_parts.get("hit_agent", 0.0) + hit_penalty
                self._bot_hits += 1

        post_threat = self._incoming_threat(self.agent, self.bot_bullets)
        dodge_reward = self._dodge_reward(
            pre_threat=pre_threat,
            post_threat=post_threat,
            moved=agent_moved,
            was_hit=agent_was_hit,
        )
        if dodge_reward:
            reward += dodge_reward
            reward_parts["dodge"] = dodge_reward

        danger_penalty = -post_threat["risk"] * DANGER_RISK_PENALTY
        if danger_penalty:
            reward += danger_penalty
            reward_parts["danger"] = danger_penalty

        if self.bot.alive and self.agent.alive:
            if aim_eval["line_clear"] and aim_eval["quality"] > 0.80:
                aim_reward = AIM_TRACK_REWARD * ((aim_eval["quality"] - 0.80) / 0.20)
                reward += aim_reward
                reward_parts["aim_track"] = aim_reward

        if bot_killed and self.agent.alive:
            reward += WIN_REWARD
            reward_parts["win"] = WIN_REWARD
        if agent_killed:
            reward += LOSE_PENALTY
            reward_parts["lose"] = LOSE_PENALTY

        self.agent_bullets = [b for b in self.agent_bullets if b.alive]
        self.bot_bullets   = [b for b in self.bot_bullets   if b.alive]

        terminated = not self.bot.alive or not self.agent.alive
        truncated  = self._step_count >= MAX_STEPS

        result = "running"
        if not self.bot.alive and self.agent.alive:
            result = "win"
        elif not self.agent.alive:
            result = "lose"
        elif truncated:
            if self.agent.hp > self.bot.hp:
                result = "timeout_win"
                reward += TIMEOUT_LEAD_REWARD
                reward_parts["timeout"] = TIMEOUT_LEAD_REWARD
            elif self.agent.hp < self.bot.hp:
                result = "timeout_loss"
                reward += TIMEOUT_BEHIND_PENALTY
                reward_parts["timeout"] = TIMEOUT_BEHIND_PENALTY
            else:
                result = "timeout_draw"
                reward += TIMEOUT_DRAW_PENALTY
                reward_parts["timeout"] = TIMEOUT_DRAW_PENALTY

        self._last_reward = reward
        self._last_reward_parts = reward_parts
        info = {
            "agent_hits": self._agent_hits,
            "bot_hits":   self._bot_hits,
            "steps":      self._step_count,
            "agent_hp":   self.agent.hp,
            "bot_hp":     self.bot.hp,
            "win":        result in ("win", "timeout_win"),
            "result":     result,
            "threat_risk": post_threat["risk"],
            "still_steps": self._agent_still_steps,
            "wall_risk":  self._wall_risk(self.agent),
            "line_clear": self._line_clear(self.agent.x, self.agent.y,
                                           self.bot.x, self.bot.y),
            "shot_opportunity": shot_opportunity,
            "reward_parts": reward_parts,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """Рисует кадр: карта, оба игрока, все пули, HUD со здоровьем."""
        if self.render_mode != "human":
            return
        self._ensure_pygame()

        self.world.draw(self.screen)
        self.bot.draw(self.screen)
        self.agent.draw(self.screen)
        for b in self.agent_bullets:
            b.draw(self.screen)
        for b in self.bot_bullets:
            b.draw(self.screen)
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(FPS)
        pygame.event.pump()

    def close(self):
        """Закрывает окно pygame (если было открыто)."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # ------------------------------------------------------------------
    # Observation — симметричная: любого игрока можно подставить
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Наблюдение агента (обёртка над _get_obs_for)."""
        return self._get_obs_for(self.agent, self.bot, self.bot_bullets)

    def _get_obs_for(self, player, opponent, incoming_bullets=None) -> np.ndarray:
        """Наблюдение с точки зрения player, где opponent — враг.

        Симметричность важна для self-play: одна и та же функция строит
        наблюдение и для агента, и для бота (просто меняются местами
        аргументы), поэтому политику агента можно без изменений
        подставить боту.

        Состав (32 числа, все нормированы в [-1, 1]):
          8 лучей до стен + 7 о себе (курс, состояние, скорость, HP,
          перезарядка) + 9 о противнике + 5 о самой опасной входящей
          пуле + 3 подсказки прицеливания с упреждением.
        """
        if incoming_bullets is None:
            incoming_bullets = []

        rays = player.cast_rays(self.world, N_RAYS, MAX_DIST)

        rad  = math.radians(player.angle)
        vx_n = float(np.clip(player.vx / MAX_SPEED, -1, 1))
        vy_n = float(np.clip(player.vy / MAX_SPEED, -1, 1))
        hp_n = player.hp / MAX_HP
        cd_n = float(np.clip(player.shoot_cooldown / PLAYER_SHOOT_COOLDOWN, 0, 1))

        player_obs = [math.sin(rad), math.cos(rad),
                      float(player.state), vx_n, vy_n, hp_n, cd_n]

        dx    = opponent.x - player.x
        dy    = opponent.y - player.y
        dist  = math.hypot(dx, dy)
        dir_a = math.atan2(dy, dx) - rad
        dn    = float(np.clip(dist / MAX_DIST, 0, 1))
        ovx_n = float(np.clip(opponent.vx / MAX_SPEED, -1, 1))
        ovy_n = float(np.clip(opponent.vy / MAX_SPEED, -1, 1))

        orad = math.radians(opponent.angle)
        opp_obs = [
            math.sin(dir_a), math.cos(dir_a), dn,
            math.sin(orad),  math.cos(orad),
            float(opponent.state), opponent.hp / MAX_HP,
            ovx_n, ovy_n,
        ]

        threat = self._incoming_threat(player, incoming_bullets)
        threat_angle = threat["bearing"] - rad
        threat_obs = [
            threat["risk"],
            threat["time_norm"],
            math.sin(threat_angle),
            math.cos(threat_angle),
            threat["escape_side"],
        ]

        lead_eval = self._shot_quality(player, opponent)
        aim_error = math.radians(lead_eval["angle_error"])
        aim_obs = [
            math.sin(aim_error),
            math.cos(aim_error),
            1.0 if lead_eval["line_clear"] else 0.0,
        ]

        obs = np.array(rays + player_obs + opp_obs + threat_obs + aim_obs,
                       dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Reward geometry
    # ------------------------------------------------------------------

    def _distance_between(self, a, b) -> float:
        """Евклидово расстояние между двумя игроками, px."""
        return math.hypot(a.x - b.x, a.y - b.y)

    def _combat_distance_error(self, player, opponent) -> float:
        """Насколько дистанция до противника вне «полезной боевой полосы».

        0, пока дистанция в пределах IDEAL_COMBAT_DIST +- COMBAT_BAND_WIDTH;
        растёт, когда слишком близко (легко словить пулю) или слишком
        далеко (не попасть самому). Используется в награде за движение.
        """
        dist = self._distance_between(player, opponent)
        return max(0.0, abs(dist - IDEAL_COMBAT_DIST) - COMBAT_BAND_WIDTH)

    def _wall_risk(self, player) -> float:
        """0 = свободно, 1 = очень близко к стене/ограждению."""
        rays = player.cast_rays(self.world, N_RAYS, MAX_DIST)
        min_ray = min(rays) if rays else 1.0
        if min_ray >= WALL_CLOSE_THRESH:
            return 0.0
        return float(np.clip((WALL_CLOSE_THRESH - min_ray) / WALL_CLOSE_THRESH,
                             0.0, 1.0))

    def _movement_reward(self, pre_dist_error: float,
                         move_dx: float, move_dy: float,
                         moved: bool) -> float:
        """Награда за полезное движение (анти-кемпинг).

        Слагаемые: небольшой плюс за само движение; плюс за приближение
        к полезной боевой дистанции; плюс за стрейф (движение ПОПЕРЁК
        линии огня — так сложнее попасть по агенту), но только когда
        линия огня открыта; растущий штраф за стояние на месте дольше
        STILL_GRACE_STEPS шагов.
        """
        reward = 0.0

        if moved:
            reward += MOVE_REWARD

            post_dist_error = self._combat_distance_error(self.agent, self.bot)
            reward += (pre_dist_error - post_dist_error) * DISTANCE_PROGRESS_COEFF

            move_len = math.hypot(move_dx, move_dy)
            to_bot_x = self.bot.x - self.agent.x
            to_bot_y = self.bot.y - self.agent.y
            to_bot_len = math.hypot(to_bot_x, to_bot_y)
            if move_len > 1e-6 and to_bot_len > 1e-6:
                # Стрейф: боковое движение относительно линии агент->бот.
                lateral = abs(move_dx * to_bot_y - move_dy * to_bot_x) / (move_len * to_bot_len)
                line_clear = self._line_clear(self.agent.x, self.agent.y, self.bot.x, self.bot.y)
                if line_clear:
                    reward += lateral * STRAFE_REWARD
        elif self._agent_still_steps > STILL_GRACE_STEPS:
            excess = self._agent_still_steps - STILL_GRACE_STEPS
            reward -= min(STILL_PENALTY_CAP, STILL_PENALTY * (1.0 + excess / 60.0))

        return reward

    def _tactical_position_reward(self, action: int, pre_dist: float,
                                  pre_los_clear: bool, pre_wall_risk: float,
                                  moved: bool) -> float:
        """Награда за тактически грамотную позицию.

        Штрафы: упёрся в стену (жал MOVE, но не сдвинулся), стоит
        вплотную к стене, линия огня перекрыта. Бонусы: отошёл от стены,
        держит открытую линию огня в боевой дальности (и особенно —
        только что её открыл), сближается, когда слишком далеко.
        """
        reward = 0.0

        post_dist = self._distance_between(self.agent, self.bot)
        post_los_clear = self._line_clear(self.agent.x, self.agent.y,
                                          self.bot.x, self.bot.y)
        post_wall_risk = self._wall_risk(self.agent)

        if action == 1 and not moved:
            reward -= WALL_HIT_PENALTY

        reward -= post_wall_risk * WALL_RISK_PENALTY
        if moved:
            reward += max(0.0, pre_wall_risk - post_wall_risk) * WALL_CLEAR_PROGRESS_REWARD

        in_fight_range = post_dist <= LOS_DIST_LIMIT
        if post_los_clear and in_fight_range:
            reward += LOS_CLEAR_REWARD
            if not pre_los_clear:
                reward += LOS_OPEN_REWARD
        else:
            reward -= LOS_BLOCKED_PENALTY

        if moved and post_dist > IDEAL_COMBAT_DIST:
            reward += max(0.0, pre_dist - post_dist) * AGGRESSIVE_ADVANCE_COEFF

        return reward

    def _incoming_threat(self, player, bullets) -> dict:
        """Находит самую опасную летящую в игрока пулю.

        Для каждой пули аналитически считается точка её максимального
        сближения с игроком (проекция на траекторию). Пуля опасна, если
        сближение меньше DANGER_RADIUS, произойдёт в ближайшие
        THREAT_HORIZON_STEPS шагов и по пути нет стены.

        Возвращает словарь: risk (0..1 — насколько страшно), time
        (шагов до сближения), bearing (направление на пулю),
        escape_side (+-1 — в какую сторону уворачиваться), intersects
        (True = траектория реально задевает корпус).
        Эти же значения идут в наблюдения — агент «видит» угрозу.
        """
        best = {
            "risk": 0.0,
            "time": THREAT_HORIZON_STEPS,
            "time_norm": 1.0,
            "bearing": math.radians(player.angle),
            "escape_side": 0.0,
            "intersects": False,
            "closest": float("inf"),
        }

        for b in bullets:
            if not b.alive:
                continue

            rel_x = player.x - b.x
            rel_y = player.y - b.y
            speed_sq = b.vx * b.vx + b.vy * b.vy
            if speed_sq <= 1e-9:
                continue

            t = (rel_x * b.vx + rel_y * b.vy) / speed_sq
            if t <= 0.0 or t > THREAT_HORIZON_STEPS:
                continue

            closest_x = b.x + b.vx * t
            closest_y = b.y + b.vy * t
            closest = math.hypot(player.x - closest_x, player.y - closest_y)
            if closest > DANGER_RADIUS:
                continue

            # Если пуля раньше упирается в стену, эта траектория не опасна.
            if not self._line_clear(b.x, b.y, closest_x, closest_y):
                continue

            proximity = 1.0 - min(closest / DANGER_RADIUS, 1.0)
            urgency = 1.0 - 0.5 * min(t / THREAT_HORIZON_STEPS, 1.0)
            risk = float(np.clip(proximity * urgency, 0.0, 1.0))
            if risk <= best["risk"]:
                continue

            bearing = math.atan2(b.y - player.y, b.x - player.x)
            cross = b.vx * rel_y - b.vy * rel_x
            if abs(cross) < 1e-6:
                escape_side = 0.0
            else:
                escape_side = 1.0 if cross > 0.0 else -1.0

            best = {
                "risk": risk,
                "time": t,
                "time_norm": float(np.clip(t / THREAT_HORIZON_STEPS, 0.0, 1.0)),
                "bearing": bearing,
                "escape_side": escape_side,
                "intersects": closest <= (RADIUS + BULLET_RADIUS),
                "closest": closest,
            }

        return best

    def _dodge_reward(self, pre_threat: dict, post_threat: dict,
                      moved: bool, was_hit: bool) -> float:
        """Награда за уклонение от входящей пули.

        Сравнивает угрозу до и после шага: снизил риск движением — плюс,
        полностью ушёл с траектории — бонус, дёрнулся бесполезно — мелкий
        штраф, стоял столбом под пулей — штраф. Если уже попали (was_hit)
        или угрозы не было — ничего не начисляется.
        """
        if was_hit or pre_threat["risk"] <= 0.0:
            return 0.0

        reward = 0.0
        if pre_threat["intersects"]:
            risk_drop = max(0.0, pre_threat["risk"] - post_threat["risk"])
            if moved:
                reward += risk_drop * DODGE_PROGRESS_REWARD
                if post_threat["risk"] < 0.05:
                    reward += pre_threat["risk"] * DODGE_CLEAR_REWARD
                elif risk_drop < 0.03:
                    reward -= pre_threat["risk"] * BAD_DODGE_PENALTY
            else:
                reward -= pre_threat["risk"] * STAND_IN_FIRE_PENALTY
        return reward

    def _lead_solution(self, shooter, target) -> dict:
        """Точка упреждения: куда стрелять, чтобы пуля встретила движущуюся цель.

        Классическая задача перехвата: пуля летит со скоростью BULLET_SPEED,
        цель — со своей скоростью; решаем квадратное уравнение относительно
        времени встречи t (коэффициенты a, b, c ниже) и берём наименьший
        положительный корень. Если решения нет (цель быстрее пули «убегает») —
        целимся в текущую позицию по времени прямого долёта.

        Возвращает: координаты точки перехвата, время долёта, угол
        выстрела и line_clear (нет ли стены на пути пули).
        """
        rx = target.x - shooter.x
        ry = target.y - shooter.y
        vx = target.vx
        vy = target.vy

        a = vx * vx + vy * vy - BULLET_SPEED * BULLET_SPEED
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry

        t = None
        if abs(a) < 1e-9:
            if abs(b) > 1e-9:
                candidate = -c / b
                if candidate > 0.0:
                    t = candidate
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = math.sqrt(disc)
                candidates = [
                    (-b - root) / (2.0 * a),
                    (-b + root) / (2.0 * a),
                ]
                candidates = [v for v in candidates if v > 0.0]
                if candidates:
                    t = min(candidates)

        if t is None:
            t = math.hypot(rx, ry) / BULLET_SPEED
        t = float(np.clip(t, 0.0, LEAD_TIME_CAP))

        px = target.x + target.vx * t
        py = target.y + target.vy * t
        px = float(np.clip(px, RADIUS, PX_W - RADIUS))
        py = float(np.clip(py, RADIUS, PX_H - RADIUS))
        angle = math.degrees(math.atan2(py - shooter.y, px - shooter.x)) % 360.0
        clear = self._line_clear(shooter.x, shooter.y, px, py)
        return {"x": px, "y": py, "time": t, "angle": angle, "line_clear": clear}

    def _shot_quality(self, shooter, target, bullet: Bullet | None = None) -> dict:
        """Оценивает качество прицеливания (0..1) относительно точки упреждения.

        quality = 1 — пушка (или уже выпущенная пуля, если передана)
        смотрит точно в точку перехвата; линейно падает до 0 при ошибке
        LEAD_FULL_ERROR_DEG градусов. Если линия огня перекрыта стеной,
        качество режется в 4 раза. Используется и в награде за выстрел,
        и как подсказка в наблюдениях.
        """
        if not target.alive:
            return {"quality": 0.0, "angle_error": 0.0, "line_clear": False}

        lead = self._lead_solution(shooter, target)
        if bullet is None:
            shot_angle = shooter.angle % 360.0
        else:
            shot_angle = math.degrees(math.atan2(bullet.vy, bullet.vx)) % 360.0

        angle_error = abs(_angle_diff_deg(shot_angle, lead["angle"]))
        quality = max(0.0, 1.0 - angle_error / LEAD_FULL_ERROR_DEG)
        if not lead["line_clear"]:
            quality *= 0.25
        return {
            "quality": float(np.clip(quality, 0.0, 1.0)),
            "angle_error": _angle_diff_deg(shot_angle, lead["angle"]),
            "line_clear": lead["line_clear"],
        }

    def _line_clear(self, x1: float, y1: float, x2: float, y2: float,
                    step: float = 6.0) -> bool:
        """Свободна ли прямая между двумя точками от стен.

        Простейший рейкаст: идём по отрезку с шагом step пикселей и
        проверяем каждую точку на попадание в стену.
        """
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist <= 1e-6:
            return True
        steps = max(1, int(dist / step))
        for i in range(1, steps + 1):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            if self.world.is_solid(x, y):
                return False
        return True

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self):
        """HUD: панели здоровья обоих игроков + счётчики шагов/попаданий.
        Подпись бота показывает, играет ли он уже RL-копией агента."""
        using_rl = self._bot_ctrl._model is not None
        bot_label = "БОТ (RL)" if using_rl else "БОТ"
        self._draw_hp_panel(10,             10, "АГЕНТ",    self.agent.hp, (60, 140, 255))
        self._draw_hp_panel(SCREEN_W - 170, 10, bot_label,  self.bot.hp,   (220, 60, 60))

        lines = [
            f"Шаг: {self._step_count}",
            f"Попаданий: {self._agent_hits}",
            f"Получено:  {self._bot_hits}",
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(surf, (SCREEN_W // 2 - 55, 10 + i * 20))

    def _draw_hp_panel(self, x, y, label, hp, color):
        """Панель здоровья: подпись + ряд прямоугольников (закрашен = живое HP)."""
        surf = self.font.render(label, True, color)
        self.screen.blit(surf, (x, y))
        for i in range(MAX_HP):
            col = color if i < hp else (60, 60, 60)
            pygame.draw.rect(self.screen, col, (x + i * 30, y + 20, 24, 12))
            pygame.draw.rect(self.screen, (200, 200, 200), (x + i * 30, y + 20, 24, 12), 1)

    def _ensure_pygame(self):
        """Ленивая инициализация pygame: окно создаётся при первом рендере
        (при обучении без графики оно не нужно)."""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("RL Shooter — Self-Play")
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.clock  = pygame.time.Clock()
            self.font   = pygame.font.SysFont("monospace", 15)
