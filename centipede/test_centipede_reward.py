"""Регрессионные проверки биомеханических слагаемых reward-функции."""

import pickle
import unittest

import numpy as np

try:
    from centipede.centipede_env import (
        GAIT_PHASE_LAG,
        CentipedeEnv,
        _metachronal_coordination_score,
        _stance_swing_score,
        _support_fraction_score,
        _support_quality,
    )
except ModuleNotFoundError as error:  # запуск unittest discover из самой centipede/
    if error.name != "centipede":
        raise
    from centipede_env import (
        GAIT_PHASE_LAG,
        CentipedeEnv,
        _metachronal_coordination_score,
        _stance_swing_score,
        _support_fraction_score,
        _support_quality,
    )


class GaitRewardUnitTests(unittest.TestCase):
    def test_metachronal_wave_beats_synchronous_gait(self):
        segments = 6
        base_phase = 0.37
        phases = base_phase + np.arange(segments)[:, None] * GAIT_PHASE_LAG
        phases = np.repeat(phases, 2, axis=1)
        phases[:, 1] += np.pi

        ideal_score, ideal_activity = _metachronal_coordination_score(
            np.sin(phases), np.cos(phases), GAIT_PHASE_LAG
        )

        synchronous = np.full((segments, 2), base_phase)
        synchronous[:, 1] += np.pi
        sync_score, _ = _metachronal_coordination_score(
            np.sin(synchronous), np.cos(synchronous), GAIT_PHASE_LAG
        )

        self.assertGreater(ideal_score, 0.99)
        self.assertGreater(ideal_activity, 0.80)
        self.assertGreater(ideal_score, sync_score + 0.90)

    def test_both_wave_directions_are_valid(self):
        segment_phase = np.arange(6)[:, None] * -GAIT_PHASE_LAG
        phases = np.repeat(segment_phase, 2, axis=1)
        phases[:, 1] += np.pi
        score, _ = _metachronal_coordination_score(
            np.sin(phases), np.cos(phases), GAIT_PHASE_LAG
        )
        self.assertGreater(score, 0.99)

    def test_static_pose_has_no_gait_activity(self):
        phases = np.arange(6)[:, None] * GAIT_PHASE_LAG
        phases = np.repeat(phases, 2, axis=1)
        phases[:, 1] += np.pi
        _, activity = _metachronal_coordination_score(
            np.sin(phases), np.zeros_like(phases), GAIT_PHASE_LAG
        )
        self.assertEqual(activity, 0.0)

    def test_high_frequency_micro_tremor_is_not_a_step(self):
        phases = np.arange(6)[:, None] * GAIT_PHASE_LAG
        phases = np.repeat(phases, 2, axis=1)
        phases[:, 1] += np.pi
        _, activity = _metachronal_coordination_score(
            0.10 * np.sin(phases),
            2.0 * np.cos(phases),
            GAIT_PHASE_LAG,
        )
        self.assertEqual(activity, 0.0)

    def test_stance_and_swing_must_follow_travel_direction(self):
        contacts = np.array([
            [True, False],
            [False, True],
            [True, False],
        ])
        correct_velocity = np.where(contacts, -0.4, 0.4)
        correct = _stance_swing_score(correct_velocity, contacts, travel_direction=1.0)
        reversed_cycle = _stance_swing_score(-correct_velocity, contacts, travel_direction=1.0)
        backward = _stance_swing_score(-correct_velocity, contacts, travel_direction=-1.0)

        self.assertGreater(correct, 0.95)
        self.assertLess(reversed_cycle, -0.95)
        self.assertAlmostEqual(correct, backward)

    def test_fixed_support_legs_do_not_imitate_stance_cycle(self):
        fixed_half = np.zeros((6, 2), dtype=bool)
        fixed_half[:3] = True
        correct_scores = []
        fixed_scores = []
        offsets = np.arange(6)[:, None] * GAIT_PHASE_LAG
        sides = np.array([[0.0, np.pi]])
        for phase in np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False):
            hip_velocity = np.cos(phase + offsets + sides)
            correct_contacts = hip_velocity < 0.0
            correct_scores.append(_stance_swing_score(
                hip_velocity, correct_contacts, travel_direction=1.0
            ))
            fixed_scores.append(_stance_swing_score(
                hip_velocity, fixed_half, travel_direction=1.0
            ))

        self.assertGreater(np.mean(correct_scores), 0.80)
        self.assertLess(abs(np.mean(fixed_scores)), 0.02)

    def test_support_fraction_targets_half_the_legs(self):
        ideal = _support_fraction_score(0.5, target=0.5, sigma2=0.04)
        airborne = _support_fraction_score(0.0, target=0.5, sigma2=0.04)
        self.assertEqual(ideal, 1.0)
        self.assertLess(airborne, 0.01)

    def test_support_gate_rejects_airborne_and_fully_planted_exploits(self):
        airborne_gate, airborne_cost = _support_quality(0.0, 0.25, 0.75)
        walking_gate, walking_cost = _support_quality(0.5, 0.25, 0.75)
        planted_gate, planted_cost = _support_quality(1.0, 0.25, 0.75)
        self.assertEqual((airborne_gate, planted_gate), (0.0, 0.0))
        self.assertEqual((airborne_cost, planted_cost), (1.0, 1.0))
        self.assertEqual((walking_gate, walking_cost), (1.0, 0.0))


class GaitRewardEnvironmentTests(unittest.TestCase):
    def test_reward_decomposition_and_metrics_are_finite(self):
        env = CentipedeEnv(
            reward_mode="forward",
            terrain_roughness=0.0,
            reset_noise_scale=0.0,
        )
        try:
            observation, _ = env.reset(seed=5)
            self.assertEqual(observation.shape, (111,))
            self.assertTrue(np.isfinite(observation).all())
            for _ in range(10):
                observation, reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                terms = sum(
                    value for key, value in info.items()
                    if key.startswith("reward_") or key.startswith("cost_")
                )
                self.assertAlmostEqual(reward, terms)
                for key in (
                    "gait_coordination",
                    "gait_activity",
                    "gait_motion_gate",
                    "support_ratio",
                    "support_fraction_score",
                    "support_gate",
                    "stance_swing_score",
                    "mean_stance_slip",
                ):
                    self.assertTrue(np.isfinite(info[key]), key)
                self.assertFalse(truncated)
                if terminated:
                    env.reset()
        finally:
            env.close()

    def test_ezpickle_preserves_new_reward_parameters(self):
        env = CentipedeEnv(
            terrain_roughness=0.0,
            gait_duty_factor=0.57,
            gait_phase_lag=0.91,
            foot_slip_speed=0.06,
        )
        try:
            restored = pickle.loads(pickle.dumps(env))
            try:
                self.assertAlmostEqual(restored._gait_duty_factor, 0.57)
                self.assertAlmostEqual(restored._gait_phase_lag, 0.91)
                self.assertAlmostEqual(restored._foot_slip_speed, 0.06)
            finally:
                restored.close()
        finally:
            env.close()

    def test_stop_command_disables_positive_gait_terms(self):
        env = CentipedeEnv(
            reward_mode="command",
            auto_command_resample=False,
            terrain_roughness=0.0,
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=3)
            env.set_command(0.0, 0.0)
            _, _, _, _, info = env.step(env.action_space.sample())
            self.assertEqual(info["gait_motion_gate"], 0.0)
            self.assertEqual(info["reward_gait_coordination"], 0.0)
            self.assertEqual(info["reward_gait_support"], 0.0)
            self.assertEqual(info["reward_stance_swing"], 0.0)
            self.assertEqual(info["cost_gait_support"], 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
