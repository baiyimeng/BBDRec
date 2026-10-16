"""CPU mathematical checks for plotting paths; no checkpoints or training."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'analysis'))
from trajectory_sampling import sample_bridge_path, sample_ddpm_path, trajectory_rng


class TrajectorySamplingTests(unittest.TestCase):
    def test_streams_reproducible_but_not_shared(self):
        a = trajectory_rng(42, 0, 0, 0).normal(size=20)
        np.testing.assert_array_equal(a, trajectory_rng(42, 0, 0, 0).normal(size=20))
        for indices in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
            self.assertFalse(np.array_equal(a, trajectory_rng(42, *indices).normal(size=20)))

    def test_bridge_exact_endpoints(self):
        beta = np.array([.1, .4, .8, 1.0])
        x0, endpoint = np.array([2., -1.]), np.array([-3., 4.])
        path = sample_bridge_path(x0, endpoint, beta, 8*beta*(1-beta), np.random.default_rng(9))
        np.testing.assert_array_equal(path[0], x0)
        np.testing.assert_allclose(path[-1], endpoint, atol=1e-14)

    def test_zero_noise_is_linear_interpolation(self):
        beta = np.array([.1, .3, .7, 1.])
        path = sample_bridge_path(np.array([2.]), np.array([6.]), beta, np.zeros(4), np.random.default_rng(1))
        np.testing.assert_allclose(path[1:, 0], (1-beta)*2+beta*6)

    def test_bridge_marginals_and_cross_time_covariance(self):
        n = 80000
        beta = np.array([.2, .6, .9])
        variance = 2*beta*(1-beta)
        path = sample_bridge_path(np.full((n, 1), 2.), np.full((n, 1), -1.), beta, variance, np.random.default_rng(7))
        states = path[1:, :, 0]
        np.testing.assert_allclose(states.mean(axis=1), (1-beta)*2-beta, atol=.012)
        np.testing.assert_allclose(states.var(axis=1), variance, atol=.012)
        covariance = np.cov(states)
        expected = 2*(np.minimum.outer(beta, beta)-np.outer(beta, beta))
        np.testing.assert_allclose(covariance, expected, atol=.012)

    def test_truncated_grid_remains_truncated(self):
        beta = np.array([1e-4, .5, 1-1e-4])
        path = sample_bridge_path(np.array([0.]), np.array([3.]), beta, beta*(1-beta), np.random.default_rng(2))
        self.assertEqual(path.shape, (4, 1))
        self.assertNotEqual(float(path[-1, 0]), 3.)

    def test_ddpm_marginals_and_markov_covariance(self):
        n = 80000
        betas = np.array([.1, .3, .7])
        path = sample_ddpm_path(np.full((n, 1), 2.), betas, np.random.default_rng(6))[1:, :, 0]
        alpha = np.cumprod(1-betas)
        np.testing.assert_allclose(path.mean(axis=1), 2*np.sqrt(alpha), atol=.012)
        np.testing.assert_allclose(path.var(axis=1), 1-alpha, atol=.012)
        expected = np.sqrt(alpha[2]/alpha[0])*(1-alpha[0])
        self.assertAlmostEqual(float(np.cov(path)[0, 2]), float(expected), delta=.012)

    def test_independent_terminal_noise(self):
        a = sample_ddpm_path(np.zeros(20), [1.], trajectory_rng(42, 0, 1, 0))[-1]
        b = sample_ddpm_path(np.zeros(20), [1.], trajectory_rng(42, 0, 1, 1))[-1]
        self.assertFalse(np.array_equal(a, b))

    def test_invalid_schedules_fail(self):
        for betas in [[], [0.], [1.1], [np.nan]]:
            with self.assertRaises(ValueError):
                sample_ddpm_path([0.], betas, np.random.default_rng(1))
        for beta, variance in [([.6, .3], [.2, .2]), ([.2, .5], [.5, 0]), ([1.], [.1])]:
            with self.assertRaises(ValueError):
                sample_bridge_path([0.], [1.], beta, variance, np.random.default_rng(1))


if __name__ == '__main__':
    unittest.main()
