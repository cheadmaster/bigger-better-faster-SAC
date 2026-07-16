import math
import unittest

from bbf import target_update


class TauSchedulerTest(unittest.TestCase):

    def test_constant_schedule_preserves_tau_exactly(self):
        scheduler = target_update.create_tau_scheduler(0.005)

        self.assertEqual(scheduler(-1), 0.005)
        self.assertEqual(scheduler(0), 0.005)
        self.assertEqual(scheduler(10_000), 0.005)

    def test_linear_schedule_reaches_and_clamps_to_endpoints(self):
        scheduler = target_update.create_tau_scheduler(
            0.005, 0.05, schedule="linear", schedule_steps=100)

        self.assertEqual(scheduler(-10), 0.005)
        self.assertEqual(scheduler(0), 0.005)
        self.assertAlmostEqual(scheduler(50), 0.0275)
        self.assertEqual(scheduler(100), 0.05)
        self.assertEqual(scheduler(1_000), 0.05)

    def test_cosine_schedule_is_smooth_and_bounded(self):
        scheduler = target_update.create_tau_scheduler(
            0.01, 0.09, schedule="cosine", schedule_steps=100)
        values = [scheduler(step) for step in range(101)]

        self.assertEqual(values[0], 0.01)
        self.assertAlmostEqual(values[50], 0.05)
        self.assertEqual(values[-1], 0.09)
        self.assertTrue(all(0.01 <= value <= 0.09 for value in values))
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_rejects_invalid_tau_values(self):
        for value in (-0.1, 1.1, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    target_update.create_tau_scheduler(value)

    def test_rejects_invalid_nonconstant_schedule(self):
        with self.assertRaises(ValueError):
            target_update.create_tau_scheduler(
                0.1, 0.2, schedule="linear", schedule_steps=0)
        with self.assertRaises(ValueError):
            target_update.create_tau_scheduler(
                0.2, 0.1, schedule="linear", schedule_steps=10)
        with self.assertRaises(ValueError):
            target_update.create_tau_scheduler(
                0.1, schedule="linear", schedule_steps=10)
        with self.assertRaises(ValueError):
            target_update.create_tau_scheduler(0.1, schedule="unknown")


try:
    import jax
    import jax.numpy as jnp
    from flax.core.frozen_dict import FrozenDict

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


@unittest.skipUnless(
    _JAX_AVAILABLE, "JAX/Flax test dependencies are unavailable")
class PolyakUpdateJaxTest(unittest.TestCase):

    def test_interpolation_preserves_shape_device_and_gradient(self):
        old = FrozenDict({
            "params": {
                "encoder": {
                    "kernel": jnp.ones((2, 3), dtype=jnp.float32),
                },
            },
        })

        def updated_sum(new_kernel):
            new = FrozenDict({
                "params": {
                    "encoder": {
                        "kernel": new_kernel,
                    },
                },
            })
            updated = target_update.polyak_update(
                old,
                new,
                tau=0.25,
            )
            return jnp.sum(updated["params"]["encoder"]["kernel"])

        new_kernel = jnp.full((2, 3), 3.0, dtype=jnp.float32)
        updated_value = jax.jit(updated_sum)(new_kernel)
        gradient = jax.jit(jax.grad(updated_sum))(new_kernel)

        self.assertEqual(gradient.shape, new_kernel.shape)
        self.assertEqual(gradient.dtype, new_kernel.dtype)
        self.assertEqual(gradient.devices(), new_kernel.devices())
        self.assertTrue(jnp.allclose(updated_value, 9.0))
        self.assertTrue(jnp.allclose(gradient, 0.25))


if __name__ == "__main__":
    unittest.main()
