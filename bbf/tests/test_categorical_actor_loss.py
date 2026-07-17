import unittest

import jax
import jax.numpy as jnp
import numpy as np

from bbf.agents import spr_agent


class CategoricalActorLossTest(unittest.TestCase):

    def test_exact_matches_discrete_sac_objective_and_is_finite_under_jit(self):
        logits = jnp.array([0.3, -0.7, 1.2], dtype=jnp.float32)
        q_values = jnp.array([1.5, -2.0, 0.25], dtype=jnp.float32)
        alpha = jnp.asarray(0.07, dtype=jnp.float32)
        key = jax.random.PRNGKey(0)
        loss, entropy = jax.jit(
            lambda z: spr_agent.categorical_actor_loss(
                q_values, z, key, alpha, "exact"))(logits)
        log_prob = jax.nn.log_softmax(logits)
        prob = jnp.exp(log_prob)
        expected = jnp.sum(prob * (alpha * log_prob - q_values))
        np.testing.assert_allclose(loss, expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(np.isfinite(np.asarray(entropy)))

    def test_exact_actor_gradient_does_not_flow_into_q_values(self):
        logits = jnp.array([0.4, -0.2, 0.1], dtype=jnp.float32)
        q_values = jnp.array([2.0, -1.0, 0.5], dtype=jnp.float32)
        key = jax.random.PRNGKey(1)

        def loss(q, z):
            return spr_agent.categorical_actor_loss(q, z, key, 0.03,
                                                    "exact")[0]

        q_grad, logits_grad = jax.grad(loss, argnums=(0, 1))(q_values, logits)
        np.testing.assert_array_equal(q_grad, jnp.zeros_like(q_values))
        self.assertTrue(np.all(np.isfinite(np.asarray(logits_grad))))
        self.assertGreater(np.linalg.norm(np.asarray(logits_grad)), 0.0)

    def test_sampled_and_exact_policy_gradients_agree_in_expectation(self):
        logits = jnp.array([0.2, -0.4, 0.9], dtype=jnp.float32)
        q_values = jnp.array([1.0, -0.5, 2.0], dtype=jnp.float32)
        alpha = 0.05
        exact_grad = jax.grad(lambda z: spr_agent.categorical_actor_loss(
            q_values, z, jax.random.PRNGKey(0), alpha, "exact")[0])(logits)
        keys = jax.random.split(jax.random.PRNGKey(7), 20000)
        sampled_grad = jax.vmap(jax.grad(
            lambda z, key: spr_agent.categorical_actor_loss(
                q_values, z, key, alpha, "sampled")[0]),
                                 in_axes=(None, 0))(logits, keys).mean(0)
        np.testing.assert_allclose(sampled_grad, exact_grad,
                                   rtol=0.03, atol=0.01)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "actor_loss_mode"):
            spr_agent.categorical_actor_loss(jnp.ones(2), jnp.zeros(2),
                                             jax.random.PRNGKey(0), 0.0,
                                             "invalid")


if __name__ == '__main__':
    unittest.main()
