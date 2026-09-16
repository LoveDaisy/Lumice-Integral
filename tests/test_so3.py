from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.so3 import exp, log, rotation_distance


@pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-5, 1e-3, 0.1, 1.0, 3.0])
def test_log_inverts_exp_on_the_injectivity_domain(angle):
    axis = np.array([0.3, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    vector = jnp.asarray(angle * axis, dtype=jnp.float64)

    recovered = np.asarray(log(exp(vector)))

    np.testing.assert_allclose(recovered, angle * axis, rtol=0.0, atol=4e-16 * (1.0 + angle))
    assert float(rotation_distance(jnp.eye(3), exp(vector))) == pytest.approx(angle, abs=2e-16)


def test_log_is_zero_and_ad_safe_at_the_identity():
    np.testing.assert_array_equal(np.asarray(log(jnp.eye(3, dtype=jnp.float64))), 0.0)
    # d/dv log(exp(v)) at v = 0 is the identity: both branches must be AD safe.
    jacobian = np.asarray(jax.jacfwd(lambda v: log(exp(v)))(jnp.zeros(3, dtype=jnp.float64)))
    np.testing.assert_allclose(jacobian, np.eye(3), rtol=0.0, atol=1e-15)
