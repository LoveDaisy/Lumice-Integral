"""Process-level test defaults, set before any test module imports JAX or numpy.

Every value is a ``setdefault``: an explicit setting in the caller's
environment wins.  They are inherited by the worker processes some tests
spawn.

- One thread per process for BLAS/OpenMP and XLA's Eigen pool: ``pytest -n``
  runs one process per core, and the multi-worker commands in ``AGENTS.md``
  pin the same knobs for the same reason (no oversubscription).
- ``JAX_PLATFORMS=cpu``: tests are deterministic and do not depend on which
  jax extra happens to be installed.
"""

import os

_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    "JAX_PLATFORMS": "cpu",
}

for _name, _value in _DEFAULTS.items():
    os.environ.setdefault(_name, _value)
