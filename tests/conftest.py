"""Process-level test defaults, set before any test module imports JAX or numpy.

Every value is a ``setdefault``: an explicit setting in the caller's
environment wins.  They are inherited by the worker processes some tests
spawn.

- One thread per process for BLAS/OpenMP and XLA's Eigen pool: ``pytest -n``
  runs one process per core, and the multi-worker commands in ``AGENTS.md``
  pin the same knobs for the same reason (no oversubscription).
- ``JAX_PLATFORMS=cpu``: tests are deterministic and do not depend on which
  jax extra happens to be installed.
- JAX's persistent compilation cache in ``<repo>/.jax-cache`` (git-ignored):
  a second run loads the compiled XLA executables instead of recompiling.
  The minimum compile time is lowered from JAX's 1 s to 0: almost every kernel
  here compiles faster than that, and with the default only one entry was
  cached (no gain); with 0 the fast tier is ~20 MB of cache and 12 xdist
  workers also share each other's compiles within one cold run.
  JAX reads these variables when it is imported, so nothing here imports JAX
  (``lumice_integral`` must still set ``JAX_ENABLE_X64`` first).
"""

import os
from pathlib import Path

_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    "JAX_PLATFORMS": "cpu",
    "JAX_COMPILATION_CACHE_DIR": str(Path(__file__).resolve().parent.parent / ".jax-cache"),
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
}

for _name, _value in _DEFAULTS.items():
    os.environ.setdefault(_name, _value)
