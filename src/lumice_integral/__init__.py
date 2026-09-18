"""Numerical building blocks for Lumice Integral."""

import os
import sys

# Enable JAX float64 without importing JAX here, so that pure-numpy
# subpackages (``geometry``) no longer pull JAX in as a side effect.
#
# Two cases, both covered:
# - JAX not imported yet: JAX reads ``JAX_ENABLE_X64`` when it is first
#   imported, and this ``__init__`` runs before any submodule, so the flag is
#   in place whichever module imports JAX first.
# - JAX already imported by the caller (tests / scripts that ``import jax``
#   before this package): the env var is read too late, so flip the runtime
#   config directly -- ``sys.modules["jax"]`` is already loaded, no new import.
os.environ.setdefault("JAX_ENABLE_X64", "1")
if "jax" in sys.modules:
    sys.modules["jax"].config.update("jax_enable_x64", True)
