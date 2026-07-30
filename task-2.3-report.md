## Task 2.3 — Quick Fix: Dead `n_starts` Parameter

**Date**: 2026-07-30

**Problem**: MEDIUM severity. `Svensson.fit` and `NelsonSiegel.fit` both accepted `n_starts: int = 200` in their signatures, but the parameter was never used after the switch from multi-start local optimization to `scipy.optimize.differential_evolution`. Differential evolution does its own exploration internally.

**Fix**: Removed `n_starts` from both `fit` method signatures. No callers passed this argument (verified via grep across the entire codebase).

**Files changed**:
- `src/curveengine/curves/parametric.py` — removed `n_starts` from `Svensson.fit` (line 97) and `NelsonSiegel.fit` (line 197)

**Verification**:
- 11/11 tests pass
- ruff: clean
- mypy: clean

**Commit**: `refactor(curves): remove dead n_starts from fit signatures`
