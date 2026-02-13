import Mathlib.NumberTheory.Real.Irrational

theorem irrational_sqrt_two_smoke : Irrational (Real.sqrt 2) := by
  simpa using irrational_sqrt_two
