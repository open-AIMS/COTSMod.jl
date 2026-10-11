# COTSMod core-model change rules

- Preserve existing defaults. New ecological mechanisms remain opt-in until they
  pass ADRIA's calibration and validation promotion gates.
- Treat population states and thresholds as COTS per hectare and recruitment
  flows as COTS per hectare per year. Keep CPUE transformations outside this
  package. Never label pre-dispersal recruit production as adult density.
- For every transition change, document the ecological hypothesis, equation,
  units, parameter bounds, control treatment, and rollback switch.
- Add deterministic unit tests for state transitions, dispersal mass accounting,
  source/sink orientation, non-negativity, and diagnostic flux decomposition.
- Do not tune an independently supported biological threshold to compensate for
  a missing connectivity, boundary, habitat, or cohort mechanism.
- Stochastic APIs must accept a caller-owned RNG or an explicitly recorded seed.
- Promotion requires a bounded control-versus-treatment experiment, plausible
  peak timing and amplitude, held-out validation, sensitivity analysis around a
  viable region, and recorded ADRIA/COTSMod revisions and input hashes.
