# COTSMod.jl

COTSMod provides the COTS state transition and dispersal API used by ADRIA.
Population states and the Allee threshold are densities in COTS per hectare.
Recruitment and immigration channels are annual density fluxes in COTS per
hectare per year.

The legacy three-stage transition remains the default. Experimental recurrence
mechanisms are opt-in through `COTSParams`:

- `separated_recruitment` makes `larval_fecundity` produce a
  pre-pelagic density flux, applies source survival during connectivity, and
  applies `settlement_probability` once at the sink;
- `juvenile_storage` retains surviving juveniles when coral-dependent maturation
  is incomplete;
- `habitat_mediation` applies an open-habitat settlement gate;
- `low_cover_mortality` adds an experimental prey-cover mortality factor to
  juvenile and adult post-baseline survival. With pre-feeding fast+slow coral
  cover `C` (fraction), the factor is
  `1 - low_cover_strength * max(0, 1 - C/low_cover_threshold)`.
  `low_cover_threshold` is a cover fraction and `low_cover_strength` is in
  `[0, 1]`. Defaults are `false`, `0.10`, and `0.0`, preserving the legacy
  transition. Turning the switch off is the rollback; bounds are diagnostic,
  not evidence-calibrated;
- `size_weighted_fecundity` tracks large adults (>25 cm) as a subset of the
  existing total-adult `N[3]` (>=15 cm) density. The remainder are 15-25 cm
  small adults. Pre-pelagic production uses
  `B = exp(beta*(D_small-D_large))*(N[3]-L) + L` in place of `N[3]` before
  the existing Ricker and Allee terms; `beta` has units mm^-1 and the two
  diameters are diagnostic class representatives in mm. The Allee term still
  uses total adults. After ordinary adult survival, a bounded fraction of
  small adults grows into the large class, optionally mediated by coral
  cover. New adults enter the small class. The default `false` switch is the
  rollback and preserves the legacy population transition. This does not
  change per-adult grazing or survival, and the size-state assumptions are
  not validated historical observations;
- `preferred_prey_starvation` drives the food-survival factor from remembered
  fast (preferred) cover `F̄` instead of total cover `F+S`:
  `F̄_t = (1-α)F̄_{t-1} + αF_t`, with `α = 1-exp(-1/starvation_memory_years)`
  (or `α=1` when the memory is 0), and `f = 1` above
  `starvation_preferred_threshold`, otherwise `(1-p_tilde)+p_tilde(F̄/θ)^3`.
  The threshold is a cover fraction of habitable area. `F̄` initializes to the
  first observed `F`, and `reset_adult_ages!` clears it;
- `adult_senescence` tracks adult density by age `2..senescence_age` in
  `adult_ages` (COTS ha^-1, summing to `N[3]`). Ages below `senescence_age`
  survive at `(1-m3)f`; the plus group survives at
  `(1-senescence_mortality)f`, an annual fraction. Initial adults are placed on
  the stable age profile under background survival (`stable_adult_ages`). That
  placement is an assumption; call `reset_adult_ages!` after setting `N`
  externally. The transition throws if `adult_ages` do not sum to `N[3]`;
- `per_capita_consumption` replaces the functional response with a fixed area
  demand, `N[3]·consumption_m2_per_adult_year/10^4` (habitable-cover fraction
  per year), met from fast prey first, then slow, and capped by availability.
  `N[3]` is adults per hectare of site area. Pass the site's habitable
  fraction (`habitable_fraction` keyword on `cots_timestep!`, or a
  per-location vector on `cots_mortality!`/`apply_predation!`) to convert the
  grazed area into a habitable-cover fraction; the default of 1 assumes the
  whole site is habitable. The legacy functional response ignores it;
- `prey_switching` (requires `per_capita_consumption`) directs a share
  `ρ = 1/(1+exp(-switch_steepness·(F - switch_midpoint)))` of grazing demand at
  fast prey, with the remainder and any unmet fast demand falling on slow prey
  (MICE-style switching; `F` is a habitable-cover fraction). Experimental: as
  preregistered it sustained high-density COTS on massives;
- `diet_linked_starvation` (requires `prey_switching` and
  `preferred_prey_starvation`) sets food survival to
  `(1-p_tilde) + p_tilde·ρ(F̄)` using the switching logistic on remembered fast
  cover, as in MICE;
- `nonpreferred_feeding_efficiency` (default 1) scales the unmet preferred
  demand that per-capita grazers take from slow prey. With switching, both
  repairs protect coral but stabilise COTS at a persistent plateau; switching is
  retained for comparison only;
- `juvenile_predation` removes `Q·J/(J+Q)` age-1 juveniles per year
  (`juvenile_predation_capacity = Q`, COTS ha^-1 yr^-1; CoCoNet saturating form);
- `preferred_maturation` sets the age-1 maturation fraction to `sqrt(F̄)`
  (remembered fast cover); matured juveniles face the food factor, retained
  juveniles stay herbivorous and avoid coral starvation (Deaker et al. 2020);
- `allee_stretched` uses fertilisation `exp(-(allee_threshold/A)^allee_exponent)`
  (CoCoNet) instead of the Hill form, keeping the same scale;
- `preferred_condition` drives body condition (the fecundity multiplier) from
  fast cover, `min(1, F/condition_saturation)`, with EMA weight
  `1-exp(-1/condition_memory_years)`, instead of total cover and `tau_condition`;
- `cots_mechanism_diagnostics` reports the food memory, the realized food
  survival factor, senescent adults, plus-group deaths beyond background,
  and the preferred diet share;
- `disperse_larvae_by_reef!` applies source-by-sink reef connectivity without
  expanding every reef edge to a dense site matrix;
- `source_survival` weights transported recruits at their source;
- vector `apply_external_supply!` adds an absolute sink-specific recruit flux;
- `apply_external_larval_supply!` records potential external production,
  pelagic survivors, and settled external recruits separately;
- `cots_flow_diagnostics` separates local fecundity, background immigration,
  internal immigration, external supply, maturation, retained juveniles, and
  settlement habitat, and also exposes small/large adults, pre-transition
  effective breeders, and small-to-large growth for opt-in diagnostics.

These mechanisms must not be made default solely because they improve a fitted
objective. Follow `AGENTS.md` and ADRIA's calibration promotion gates.
The one-seed ADRIA low-cover screen did not pass its trough or peak gate;
`low_cover_mortality` remains experimental and off by default.
The 1991 Owen size-weighted one-seed screen also failed its qualitative
peak/trough gate despite reducing effective breeders and local/internal
recruitment. `size_weighted_fecundity` remains experimental and off by default;
see ADRIA's `sandbox/calibration/README.md` for the paired audit and coral plot.

### Post-outbreak collapse (`outbreak_collapse`, default off)

ReefMod 7.0 disease-style die-off. `outbreak_clock` counts consecutive annual steps with post-transition adults `N3 >= collapse_threshold_ha` (default 67.5 COTS/ha = 2.7 per 400 m²); a step below the threshold resets it. When the clock reaches `collapse_duration_years` (default 4), adults are rescaled, keeping age-class proportions, to `min(N3, collapse_background_ha)` (default 0.25 COTS/ha = 0.01 per 400 m²). With `collapse_juveniles = true`, `N2 + N3` together are rescaled. N1, larvae produced that step and external supply are unchanged. Diagnostics: `cots_mechanism_diagnostics` fields `outbreak_clock` and `collapse_removed_ha`. This is phenomenology compatible with a pathogen, not an identified epizootic.

### Post-grazing starvation (`starvation_post_grazing`, default off)

When on, the preferred-prey starvation memory is updated with the preferred cover left after the current adults' per-capita demand, `max(F - N3 * consumption_m2_per_adult_year / (1e4 * habitable_fraction), 0)`, instead of pre-grazing `F`. It requires `per_capita_consumption` and `preferred_prey_starvation` without `prey_switching`. Body condition still uses `F`.

### Starvation grace (`starvation_grace_years`, default 0)

`low_food_years` counts consecutive annual steps with remembered preferred food at or below `starvation_preferred_threshold`; food above the threshold resets it. While `low_food_years <= starvation_grace_years`, the H1 food factor is 1, and the cubic response applies afterwards. 0 reproduces the immediate response. Body condition still follows food. Requires `preferred_prey_starvation` without `diet_linked_starvation`. Diagnostic: `cots_mechanism_diagnostics(...).low_food_years`.

### Forced grazing (`forced_grazing!`)

`forced_grazing!(C_cover_t, adults_ha, prey_map; habitable_fraction, consumption_m2_per_adult_year=10.0)` applies per-capita, preferred-first grazing by a prescribed adult density (COTS per ha of site area, one value per location), without population dynamics. It returns the preferred and non-preferred cover consumed. Its arithmetic equals the `per_capita_consumption` path of `cots_mortality!` (no prey switching, full non-preferred efficiency); a unit test checks this. ADRIA uses it to calibrate coral against observed COTS.

