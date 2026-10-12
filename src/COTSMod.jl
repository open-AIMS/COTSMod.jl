module COTSMod

using Random
using SparseArrays
using StaticArrays

export AbstractCOTSModel,
    COTSParams,
    COTSHuman,
    COTSState,
    COTSPreyMap,
    COTSPreyState,
    cots_timestep!,
    init_cots_populations,
    init_cots_from_spatial,
    cots_mortality!,
    disperse_cots_larvae!,
    disperse_cots_larvae_by_reef!,
    inject_upstream_pulse!,
    initialize_cots,
    apply_predation!,
    disperse_larvae!,
    disperse_larvae_by_reef!,
    apply_external_supply!,
    apply_external_larval_supply!,
    cots_flow_diagnostics,
    cots_hazard_diagnostics,
    cots_mechanism_diagnostics,
    stable_adult_ages,
    reset_adult_ages!

abstract type AbstractCOTSModel end

Base.@kwdef struct COTSParams
    a::Float64 = 1.5
    b::Float64 = 0.5
    IMM::Float64 = 0.002
    p_tilde::Float64 = 0.97
    C_max::Float64 = 0.8
    m1::Float64 = 0.4
    m2::Float64 = 0.2
    m3::Float64 = 0.1
    a_F::Float64 = 0.6
    a_S::Float64 = 0.15
    h::Float64 = 0.0
    eta_F::Float64 = 1.0
    eta_S::Float64 = 1.0
    eta_starve::Float64 = 2.0
    eta_imm::Float64 = 2.0
    imm_threshold::Float64 = 0.35
    fecundity_gate::Bool = false
    a_ricker::Float64 = 6.0
    b_ricker::Float64 = 0.1
    tau_condition::Float64 = 5.0
    allee_threshold::Float64 = 1.0
    # Experimental separation of pre-pelagic production, pelagic survival, and
    # settlement. The legacy effective-recruitment path remains the default.
    separated_recruitment::Bool = false
    larval_fecundity::Float64 = 6.0
    settlement_probability::Float64 = 1.0
    # Experimental cohort and habitat mechanisms. Defaults preserve legacy dynamics.
    juvenile_storage::Bool = false
    juvenile_maturation_min::Float64 = 0.05
    juvenile_maturation_max::Float64 = 1.0
    juvenile_maturation_cover::Float64 = 0.4
    habitat_mediation::Bool = false
    settlement_floor::Float64 = 0.05
    # Experimental additional mortality at low pre-feeding coral prey cover.
    # Off by default; cover is a dimensionless fraction, strength is in [0, 1].
    low_cover_mortality::Bool = false
    low_cover_threshold::Float64 = 0.10
    low_cover_strength::Float64 = 0.0
    # Opt-in two-adult-size breeder proxy. All densities remain COTS ha^-1.
    size_weighted_fecundity::Bool = false
    size_small_diameter_mm::Float64 = 200.0
    size_large_diameter_mm::Float64 = 300.0
    size_fecundity_slope_per_mm::Float64 = 0.0115
    size_initial_large_fraction::Float64 = 0.5
    size_growth_max::Float64 = 0.5
    size_growth_cover::Float64 = 0.4
    size_growth_food_mediated::Bool = false
    # Opt-in lagged adult-loss hypothesis. Burden/threshold/width are adult COTS ha^-1;
    # hazard_max is a dimensionless one-year cumulative hazard (annual step).
    lagged_adult_hazard::Bool = false
    hazard_food_coupled::Bool = false
    hazard_rho::Float64 = 0.5
    hazard_threshold_ha::Float64 = 1.5
    hazard_width_ha::Float64 = 0.5
    hazard_coral_threshold::Float64 = 0.15
    hazard_coral_width::Float64 = 0.03
    hazard_max::Float64 = 0.0
    # Opt-in starvation driven by preferred (fast) prey cover rather than total
    # cover. Threshold is a cover fraction of habitable area; memory is in years
    # (0 = contemporaneous). Survival keeps the legacy cubic shape and p_tilde.
    preferred_prey_starvation::Bool = false
    starvation_preferred_threshold::Float64 = 0.05
    starvation_memory_years::Float64 = 0.0
    # Opt-in adult age structure. Adults enter at age 2; ages 2..senescence_age-1
    # die at m3 and the plus group aged >= senescence_age (years) at
    # senescence_mortality (annual fraction). N[3] remains total adults ha^-1.
    adult_senescence::Bool = false
    senescence_age::Int = 6
    senescence_mortality::Float64 = 0.8
    # Opt-in per-capita area consumption, preferred prey first and capped by
    # available cover. Interprets N[3] as adult COTS per habitable ha; rate is
    # m^2 of coral per adult per year.
    per_capita_consumption::Bool = false
    consumption_m2_per_adult_year::Float64 = 10.0
    # Opt-in MICE-style prey switching for per-capita grazing: the share of demand
    # aimed at preferred prey is logistic in preferred cover (fraction of habitable
    # area, K=1): rho = 1/(1+exp(-steepness*(F - midpoint))).
    prey_switching::Bool = false
    switch_midpoint::Float64 = 0.10
    switch_steepness::Float64 = 70.0
    # Opt-in body condition driven by preferred cover rather than total cover:
    # signal = min(1, F/condition_saturation); EMA weight 1-exp(-1/memory years).
    preferred_condition::Bool = false
    condition_saturation::Float64 = 0.10
    condition_memory_years::Float64 = 1.0
    # Opt-in MICE-consistent starvation: food survival follows the same switching
    # logistic as the diet, evaluated on remembered preferred cover:
    # f = (1 - p_tilde) + p_tilde * rho(F_memory). Requires prey switching and
    # preferred-prey starvation; replaces the cubic below the H1 threshold.
    diet_linked_starvation::Bool = false
    # Fraction of unmet preferred demand that per-capita grazers take from
    # non-preferred (slow) prey. 1 preserves the legacy reallocation.
    nonpreferred_feeding_efficiency::Float64 = 1.0
    # Opt-in saturating predation on age-1 juveniles (CoCoNet benthic-invertebrate
    # form): predators remove Q*J/(J + Q) COTS ha^-1 yr^-1 from background
    # survivors J, with capacity Q = juvenile_predation_capacity (COTS ha^-1 yr^-1).
    juvenile_predation::Bool = false
    juvenile_predation_capacity::Float64 = 20.0
    # Opt-in preferred-prey maturation delay: fraction maturing = sqrt(F_memory)
    # (CoCoNet age-1 -> 2 transition). Matured juveniles face the food factor;
    # retained juveniles stay herbivorous and are not coral-starved.
    preferred_maturation::Bool = false
    # Opt-in gradual Allee fertilisation: exp(-(allee_threshold / A)^allee_exponent)
    # (CoCoNet form, scale unchanged) instead of A^2 / (allee_threshold^2 + A^2).
    allee_stretched::Bool = false
    allee_exponent::Float64 = 0.5
    # Opt-in post-outbreak collapse (ReefMod 7.0 disease form): after
    # collapse_duration_years consecutive years with post-transition adults >=
    # collapse_threshold_ha (COTS ha^-1), adults (and N2 when collapse_juveniles)
    # are rescaled to collapse_background_ha (COTS ha^-1). ReefMod: 2.7 and 0.01
    # per 400 m^2 = 67.5 and 0.25 COTS ha^-1.
    outbreak_collapse::Bool = false
    collapse_threshold_ha::Float64 = 67.5
    collapse_duration_years::Int = 4
    collapse_background_ha::Float64 = 0.25
    collapse_juveniles::Bool = false
end

struct COTSPreyMap
    fast_group_indices::Vector{Int}
    slow_group_indices::Vector{Int}
end

struct COTSPreyState
    fast_cover::Vector{Float64}
    slow_cover::Vector{Float64}
end

mutable struct COTSHuman <: AbstractCOTSModel
    N::MVector{3,Float64}
    body_condition::Float64
    params::COTSParams
    large_adults::Float64
    last_effective_breeders::Float64
    last_small_adult_growth::Float64
    last_local_fecundity::Float64
    last_background_immigration::Float64
    last_internal_immigration::Float64
    last_external_immigration::Float64
    last_maturation::Float64
    last_retained_juveniles::Float64
    last_settlement_gate::Float64
    last_local_retention::Float64
    last_pelagic_survivors::Float64
    last_settled_recruits::Float64
    last_external_potential::Float64
    last_external_pelagic::Float64
    burden_ha::Float64
    last_burden_ha::Float64
    last_hazard::Float64
    last_hazard_deaths_ha::Float64
    # Adult density by age (COTS ha^-1): ages 2..senescence_age, last is the
    # senescent plus group. Empty unless adult_senescence is enabled.
    adult_ages::Vector{Float64}
    # Remembered preferred prey cover; NaN until the first transition.
    preferred_food_memory::Float64
    last_food_survival::Float64
    last_senescent_adults::Float64
    last_senescent_deaths_ha::Float64
    # Share of grazing demand directed at preferred prey (1 unless prey_switching).
    last_preferred_diet_share::Float64
    last_juvenile_predation_ha::Float64
    last_maturation_fraction::Float64
    # Consecutive years with adults at or above collapse_threshold_ha.
    outbreak_clock::Int
    # Adults plus (when collapse_juveniles) N2 removed by a collapse (COTS ha^-1).
    last_collapse_removed_ha::Float64
end

COTSHuman(N::MVector{3,Float64}, body_condition::Float64, params::COTSParams) =
    COTSHuman(
        N, body_condition, params,
        params.size_weighted_fecundity ? N[3] * params.size_initial_large_fraction : 0.0,
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        params.adult_senescence ? stable_adult_ages(params, N[3]) : Float64[],
        NaN, 1.0, 0.0, 0.0,
        1.0, 0.0, 1.0,
        0, 0.0
    )

function _validate_senescence(p::COTSParams)
    p.senescence_age >= 3 ||
        throw(ArgumentError("senescence_age must be at least 3 years"))
    isfinite(p.senescence_mortality) && 0.0 < p.senescence_mortality <= 1.0 ||
        throw(ArgumentError("senescence_mortality must lie in (0, 1]"))
    return nothing
end

"""
Stable adult age distribution under background survival with food survival 1.

Returns densities for ages 2..senescence_age (the last is the plus group)
summing to `adults` (COTS ha^-1). This is an initialization assumption.
"""
function stable_adult_ages(p::COTSParams, adults::Float64)::Vector{Float64}
    _validate_senescence(p)
    n_classes = p.senescence_age - 1
    s_young = 1.0 - p.m3
    weights = ones(Float64, n_classes)
    for k in 2:(n_classes - 1)
        weights[k] = weights[k - 1] * s_young
    end
    weights[n_classes] = weights[n_classes - 1] * s_young / p.senescence_mortality
    return max(0.0, adults) .* weights ./ sum(weights)
end

"""Reset adult age classes, preferred-food memory and the outbreak clock after `N`
is set externally."""
function reset_adult_ages!(model::COTSHuman)::Nothing
    model.adult_ages = model.params.adult_senescence ?
        stable_adult_ages(model.params, model.N[3]) : Float64[]
    model.preferred_food_memory = NaN
    model.outbreak_clock = 0
    return nothing
end

const COTSState = Vector{COTSHuman}

function _uses_separated_recruitment(cots_models::COTSState)::Bool
    isempty(cots_models) && return false
    separated = first(cots_models).params.separated_recruitment
    all(model.params.separated_recruitment == separated for model in cots_models) ||
        throw(ArgumentError(string(:mixed_separated_recruitment_configuration)))
    return separated
end

function cots_timestep!(model::COTSHuman, F::Float64, S::Float64;
    habitable_fraction::Float64=1.0)
    p = model.params
    N = model.N

    if p.size_weighted_fecundity
        150.0 <= p.size_small_diameter_mm <= 250.0 ||
            throw(ArgumentError("size_small_diameter_mm must represent 15-25 cm adults"))
        p.size_large_diameter_mm > 250.0 ||
            throw(ArgumentError("size_large_diameter_mm must represent >25 cm adults"))
        p.size_fecundity_slope_per_mm >= 0.0 ||
            throw(ArgumentError("size_fecundity_slope_per_mm must be nonnegative"))
        0.0 <= p.size_initial_large_fraction <= 1.0 ||
            throw(ArgumentError("size_initial_large_fraction must lie in [0, 1]"))
        0.0 <= p.size_growth_max <= 1.0 ||
            throw(ArgumentError("size_growth_max must lie in [0, 1]"))
        p.size_growth_cover > 0.0 ||
            throw(ArgumentError("size_growth_cover must be positive"))
    end
    if p.lagged_adult_hazard
        isfinite(p.hazard_rho) && 0.0 <= p.hazard_rho < 1.0 ||
            throw(ArgumentError("hazard_rho must lie in [0, 1)"))
        isfinite(p.hazard_threshold_ha) && p.hazard_threshold_ha >= 0.0 ||
            throw(ArgumentError("hazard_threshold_ha must be nonnegative"))
        isfinite(p.hazard_width_ha) && p.hazard_width_ha > 0.0 ||
            throw(ArgumentError("hazard_width_ha must be positive"))
        isfinite(p.hazard_coral_threshold) &&
            0.0 <= p.hazard_coral_threshold <= 1.0 ||
            throw(ArgumentError("hazard_coral_threshold must lie in [0, 1]"))
        isfinite(p.hazard_coral_width) && p.hazard_coral_width > 0.0 ||
            throw(ArgumentError("hazard_coral_width must be positive"))
        isfinite(p.hazard_max) && p.hazard_max >= 0.0 ||
            throw(ArgumentError("hazard_max must be nonnegative"))
    end
    if p.preferred_prey_starvation
        isfinite(p.starvation_preferred_threshold) &&
            0.0 < p.starvation_preferred_threshold <= 1.0 ||
            throw(ArgumentError("starvation_preferred_threshold must lie in (0, 1]"))
        isfinite(p.starvation_memory_years) && p.starvation_memory_years >= 0.0 ||
            throw(ArgumentError("starvation_memory_years must be nonnegative"))
    end
    if p.adult_senescence
        _validate_senescence(p)
        length(model.adult_ages) == p.senescence_age - 1 ||
            throw(ArgumentError("adult_ages length must equal senescence_age - 1"))
        abs(sum(model.adult_ages) - N[3]) <= 1e-9 * max(1.0, N[3]) ||
            throw(ArgumentError("adult_ages do not sum to N[3]; call reset_adult_ages!"))
    end
    if p.per_capita_consumption
        isfinite(p.consumption_m2_per_adult_year) &&
            p.consumption_m2_per_adult_year >= 0.0 ||
            throw(ArgumentError("consumption_m2_per_adult_year must be nonnegative"))
        isfinite(habitable_fraction) && 0.0 < habitable_fraction <= 1.0 ||
            throw(ArgumentError("habitable_fraction must lie in (0, 1]"))
    end
    if p.prey_switching
        p.per_capita_consumption ||
            throw(ArgumentError("prey_switching requires per_capita_consumption"))
        isfinite(p.switch_midpoint) && 0.0 <= p.switch_midpoint <= 1.0 ||
            throw(ArgumentError("switch_midpoint must lie in [0, 1]"))
        isfinite(p.switch_steepness) && p.switch_steepness > 0.0 ||
            throw(ArgumentError("switch_steepness must be positive"))
    end
    if p.preferred_condition
        isfinite(p.condition_saturation) && 0.0 < p.condition_saturation <= 1.0 ||
            throw(ArgumentError("condition_saturation must lie in (0, 1]"))
        isfinite(p.condition_memory_years) && p.condition_memory_years > 0.0 ||
            throw(ArgumentError("condition_memory_years must be positive"))
    end
    if p.diet_linked_starvation
        p.prey_switching && p.preferred_prey_starvation ||
            throw(ArgumentError(
                "diet_linked_starvation requires prey_switching and preferred_prey_starvation"))
    end
    isfinite(p.nonpreferred_feeding_efficiency) &&
        0.0 <= p.nonpreferred_feeding_efficiency <= 1.0 ||
        throw(ArgumentError("nonpreferred_feeding_efficiency must lie in [0, 1]"))
    if p.juvenile_predation
        isfinite(p.juvenile_predation_capacity) && p.juvenile_predation_capacity > 0.0 ||
            throw(ArgumentError("juvenile_predation_capacity must be positive"))
    end
    if p.preferred_maturation
        !p.juvenile_storage ||
            throw(ArgumentError("preferred_maturation and juvenile_storage are exclusive"))
    end
    if p.allee_stretched
        isfinite(p.allee_exponent) && p.allee_exponent > 0.0 ||
            throw(ArgumentError("allee_exponent must be positive"))
    end
    if p.outbreak_collapse
        isfinite(p.collapse_threshold_ha) && p.collapse_threshold_ha > 0.0 ||
            throw(ArgumentError("collapse_threshold_ha must be positive"))
        p.collapse_duration_years >= 1 ||
            throw(ArgumentError("collapse_duration_years must be at least 1"))
        isfinite(p.collapse_background_ha) &&
            0.0 <= p.collapse_background_ha < p.collapse_threshold_ha ||
            throw(ArgumentError("collapse_background_ha must lie in [0, collapse_threshold_ha)"))
    end

    F = isnan(F) || isinf(F) ? 0.0 : max(0.0, F)
    S = isnan(S) || isinf(S) ? 0.0 : max(0.0, S)

    alpha, food_signal = if p.preferred_condition
        (1.0 - exp(-1.0 / p.condition_memory_years), min(1.0, F / p.condition_saturation))
    else
        (1.0 / p.tau_condition, min(1.0, (F + S) / (p.C_max * 0.5)))
    end
    model.body_condition = clamp((1.0 - alpha) * model.body_condition + alpha * food_signal, 0.0, 1.0)

    starve_threshold = p.C_max * 0.15
    f = if (F + S) > starve_threshold
        1.0
    else
        frac = (F + S) / starve_threshold
        (1.0 - p.p_tilde) + p.p_tilde * frac^3
    end
    if p.preferred_prey_starvation
        memory_weight = p.starvation_memory_years > 0.0 ?
            1.0 - exp(-1.0 / p.starvation_memory_years) : 1.0
        food_memory = isnan(model.preferred_food_memory) ? F :
            (1.0 - memory_weight) * model.preferred_food_memory + memory_weight * F
        model.preferred_food_memory = food_memory
        threshold = p.starvation_preferred_threshold
        f = if p.diet_linked_starvation
            share = 1.0 / (1.0 + exp(-p.switch_steepness * (food_memory - p.switch_midpoint)))
            (1.0 - p.p_tilde) + p.p_tilde * share
        else
            food_memory > threshold ? 1.0 :
                (1.0 - p.p_tilde) + p.p_tilde * (food_memory / threshold)^3
        end
    end
    if p.low_cover_mortality
        p.low_cover_threshold > 0.0 ||
            throw(ArgumentError("low_cover_threshold must be positive"))
        0.0 <= p.low_cover_strength <= 1.0 ||
            throw(ArgumentError("low_cover_strength must lie in [0, 1]"))
        stress = clamp(1.0 - (F + S) / p.low_cover_threshold, 0.0, 1.0)
        f *= 1.0 - p.low_cover_strength * stress
    end

    fecundity_scale = p.separated_recruitment ? p.larval_fecundity : p.a_ricker
    fecundity = fecundity_scale * model.body_condition^2
    large_adults = p.size_weighted_fecundity ? clamp(model.large_adults, 0.0, N[3]) : 0.0
    small_adults = N[3] - large_adults
    effective_breeders = if p.size_weighted_fecundity
        small_weight = exp(p.size_fecundity_slope_per_mm *
            (p.size_small_diameter_mm - p.size_large_diameter_mm))
        small_weight * small_adults + large_adults
    else
        N[3]
    end
    allee_effect = if p.allee_stretched
        N[3] > 0.0 ? exp(-(p.allee_threshold / N[3])^p.allee_exponent) : 0.0
    else
        (N[3]^2) / (p.allee_threshold^2 + N[3]^2)
    end
    potential_larvae = fecundity * effective_breeders *
        exp(-p.b_ricker * N[3]) * allee_effect

    s1 = 1.0 - p.m1
    s2 = 1.0 - p.m2
    s3 = 1.0 - p.m3
    imm_gate = min(1.0, ((F + S) / (p.imm_threshold * p.C_max))^p.eta_imm)
    settlement_gate = if p.habitat_mediation
        open_habitat = 1.0 - clamp((F + S) / p.C_max, 0.0, 1.0)
        clamp(p.settlement_floor + (1.0 - p.settlement_floor) * open_habitat, 0.0, 1.0)
    else
        1.0
    end
    settlement_probability = clamp(p.settlement_probability, 0.0, 1.0)
    R = p.separated_recruitment ?
        0.0 : potential_larvae * settlement_probability * settlement_gate
    imm = p.IMM * imm_gate * settlement_gate

    maturation_fraction = if p.juvenile_storage
        coral_signal = clamp((F + S) / p.juvenile_maturation_cover, 0.0, 1.0)
        clamp(
            p.juvenile_maturation_min +
            (p.juvenile_maturation_max - p.juvenile_maturation_min) * sqrt(coral_signal),
            0.0,
            1.0
        )
    else
        1.0
    end
    model.last_juvenile_predation_ha = 0.0
    if p.juvenile_predation || p.preferred_maturation
        juveniles = N[2] * s2
        if p.juvenile_predation
            capacity = p.juvenile_predation_capacity
            removed = juveniles > 0.0 ? capacity * juveniles / (juveniles + capacity) : 0.0
            model.last_juvenile_predation_ha = removed
            juveniles -= removed
        end
        if p.preferred_maturation
            memory = p.preferred_prey_starvation ? model.preferred_food_memory : F
            maturation_fraction = sqrt(clamp(memory, 0.0, 1.0))
            matured_juveniles = juveniles * maturation_fraction * f
            retained_juveniles = juveniles * (1.0 - maturation_fraction)
        else
            matured_juveniles = juveniles * f * maturation_fraction
            retained_juveniles = juveniles * f - matured_juveniles
        end
        surviving_juveniles = matured_juveniles + retained_juveniles
    else
        surviving_juveniles = N[2] * (s2 * f)
        matured_juveniles = surviving_juveniles * maturation_fraction
        retained_juveniles = surviving_juveniles - matured_juveniles
    end
    model.last_maturation_fraction = maturation_fraction

    model.last_food_survival = f
    model.last_senescent_adults = 0.0
    model.last_senescent_deaths_ha = 0.0
    adult_survivors = N[3] * (s3 * f)
    adult_survival_factor = s3 * f
    if p.adult_senescence
        ages = model.adult_ages
        n_classes = length(ages)
        s_old = (1.0 - p.senescence_mortality) * f
        senescent = ages[n_classes]
        model.last_senescent_adults = senescent
        # Deaths in the plus group beyond what background mortality would remove.
        model.last_senescent_deaths_ha = max(0.0, senescent * (s3 * f - s_old))
        ages[n_classes] = senescent * s_old + ages[n_classes - 1] * s3 * f
        for k in (n_classes - 1):-1:2
            ages[k] = ages[k - 1] * s3 * f
        end
        ages[1] = 0.0
        adult_survivors = sum(ages)
        adult_survival_factor = N[3] > 0.0 ? adult_survivors / N[3] : s3 * f
    end
    model.last_burden_ha = 0.0
    model.last_hazard = 0.0
    model.last_hazard_deaths_ha = 0.0
    if p.lagged_adult_hazard
        burden = model.burden_ha
        excess = max(burden - p.hazard_threshold_ha, 0.0)
        density_trigger = excess / (p.hazard_width_ha + excess)
        coral_trigger = p.hazard_food_coupled ?
            1.0 / (1.0 + exp(-(p.hazard_coral_threshold - (F + S)) /
                p.hazard_coral_width)) : 1.0
        hazard = p.hazard_max * density_trigger * coral_trigger
        hazard_survival = exp(-hazard)
        model.last_burden_ha = burden
        model.last_hazard = hazard
        model.last_hazard_deaths_ha = adult_survivors * (1.0 - hazard_survival)
        adult_survivors *= hazard_survival
        adult_survival_factor *= hazard_survival
        p.adult_senescence && (model.adult_ages .*= hazard_survival)
        model.burden_ha = p.hazard_rho * burden + (1.0 - p.hazard_rho) * N[3]
    end

    N_next = MVector{3,Float64}(0.0, 0.0, 0.0)
    N_next[3] = matured_juveniles + adult_survivors
    N_next[2] = N[1] * s1 + retained_juveniles
    N_next[1] = R + imm

    for i in 1:3
        val = N_next[i]
        N_next[i] = isnan(val) || isinf(val) ? 0.0 : max(0.0, val)
    end
    if p.adult_senescence
        ages = model.adult_ages
        ages[1] = max(0.0, matured_juveniles)
        all(isfinite, ages) && all(>=(0.0), ages) ||
            throw(ArgumentError("adult age classes became nonfinite or negative"))
        N_next[3] = sum(ages)
    end
    model.last_collapse_removed_ha = 0.0
    if p.outbreak_collapse
        model.outbreak_clock = N_next[3] >= p.collapse_threshold_ha ?
            model.outbreak_clock + 1 : 0
        if model.outbreak_clock >= p.collapse_duration_years
            # Rescale the dieback classes to the background density, keeping
            # their proportions (ReefMod f_runmodel reset).
            dieback = p.collapse_juveniles ? N_next[2] + N_next[3] : N_next[3]
            scale = dieback > 0.0 ? min(dieback, p.collapse_background_ha) / dieback : 0.0
            model.last_collapse_removed_ha = dieback * (1.0 - scale)
            if p.adult_senescence
                model.adult_ages .*= scale
                N_next[3] = sum(model.adult_ages)
            else
                N_next[3] *= scale
            end
            p.collapse_juveniles && (N_next[2] *= scale)
            model.outbreak_clock = 0
        end
    end
    model.N .= N_next
    if p.size_weighted_fecundity
        growth = p.size_growth_max * (p.size_growth_food_mediated ?
            clamp((F + S) / p.size_growth_cover, 0.0, 1.0) : 1.0)
        model.last_small_adult_growth = growth * small_adults * adult_survival_factor
        model.large_adults = min(N_next[3],
            large_adults * adult_survival_factor + model.last_small_adult_growth)
    else
        model.last_small_adult_growth = 0.0
        model.large_adults = 0.0
    end
    model.last_effective_breeders = effective_breeders
    model.last_local_fecundity = p.separated_recruitment ? potential_larvae : R
    model.last_background_immigration = imm
    model.last_internal_immigration = 0.0
    model.last_external_immigration = 0.0
    model.last_maturation = matured_juveniles
    model.last_retained_juveniles = retained_juveniles
    model.last_settlement_gate = settlement_gate
    model.last_local_retention = p.separated_recruitment ? 0.0 : R
    model.last_pelagic_survivors = p.separated_recruitment ? 0.0 : potential_larvae
    model.last_settled_recruits = R
    model.last_external_potential = 0.0
    model.last_external_pelagic = 0.0

    # Grazing uses post-transition adults (`N` aliases `model.N`), as in the
    # legacy response below.
    if p.per_capita_consumption
        # Adults are per ha of site area; prey cover is a fraction of habitable
        # area, so the grazed area is divided by the site's habitable area.
        demand = N[3] * p.consumption_m2_per_adult_year /
            (10_000.0 * habitable_fraction)
        share = p.prey_switching ?
            1.0 / (1.0 + exp(-p.switch_steepness * (F - p.switch_midpoint))) : 1.0
        model.last_preferred_diet_share = share
        Cons_F = min(share * demand, F)
        Cons_S = min(p.nonpreferred_feeding_efficiency * (demand - Cons_F), S)
        return Cons_F, Cons_S
    end
    F_pow = F^p.eta_F
    S_pow = S^p.eta_S
    denom = 1.0 + p.h * (p.a_F * F_pow + p.a_S * S_pow)
    return N[3] * (p.a_F * F_pow) / denom, N[3] * (p.a_S * S_pow) / denom
end

function disperse_cots_larvae!(
    cots_models::COTSState,
    conn::SparseMatrixCSC{Float64,Int64};
    immigration_scalar::Float64=1.0,
    source_survival=nothing
)::Nothing
    n_locs = length(cots_models)
    isnothing(source_survival) || length(source_survival) == n_locs ||
        throw(DimensionMismatch(string(:source_survival_length_must_match_locations)))
    separated = _uses_separated_recruitment(cots_models)
    local_recruits = [cots_models[loc].N[1] for loc in 1:n_locs]
    source_production = separated ?
        [model.last_local_fecundity for model in cots_models] : local_recruits

    for sink in 1:n_locs
        local_pelagic = 0.0
        internal_pelagic = 0.0
        for k in SparseArrays.nzrange(conn, sink)
            src = SparseArrays.rowvals(conn)[k]
            !separated && src == sink && continue
            p = SparseArrays.nonzeros(conn)[k]
            survival = isnothing(source_survival) ? 1.0 : max(0.0, Float64(source_survival[src]))
            val = source_production[src] * survival * p * immigration_scalar
            val = isnan(val) || isinf(val) ? 0.0 : val
            if separated && src == sink
                local_pelagic += val
            else
                internal_pelagic += val
            end
        end
        if separated
            settlement = clamp(
                cots_models[sink].params.settlement_probability, 0.0, 1.0
            ) * cots_models[sink].last_settlement_gate
            local_retention = local_pelagic * settlement
            immigration = internal_pelagic * settlement
            cots_models[sink].N[1] = max(
                0.0, local_recruits[sink] + local_retention + immigration
            )
            cots_models[sink].last_local_retention = max(0.0, local_retention)
            cots_models[sink].last_internal_immigration = max(0.0, immigration)
            cots_models[sink].last_pelagic_survivors = max(
                0.0, local_pelagic + internal_pelagic
            )
            cots_models[sink].last_settled_recruits += max(
                0.0, local_retention + immigration
            )
            continue
        end
        immigration = internal_pelagic
        immigration *= cots_models[sink].last_settlement_gate
        total = local_recruits[sink] + immigration
        cots_models[sink].N[1] = isnan(total) || isinf(total) ? 0.0 : max(0.0, total)
        cots_models[sink].last_internal_immigration = max(0.0, immigration)
        cots_models[sink].last_pelagic_survivors += max(0.0, internal_pelagic)
        cots_models[sink].last_settled_recruits += max(0.0, immigration)
    end
    return nothing
end

# Disperse site-level recruits using a reef-to-reef source-by-sink matrix.
# Production is aggregated across source sites and divided uniformly among sites
# in the sink reef. Exact site self-connections are excluded; other sites on the
# same reef retain the reef diagonal.
function disperse_cots_larvae_by_reef!(
    cots_models::COTSState,
    reef_conn::AbstractMatrix,
    site_to_reef::AbstractVector{<:Integer};
    immigration_scalar::Float64=1.0,
    source_survival=nothing,
    reef_areas=nothing
)::Nothing
    n_locs = length(cots_models)
    length(site_to_reef) == n_locs ||
        throw(DimensionMismatch(string(:site_to_reef_length_must_match_locations)))
    n_reefs = size(reef_conn, 1)
    size(reef_conn, 2) == n_reefs ||
        throw(DimensionMismatch(string(:reef_connectivity_must_be_square)))
    all((1 .<= site_to_reef) .& (site_to_reef .<= n_reefs)) ||
        throw(ArgumentError(string(:invalid_site_to_reef_index)))
    isnothing(source_survival) || length(source_survival) == n_reefs ||
        throw(DimensionMismatch(string(:source_survival_length_must_match_reefs)))
    isnothing(reef_areas) || length(reef_areas) == n_reefs ||
        throw(DimensionMismatch(string(:reef_areas_length_must_match_reefs)))
    separated = _uses_separated_recruitment(cots_models)
    separated && isnothing(reef_areas) &&
        throw(ArgumentError(string(:separated_recruitment_requires_reef_areas)))
    areas = isnothing(reef_areas) ? ones(Float64, n_reefs) : Float64.(reef_areas)
    all(isfinite, areas) && all(areas .> 0.0) ||
        throw(ArgumentError(string(:reef_areas_must_be_positive_finite)))

    local_recruits = [model.N[1] for model in cots_models]
    sites_per_reef = zeros(Int, n_reefs)
    reef_sources = zeros(Float64, n_reefs)
    for loc in 1:n_locs
        reef = Int(site_to_reef[loc])
        survival = isnothing(source_survival) ? 1.0 : max(0.0, Float64(source_survival[reef]))
        sites_per_reef[reef] += 1
        production = separated ? cots_models[loc].last_local_fecundity : local_recruits[loc]
        reef_sources[reef] += production * survival
    end
    all(sites_per_reef .> 0) || throw(ArgumentError(string(:reef_without_sites)))

    for sink in 1:n_locs
        sink_reef = Int(site_to_reef[sink])
        local_pelagic = 0.0
        internal_pelagic = 0.0
        for source_reef in 1:n_reefs
            source_total = reef_sources[source_reef]
            if !separated && source_reef == sink_reef
                survival = isnothing(source_survival) ? 1.0 : max(0.0, Float64(source_survival[source_reef]))
                source_total -= local_recruits[sink] * survival
            end
            val = if separated
                source_density = source_total / sites_per_reef[source_reef]
                source_density * areas[source_reef] / areas[sink_reef] *
                    Float64(reef_conn[source_reef, sink_reef]) * immigration_scalar
            else
                source_total * Float64(reef_conn[source_reef, sink_reef]) /
                    sites_per_reef[sink_reef] * immigration_scalar
            end
            val = isnan(val) || isinf(val) ? 0.0 : val
            if separated && source_reef == sink_reef
                local_pelagic += val
            else
                internal_pelagic += val
            end
        end
        if separated
            settlement = clamp(
                cots_models[sink].params.settlement_probability, 0.0, 1.0
            ) * cots_models[sink].last_settlement_gate
            local_retention = local_pelagic * settlement
            immigration = internal_pelagic * settlement
            cots_models[sink].N[1] = max(
                0.0, local_recruits[sink] + local_retention + immigration
            )
            cots_models[sink].last_local_retention = max(0.0, local_retention)
            cots_models[sink].last_internal_immigration = max(0.0, immigration)
            cots_models[sink].last_pelagic_survivors = max(
                0.0, local_pelagic + internal_pelagic
            )
            cots_models[sink].last_settled_recruits += max(
                0.0, local_retention + immigration
            )
            continue
        end
        immigration = internal_pelagic
        immigration *= cots_models[sink].last_settlement_gate
        cots_models[sink].N[1] = max(0.0, local_recruits[sink] + immigration)
        cots_models[sink].last_internal_immigration = max(0.0, immigration)
        cots_models[sink].last_pelagic_survivors += max(0.0, internal_pelagic)
        cots_models[sink].last_settled_recruits += max(0.0, immigration)
    end
    return nothing
end

function init_cots_populations(
    n_locs::Int,
    params::COTSParams;
    outbreak_fraction::Float64=0.25,
    seed_locs::Union{Set{Int},Nothing}=nothing,
    init_density::Float64=0.1,
    rng=Random.GLOBAL_RNG
)::COTSState
    seeded_locs = if !isnothing(seed_locs)
        seed_locs
    else
        n_seeded = max(1, round(Int, n_locs * outbreak_fraction))
        Set(randperm(rng, n_locs)[1:n_seeded])
    end

    return [
        COTSHuman(
            loc in seeded_locs ? MVector{3,Float64}(0.02, 0.02, init_density) : MVector{3,Float64}(0.0, 0.0, 0.0),
            0.8,
            params
        ) for loc in 1:n_locs
    ]
end

function init_cots_from_spatial(
    n_locs::Int,
    params::COTSParams,
    probabilities::AbstractVector{<:Real};
    initial_multiplier::Float64=1.5
)::COTSState
    @assert length(probabilities) == n_locs "Probability vector length must match n_locs"
    models = Vector{COTSHuman}(undef, n_locs)
    for loc in 1:n_locs
        p = Float64(probabilities[loc])
        N_init = if p > 0.1
            scale = min(p / 0.5, 1.0)
            MVector{3,Float64}(0.6 * scale * initial_multiplier, 0.3 * scale * initial_multiplier, 0.1 * scale * initial_multiplier)
        else
            MVector{3,Float64}(0.0, 0.0, 0.0)
        end
        models[loc] = COTSHuman(N_init, 0.8, params)
    end
    return models
end

# `habitable_fraction` (per location, habitable area / site area) is used only by
# per-capita consumption. Locations without habitable area hold no prey cover and
# are passed a fraction of one.
function cots_mortality!(
    C_cover_t::AbstractArray{Float64,3},
    cots_models::COTSState,
    prey_map::COTSPreyMap;
    habitable_fraction=nothing
)::Nothing
    _, n_sizes, n_locs = size(C_cover_t)
    isnothing(habitable_fraction) || length(habitable_fraction) == n_locs ||
        throw(DimensionMismatch(string(:habitable_fraction_length_must_match_locations)))
    @inbounds for loc in 1:n_locs
        F_cover = 0.0
        for g in prey_map.fast_group_indices, s in 1:n_sizes
            F_cover += C_cover_t[g, s, loc]
        end
        S_cover = 0.0
        for g in prey_map.slow_group_indices, s in 1:n_sizes
            S_cover += C_cover_t[g, s, loc]
        end

        fraction = isnothing(habitable_fraction) ? 1.0 : Float64(habitable_fraction[loc])
        fraction = fraction > 0.0 ? min(fraction, 1.0) : 1.0
        Cons_F, Cons_S = cots_timestep!(cots_models[loc], F_cover, S_cover;
            habitable_fraction=fraction)
        Cons_F = min(Cons_F, F_cover)
        Cons_S = min(Cons_S, S_cover)
        surv_F = F_cover > 0.0 ? max(0.0, 1.0 - Cons_F / F_cover) : 1.0
        surv_S = S_cover > 0.0 ? max(0.0, 1.0 - Cons_S / S_cover) : 1.0
        surv_F = isnan(surv_F) || isinf(surv_F) ? 1.0 : clamp(surv_F, 0.0, 1.0)
        surv_S = isnan(surv_S) || isinf(surv_S) ? 1.0 : clamp(surv_S, 0.0, 1.0)

        for g in prey_map.fast_group_indices, s in 1:n_sizes
            C_cover_t[g, s, loc] *= surv_F
        end
        for g in prey_map.slow_group_indices, s in 1:n_sizes
            C_cover_t[g, s, loc] *= surv_S
        end
    end
    clamp!(C_cover_t, 0.0, 1.0)
    return nothing
end

function inject_upstream_pulse!(cots_models::COTSState, pulse_locs::Set{Int}; pulse_val::Float64=2.0)::Nothing
    for loc in pulse_locs
        if loc <= length(cots_models)
            added = max(0.0, pulse_val) * cots_models[loc].last_settlement_gate
            cots_models[loc].N[1] += added
            cots_models[loc].last_external_immigration += added
            cots_models[loc].last_external_potential += added
            cots_models[loc].last_external_pelagic += added
            cots_models[loc].last_settled_recruits += added
        end
    end
    return nothing
end

function inject_upstream_pulse!(cots_models::COTSState, pulse_locs::Set{Int}, pulse_vals::AbstractVector{Float64})::Nothing
    for loc in pulse_locs
        if loc <= length(cots_models) && loc <= length(pulse_vals)
            added = max(0.0, pulse_vals[loc]) * cots_models[loc].last_settlement_gate
            cots_models[loc].N[1] += added
            cots_models[loc].last_external_immigration += added
            cots_models[loc].last_external_potential += added
            cots_models[loc].last_external_pelagic += added
            cots_models[loc].last_settled_recruits += added
        end
    end
    return nothing
end

function inject_upstream_pulse!(cots_models::COTSState, pulse_vals::AbstractVector{<:Real})::Nothing
    length(pulse_vals) == length(cots_models) ||
        throw(DimensionMismatch(string(:pulse_values_length_must_match_locations)))
    for loc in eachindex(cots_models)
        added = max(0.0, Float64(pulse_vals[loc])) * cots_models[loc].last_settlement_gate
        cots_models[loc].N[1] += added
        cots_models[loc].last_external_immigration += added
        cots_models[loc].last_external_potential += added
        cots_models[loc].last_external_pelagic += added
        cots_models[loc].last_settled_recruits += added
    end
    return nothing
end

"""
Add evidence-derived external larvae after connectivity but before settlement.

`potential_supply` and `pelagic_supply` are sink-specific density fluxes in
COTS ha^-1 yr^-1 before and after pelagic survival, respectively. Settlement is
applied exactly once using the sink's explicit settlement probability and
optional habitat gate.
"""
function apply_external_larval_supply!(
    cots_models::COTSState,
    potential_supply::AbstractVector{<:Real},
    pelagic_supply::AbstractVector{<:Real}
)::Nothing
    n_locs = length(cots_models)
    length(potential_supply) == n_locs ||
        throw(DimensionMismatch(string(:potential_supply_length_must_match_locations)))
    length(pelagic_supply) == n_locs ||
        throw(DimensionMismatch(string(:pelagic_supply_length_must_match_locations)))
    for loc in eachindex(cots_models)
        potential = max(0.0, Float64(potential_supply[loc]))
        pelagic = max(0.0, Float64(pelagic_supply[loc]))
        settlement = clamp(
            cots_models[loc].params.settlement_probability, 0.0, 1.0
        ) * cots_models[loc].last_settlement_gate
        added = pelagic * settlement
        cots_models[loc].N[1] += added
        cots_models[loc].last_external_potential += potential
        cots_models[loc].last_external_pelagic += pelagic
        cots_models[loc].last_external_immigration += added
        cots_models[loc].last_settled_recruits += added
    end
    return nothing
end

function initialize_cots(
    n_locs::Int,
    params::COTSParams;
    enabled::Bool=true,
    spatial_initial_density::Union{AbstractVector{<:Real},Nothing}=nothing,
    seed_locs::Union{Set{Int},Nothing}=nothing,
    init_density::Float64=0.1,
    initial_multiplier::Float64=1.5,
    rng=Random.GLOBAL_RNG
)::COTSState
    !enabled && return [COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, params) for _ in 1:n_locs]
    !isnothing(spatial_initial_density) && return init_cots_from_spatial(n_locs, params, spatial_initial_density; initial_multiplier=initial_multiplier)
    return init_cots_populations(n_locs, params; seed_locs=seed_locs, init_density=init_density, rng=rng)
end

apply_predation!(coral_cover::AbstractArray{Float64,3}, cots_state::COTSState, prey_map::COTSPreyMap; habitable_fraction=nothing)::Nothing =
    cots_mortality!(coral_cover, cots_state, prey_map; habitable_fraction=habitable_fraction)

disperse_larvae!(cots_state::COTSState, conn::SparseMatrixCSC{Float64,Int64}; scalar::Float64=1.0, source_survival=nothing)::Nothing =
    disperse_cots_larvae!(cots_state, conn; immigration_scalar=scalar, source_survival=source_survival)

disperse_larvae_by_reef!(cots_state::COTSState, reef_conn::AbstractMatrix, site_to_reef::AbstractVector{<:Integer}; scalar::Float64=1.0, source_survival=nothing, reef_areas=nothing)::Nothing =
    disperse_cots_larvae_by_reef!(cots_state, reef_conn, site_to_reef; immigration_scalar=scalar, source_survival=source_survival, reef_areas=reef_areas)

apply_external_supply!(cots_state::COTSState, pulse_locs::Set{Int}; pulse_val::Float64=2.0)::Nothing =
    inject_upstream_pulse!(cots_state, pulse_locs; pulse_val=pulse_val)

apply_external_supply!(cots_state::COTSState, pulse_locs::Set{Int}, pulse_vals::AbstractVector{Float64})::Nothing =
    inject_upstream_pulse!(cots_state, pulse_locs, pulse_vals)

apply_external_supply!(cots_state::COTSState, pulse_vals::AbstractVector{<:Real})::Nothing =
    inject_upstream_pulse!(cots_state, pulse_vals)

function cots_flow_diagnostics(cots_state::COTSState)
    return (
        local_fecundity=[m.last_local_fecundity for m in cots_state],
        background_immigration=[m.last_background_immigration for m in cots_state],
        internal_immigration=[m.last_internal_immigration for m in cots_state],
        external_immigration=[m.last_external_immigration for m in cots_state],
        maturation=[m.last_maturation for m in cots_state],
        retained_juveniles=[m.last_retained_juveniles for m in cots_state],
        settlement_gate=[m.last_settlement_gate for m in cots_state],
        local_retention=[m.last_local_retention for m in cots_state],
        pelagic_survivors=[m.last_pelagic_survivors for m in cots_state],
        settled_recruits=[m.last_settled_recruits for m in cots_state],
        external_potential=[m.last_external_potential for m in cots_state],
        external_pelagic=[m.last_external_pelagic for m in cots_state],
        small_adults=[m.N[3] - m.large_adults for m in cots_state],
        large_adults=[m.large_adults for m in cots_state],
        effective_breeders=[m.last_effective_breeders for m in cots_state],
        small_adult_growth=[m.last_small_adult_growth for m in cots_state]
    )
end

function cots_hazard_diagnostics(cots_state::COTSState)
    return (
        burden_ha=[m.last_burden_ha for m in cots_state],
        hazard=[m.last_hazard for m in cots_state],
        adult_deaths_ha=[m.last_hazard_deaths_ha for m in cots_state]
    )
end

"""
Food and senescence diagnostics: remembered preferred cover (habitable-area
fraction; NaN when the switch is off), realized food survival factor applied to
N2 and adults, senescent plus-group adults before transition (COTS ha^-1), and
plus-group deaths beyond background mortality (COTS ha^-1 yr^-1), and the share
of per-capita grazing demand aimed at preferred prey (1 without prey switching),
juvenile predation removals (COTS ha^-1 yr^-1), the juvenile maturation fraction,
the outbreak clock (consecutive years above the collapse threshold) and COTS
removed by an outbreak collapse (COTS ha^-1).
"""
function cots_mechanism_diagnostics(cots_state::COTSState)
    return (
        preferred_food_memory=[m.preferred_food_memory for m in cots_state],
        food_survival=[m.last_food_survival for m in cots_state],
        senescent_adults=[m.last_senescent_adults for m in cots_state],
        senescent_deaths_ha=[m.last_senescent_deaths_ha for m in cots_state],
        preferred_diet_share=[m.last_preferred_diet_share for m in cots_state],
        juvenile_predation_ha=[m.last_juvenile_predation_ha for m in cots_state],
        maturation_fraction=[m.last_maturation_fraction for m in cots_state],
        outbreak_clock=[Float64(m.outbreak_clock) for m in cots_state],
        collapse_removed_ha=[m.last_collapse_removed_ha for m in cots_state]
    )
end

end
