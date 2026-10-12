using Test
using Random
using SparseArrays
using StaticArrays
using COTSMod

const MVector = StaticArrays.MVector

test_cots_params() = COTSParams(
    a = 1.5,
    b = 0.5,
    IMM = 0.002,
    p_tilde = 0.97,
    C_max = 0.8,
    m1 = 0.4,
    m2 = 0.2,
    m3 = 0.1,
    a_F = 0.6,
    a_S = 0.15,
    h = 0.0,
    eta_F = 1.0,
    eta_S = 1.0,
    eta_starve = 2.0,
    eta_imm = 2.0,
    imm_threshold = 0.35,
    fecundity_gate = false,
    a_ricker = 6.0,
    b_ricker = 0.1,
    tau_condition = 5.0,
    allee_threshold = 1.0
)

@testset "COTSMod" begin
    @testset "COTSHuman construction" begin
        p = test_cots_params()
        model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.5), 0.8, p)
        @test model.N[1] == 0.0
        @test model.N[2] == 0.0
        @test model.N[3] == 0.5
        @test model.body_condition == 0.8
        @test model.params.p_tilde == 0.97
    end

    @testset "cots_timestep! basic dynamics" begin
        p = test_cots_params()
        model = COTSHuman(MVector{3,Float64}(0.1, 0.2, 1.0), 0.8, p)
        Cons_F, Cons_S = cots_timestep!(model, 0.3, 0.1)
        @test all(model.N .>= 0.0)
        @test all(isfinite.(model.N))
        @test 0.0 <= Cons_F <= 0.3
        @test 0.0 <= Cons_S <= 0.1
    end

    @testset "cots_timestep! starvation with zero coral" begin
        p = test_cots_params()
        model = COTSHuman(MVector{3,Float64}(0.0, 0.5, 2.0), 0.8, p)
        cots_timestep!(model, 0.0, 0.0)
        @test model.N[3] < 2.0
        @test all(model.N .>= 0.0)
    end

    @testset "opt-in size weighting conserves adults and uses the supported Allee density" begin
        initial = MVector{3,Float64}(0.0, 0.0, 3.0)
        control = COTSHuman(copy(initial), 1.0, COTSParams(
            IMM=0.0, separated_recruitment=true, larval_fecundity=8.0,
            allee_threshold=3.0))
        off = COTSHuman(copy(initial), 1.0, COTSParams(
            IMM=0.0, separated_recruitment=true, larval_fecundity=8.0,
            allee_threshold=3.0, size_weighted_fecundity=false,
            size_initial_large_fraction=0.0))
        weighted = COTSHuman(copy(initial), 1.0, COTSParams(
            IMM=0.0, separated_recruitment=true, larval_fecundity=8.0,
            allee_threshold=3.0, size_weighted_fecundity=true,
            size_initial_large_fraction=0.5, size_growth_max=0.5))
        for model in (control, off, weighted)
            cots_timestep!(model, 0.4, 0.1)
        end
        @test control.N == off.N
        @test control.last_local_fecundity == off.last_local_fecundity
        @test weighted.N[3] == control.N[3]
        @test weighted.large_adults <= weighted.N[3]
        @test weighted.last_effective_breeders ≈ 1.5 * (1 + exp(-1.15))
        @test weighted.last_local_fecundity ≈
            8.0 * weighted.body_condition^2 * weighted.last_effective_breeders *
            exp(-0.3) * 0.5
        @test weighted.last_local_fecundity < control.last_local_fecundity
        @test weighted.last_small_adult_growth ≈ 0.5 * 1.5 * (1 - weighted.params.m3)
        @test weighted.large_adults ≈ 1.5 * (1 - weighted.params.m3) +
            weighted.last_small_adult_growth
        @test weighted.N[3] - weighted.large_adults >= 0.0

        fed = COTSHuman(copy(initial), 1.0, COTSParams(
            size_weighted_fecundity=true, size_growth_food_mediated=true,
            size_initial_large_fraction=0.5, size_growth_max=0.5, IMM=0.0))
        poor = COTSHuman(copy(initial), 1.0, fed.params)
        cots_timestep!(fed, 0.4, 0.1)
        cots_timestep!(poor, 0.01, 0.0)
        @test poor.last_small_adult_growth < fed.last_small_adult_growth
        @test poor.large_adults <= poor.N[3]
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 1.0,
            COTSParams(size_weighted_fecundity=true, size_growth_max=1.1)), 0.4, 0.1)
    end

    @testset "opt-in low-cover mortality preserves legacy control" begin
        initial = MVector{3,Float64}(0.0, 1.0, 1.0)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0))
        off = COTSHuman(copy(initial), 0.8, COTSParams(
            IMM=0.0, low_cover_mortality=false,
            low_cover_threshold=0.10, low_cover_strength=1.0))
        treated = COTSHuman(copy(initial), 0.8, COTSParams(
            IMM=0.0, low_cover_mortality=true,
            low_cover_threshold=0.10, low_cover_strength=0.5))
        cots_timestep!(legacy, 0.04, 0.02)
        cots_timestep!(off, 0.04, 0.02)
        cots_timestep!(treated, 0.04, 0.02)
        @test legacy.N == off.N
        @test legacy.body_condition == off.body_condition
        @test treated.N[3] < legacy.N[3]
        @test treated.last_maturation < legacy.last_maturation
        @test treated.N[1] == legacy.N[1]
        @test all(treated.N .>= 0.0)

        fed = COTSHuman(copy(initial), 0.8, COTSParams(
            IMM=0.0, low_cover_mortality=true,
            low_cover_threshold=0.10, low_cover_strength=1.0))
        fed_control = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0))
        cots_timestep!(fed, 0.20, 0.10)
        cots_timestep!(fed_control, 0.20, 0.10)
        @test fed.N == fed_control.N

        zero_food = COTSHuman(copy(initial), 0.8, COTSParams(
            IMM=0.0, low_cover_mortality=true,
            low_cover_threshold=0.10, low_cover_strength=1.0))
        cots_timestep!(zero_food, 0.0, 0.0)
        @test zero_food.N[3] == 0.0
        @test zero_food.last_maturation == 0.0

        invalid = COTSHuman(copy(initial), 0.8, COTSParams(
            low_cover_mortality=true, low_cover_threshold=0.0))
        @test_throws ArgumentError cots_timestep!(invalid, 0.04, 0.02)
    end

    @testset "lagged adult hazard is opt-in, delayed, and mass-accounted" begin
        initial = MVector{3,Float64}(0.2, 1.0, 3.0)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0))
        off = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0,
            lagged_adult_hazard=false, hazard_max=1.5))
        zero = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0,
            lagged_adult_hazard=true, hazard_max=0.0))
        @test zero.burden_ha == 0.0
        for _ in 1:3
            cots_timestep!(legacy, 0.2, 0.1)
            cots_timestep!(off, 0.2, 0.1)
            cots_timestep!(zero, 0.2, 0.1)
            @test legacy.N == off.N == zero.N
            @test legacy.body_condition == off.body_condition == zero.body_condition
            @test legacy.last_local_fecundity == off.last_local_fecundity ==
                zero.last_local_fecundity
            @test off.burden_ha == 0.0
            @test zero.last_hazard == 0.0
            @test zero.last_hazard_deaths_ha == 0.0
        end
        @test zero.burden_ha > 0.0

        p = COTSParams(IMM=0.0, p_tilde=0.0, lagged_adult_hazard=true,
            hazard_max=1.5, hazard_threshold_ha=1.5, hazard_width_ha=0.5,
            hazard_rho=0.5, hazard_coral_threshold=0.15,
            hazard_coral_width=0.03)
        density = COTSHuman(copy(initial), 0.8, p)
        fed = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0,
            p_tilde=0.0, lagged_adult_hazard=true, hazard_food_coupled=true,
            hazard_max=1.5))
        poor = COTSHuman(copy(initial), 0.8, fed.params)
        for model in (density, fed, poor)
            model.burden_ha = 2.0
        end
        cots_timestep!(density, 0.1, 0.0)
        cots_timestep!(fed, 0.3, 0.0)
        cots_timestep!(poor, 0.1, 0.0)
        expected_hazard = 1.5 * (0.5 / (0.5 + 0.5))
        expected_deaths = initial[3] * (1.0 - p.m3) * (1.0 - exp(-expected_hazard))
        @test density.last_burden_ha == 2.0
        @test density.last_hazard ≈ expected_hazard
        @test density.last_hazard_deaths_ha ≈ expected_deaths
        @test density.N[3] ≈ density.last_maturation +
            initial[3] * (1.0 - p.m3) - expected_deaths
        @test density.burden_ha ≈ 0.5 * 2.0 + 0.5 * initial[3]
        @test 0.0 < fed.last_hazard < poor.last_hazard < density.last_hazard
        @test poor.last_hazard_deaths_ha > fed.last_hazard_deaths_ha
        @test all(density.N .>= 0.0)
        @test cots_hazard_diagnostics([density]).adult_deaths_ha == [expected_deaths]

        below = COTSHuman(copy(initial), 0.8, p)
        below.burden_ha = p.hazard_threshold_ha
        cots_timestep!(below, 0.1, 0.0)
        @test below.last_hazard == 0.0
        @test below.last_hazard_deaths_ha == 0.0
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(lagged_adult_hazard=true, hazard_width_ha=0.0)), 0.1, 0.0)
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(lagged_adult_hazard=true, hazard_max=-1.0)), 0.1, 0.0)
    end

    @testset "structural switches are off by default and replay legacy exactly" begin
        initial = MVector{3,Float64}(0.4, 1.2, 2.5)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.01))
        off = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.01,
            starvation_preferred_threshold=0.2, starvation_memory_years=3.0,
            senescence_age=4, senescence_mortality=0.9,
            consumption_m2_per_adult_year=50.0))
        @test isempty(off.adult_ages)
        for (F, S) in ((0.3, 0.1), (0.02, 0.2), (0.0, 0.0), (0.06, 0.01))
            consumed_legacy = cots_timestep!(legacy, F, S)
            consumed_off = cots_timestep!(off, F, S)
            @test consumed_legacy == consumed_off
            @test legacy.N == off.N
            @test legacy.body_condition == off.body_condition
            @test legacy.last_local_fecundity == off.last_local_fecundity
            @test isnan(off.preferred_food_memory)
            @test off.last_senescent_adults == 0.0
        end
    end

    @testset "preferred-prey starvation ignores non-preferred cover" begin
        p_tilde = 0.95
        initial = MVector{3,Float64}(0.0, 1.0, 2.0)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0, p_tilde=p_tilde))
        preferred = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0,
            p_tilde=p_tilde, preferred_prey_starvation=true,
            starvation_preferred_threshold=0.05))
        cots_timestep!(legacy, 0.01, 0.30)
        cots_timestep!(preferred, 0.01, 0.30)
        expected_f = (1 - p_tilde) + p_tilde * (0.01 / 0.05)^3
        @test legacy.last_food_survival == 1.0
        @test preferred.last_food_survival ≈ expected_f
        s2 = 1 - preferred.params.m2
        s3 = 1 - preferred.params.m3
        @test preferred.N[3] ≈ 1.0 * s2 * expected_f + 2.0 * s3 * expected_f
        @test preferred.N[1] == legacy.N[1]
        @test preferred.preferred_food_memory == 0.01

        fed = COTSHuman(copy(initial), 0.8, preferred.params)
        cots_timestep!(fed, 0.06, 0.0)
        @test fed.last_food_survival == 1.0

        tau = 1.0
        weight = 1 - exp(-1 / tau)
        lagged = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0,
            preferred_prey_starvation=true, starvation_memory_years=tau))
        cots_timestep!(lagged, 0.20, 0.0)
        @test lagged.preferred_food_memory == 0.20
        cots_timestep!(lagged, 0.0, 0.0)
        @test lagged.preferred_food_memory ≈ (1 - weight) * 0.20
        @test lagged.last_food_survival == 1.0
        cots_timestep!(lagged, 0.0, 0.0)
        @test lagged.preferred_food_memory ≈ (1 - weight)^2 * 0.20
        @test lagged.last_food_survival < 1.0

        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(preferred_prey_starvation=true,
                starvation_preferred_threshold=0.0)), 0.1, 0.0)
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(preferred_prey_starvation=true,
                starvation_memory_years=-1.0)), 0.1, 0.0)
    end

    @testset "adult senescence ages cohorts and conserves adults" begin
        p = COTSParams(IMM=0.0, m3=0.2, adult_senescence=true,
            senescence_age=6, senescence_mortality=0.8)
        ages = stable_adult_ages(p, 3.5)
        @test length(ages) == 5
        @test sum(ages) ≈ 3.5
        @test ages[2] / ages[1] ≈ 0.8
        @test ages[5] ≈ ages[4] * 0.8 / 0.8

        model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p)
        model.N[3] = 1.0
        @test_throws ArgumentError cots_timestep!(model, 0.4, 0.1)
        model.adult_ages .= [1.0, 0.0, 0.0, 0.0, 0.0]
        for year in 1:4
            cots_timestep!(model, 0.4, 0.1)
            @test model.N[3] ≈ sum(model.adult_ages)
            @test model.adult_ages[year + 1] ≈ 0.8^year
        end
        @test model.last_senescent_adults == 0.0
        cots_timestep!(model, 0.4, 0.1)
        @test model.last_senescent_adults ≈ 0.8^4
        @test model.last_senescent_deaths_ha ≈ 0.8^4 * (0.8 - 0.2)
        @test model.adult_ages[5] ≈ 0.8^4 * 0.2
        @test cots_mechanism_diagnostics([model]).senescent_adults ≈ [0.8^4]

        initial = MVector{3,Float64}(0.5, 1.0, 2.0)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0, m3=0.2))
        matched = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0, m3=0.2,
            adult_senescence=true, senescence_age=5, senescence_mortality=0.2))
        for (F, S) in ((0.3, 0.1), (0.02, 0.01), (0.2, 0.0))
            cots_timestep!(legacy, F, S)
            cots_timestep!(matched, F, S)
            @test matched.N[3] ≈ legacy.N[3]
            @test matched.N[1] ≈ legacy.N[1]
            @test matched.N[2] ≈ legacy.N[2]
        end

        reset = COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p)
        reset.N .= [0.0, 0.0, 2.0]
        reset.preferred_food_memory = 0.3
        reset_adult_ages!(reset)
        @test sum(reset.adult_ages) ≈ 2.0
        @test isnan(reset.preferred_food_memory)

        @test_throws ArgumentError stable_adult_ages(COTSParams(senescence_age=2), 1.0)
        @test_throws ArgumentError stable_adult_ages(
            COTSParams(senescence_mortality=0.0), 1.0)
    end

    @testset "per-capita consumption is area-based and preferred-first" begin
        p = COTSParams(IMM=0.0, per_capita_consumption=true,
            consumption_m2_per_adult_year=10.0)
        # Like the legacy response, grazing uses post-transition adults.
        model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 50.0), 0.8, p)
        Cons_F, Cons_S = cots_timestep!(model, 0.30, 0.20)
        @test model.N[3] ≈ 50.0 * (1 - p.m3)
        @test Cons_F ≈ model.N[3] * 10.0 / 10_000.0
        @test Cons_S == 0.0
        model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 50.0), 0.8, p)
        Cons_F, Cons_S = cots_timestep!(model, 0.02, 0.20)
        @test Cons_F == 0.02
        @test Cons_S ≈ model.N[3] * 10.0 / 10_000.0 - 0.02
        model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 500.0), 0.8, p)
        Cons_F, Cons_S = cots_timestep!(model, 0.1, 0.1)
        @test (Cons_F, Cons_S) == (0.1, 0.1)
        @test_throws ArgumentError cots_timestep!(COTSHuman(
            MVector{3,Float64}(0.0, 0.0, 1.0), 0.8, COTSParams(
                per_capita_consumption=true, consumption_m2_per_adult_year=-1.0)), 0.1, 0.1)
    end

    @testset "per-capita grazing converts site density to habitable cover" begin
        p = COTSParams(IMM=0.0, per_capita_consumption=true,
            consumption_m2_per_adult_year=10.0)
        full = COTSHuman(MVector{3,Float64}(0.0, 0.0, 50.0), 0.8, p)
        half = COTSHuman(MVector{3,Float64}(0.0, 0.0, 50.0), 0.8, p)
        Cons_full, _ = cots_timestep!(full, 0.30, 0.20)
        Cons_half, _ = cots_timestep!(half, 0.30, 0.20; habitable_fraction=0.5)
        @test Cons_half ≈ 2.0 * Cons_full
        @test_throws ArgumentError cots_timestep!(COTSHuman(
            MVector{3,Float64}(0.0, 0.0, 1.0), 0.8, p), 0.1, 0.1; habitable_fraction=0.0)

        cover = zeros(5, 1, 2)
        cover[1, 1, :] .= 0.3
        models = [COTSHuman(MVector{3,Float64}(0.0, 0.0, 50.0), 0.8, p) for _ in 1:2]
        cots_mortality!(cover, models, COTSPreyMap([1, 2, 3], [4, 5]);
            habitable_fraction=[1.0, 0.5])
        @test 0.3 - cover[1, 1, 2] ≈ 2.0 * (0.3 - cover[1, 1, 1])
        @test_throws DimensionMismatch cots_mortality!(cover, models,
            COTSPreyMap([1, 2, 3], [4, 5]); habitable_fraction=[1.0])
    end

    @testset "density-unit rescaling is an exact reparameterisation" begin
        k = 19.6
        for structural in (false, true)
            base = COTSParams(IMM=0.03, a_F=1.18, a_S=0.165, b_ricker=0.0588,
                allee_threshold=3.2, m3=0.195, C_max=0.498,
                preferred_prey_starvation=structural, adult_senescence=structural,
                starvation_memory_years=0.5)
            scaled = COTSParams(IMM=0.03 * k, a_F=1.18 / k, a_S=0.165 / k,
                b_ricker=0.0588 / k, allee_threshold=3.2 * k, m3=0.195, C_max=0.498,
                preferred_prey_starvation=structural, adult_senescence=structural,
                starvation_memory_years=0.5)
            original = COTSHuman(MVector{3,Float64}(0.6, 0.3, 0.5), 0.8, base)
            migrated = COTSHuman(MVector{3,Float64}(0.6k, 0.3k, 0.5k), 0.8, scaled)
            for (F, S) in ((0.3, 0.1), (0.04, 0.2), (0.12, 0.05), (0.01, 0.01))
                consumed = cots_timestep!(original, F, S)
                consumed_migrated = cots_timestep!(migrated, F, S)
                @test collect(migrated.N) ≈ k .* collect(original.N) rtol = 1e-12
                @test collect(consumed_migrated) ≈ collect(consumed) rtol = 1e-12
                @test migrated.body_condition ≈ original.body_condition
                structural && @test migrated.adult_ages ≈ k .* original.adult_ages rtol = 1e-12
            end
        end
    end

    @testset "prey switching and preferred-food condition are opt-in" begin
        initial = MVector{3,Float64}(0.4, 1.2, 25.0)
        base = COTSParams(IMM=0.0, per_capita_consumption=true)
        legacy = COTSHuman(copy(initial), 0.8, base)
        off = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0,
            per_capita_consumption=true, switch_midpoint=0.3, switch_steepness=5.0,
            condition_saturation=0.5, condition_memory_years=3.0))
        for (F, S) in ((0.3, 0.1), (0.02, 0.2), (0.0, 0.0))
            @test cots_timestep!(legacy, F, S) == cots_timestep!(off, F, S)
            @test legacy.N == off.N
            @test legacy.body_condition == off.body_condition
            @test off.last_preferred_diet_share == 1.0
        end

        switching = COTSParams(IMM=0.0, per_capita_consumption=true,
            consumption_m2_per_adult_year=10.0, prey_switching=true,
            switch_midpoint=0.10, switch_steepness=70.0)
        rho(F) = 1 / (1 + exp(-70 * (F - 0.10)))
        for F in (0.30, 0.10, 0.02)
            model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 50.0), 0.8, switching)
            Cons_F, Cons_S = cots_timestep!(model, F, 0.30)
            demand = model.N[3] * 10.0 / 10_000.0
            @test model.last_preferred_diet_share ≈ rho(F)
            @test Cons_F ≈ min(rho(F) * demand, F)
            @test Cons_F + Cons_S ≈ demand
        end
        @test rho(0.02) < 0.01
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(prey_switching=true)), 0.1, 0.1)

        tau = 1.0
        weight = 1 - exp(-1 / tau)
        condition = COTSParams(IMM=0.0, preferred_condition=true,
            condition_saturation=0.10, condition_memory_years=tau)
        starving = COTSHuman(copy(initial), 0.8, condition)
        legacy_condition = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0))
        cots_timestep!(starving, 0.02, 0.40)
        cots_timestep!(legacy_condition, 0.02, 0.40)
        @test starving.body_condition ≈ (1 - weight) * 0.8 + weight * 0.2
        @test legacy_condition.body_condition > starving.body_condition
        fed = COTSHuman(copy(initial), 0.8, condition)
        cots_timestep!(fed, 0.25, 0.0)
        @test fed.body_condition ≈ (1 - weight) * 0.8 + weight * 1.0
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(preferred_condition=true, condition_memory_years=0.0)), 0.1, 0.1)
    end

    @testset "diet-linked starvation and non-preferred feeding efficiency" begin
        rho(F) = 1 / (1 + exp(-70 * (F - 0.10)))
        p_tilde = 0.95
        linked = COTSParams(IMM=0.0, p_tilde=p_tilde, per_capita_consumption=true,
            prey_switching=true, preferred_prey_starvation=true,
            diet_linked_starvation=true)
        for F in (0.30, 0.10, 0.06, 0.02)
            model = COTSHuman(MVector{3,Float64}(0.0, 1.0, 2.0), 0.8, linked)
            cots_timestep!(model, F, 0.3)
            @test model.last_food_survival ≈ (1 - p_tilde) + p_tilde * rho(F)
        end
        # Cubic H1 says fed at F=0.06 (above theta=0.05); diet-linked says starving.
        cubic = COTSHuman(MVector{3,Float64}(0.0, 1.0, 2.0), 0.8, COTSParams(IMM=0.0,
            p_tilde=p_tilde, per_capita_consumption=true, prey_switching=true,
            preferred_prey_starvation=true))
        cots_timestep!(cubic, 0.06, 0.3)
        @test cubic.last_food_survival == 1.0
        @test_throws ArgumentError cots_timestep!(COTSHuman(MVector{3,Float64}(0.0, 1.0, 2.0),
            0.8, COTSParams(per_capita_consumption=true, preferred_prey_starvation=true,
                diet_linked_starvation=true)), 0.1, 0.1)

        efficient = COTSParams(IMM=0.0, per_capita_consumption=true,
            consumption_m2_per_adult_year=10.0)
        reduced = COTSParams(IMM=0.0, per_capita_consumption=true,
            consumption_m2_per_adult_year=10.0, nonpreferred_feeding_efficiency=0.2)
        full_model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 50.0), 0.8, efficient)
        reduced_model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 50.0), 0.8, reduced)
        Cons_F_full, Cons_S_full = cots_timestep!(full_model, 0.01, 0.30)
        Cons_F_red, Cons_S_red = cots_timestep!(reduced_model, 0.01, 0.30)
        @test Cons_F_red == Cons_F_full == 0.01
        @test Cons_S_red ≈ 0.2 * Cons_S_full
        @test_throws ArgumentError cots_timestep!(COTSHuman(MVector{3,Float64}(0.0, 0.0, 1.0),
            0.8, COTSParams(nonpreferred_feeding_efficiency=1.5)), 0.1, 0.1)
    end

    @testset "juvenile predator pit, preferred maturation and gradual Allee" begin
        initial = MVector{3,Float64}(0.0, 10.0, 2.0)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0))
        off = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0,
            juvenile_predation_capacity=5.0, allee_exponent=2.0))
        cots_timestep!(legacy, 0.3, 0.1)
        cots_timestep!(off, 0.3, 0.1)
        @test legacy.N == off.N
        @test off.last_juvenile_predation_ha == 0.0
        @test off.last_maturation_fraction == 1.0

        Q = 20.0
        pit = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0,
            juvenile_predation=true, juvenile_predation_capacity=Q))
        cots_timestep!(pit, 0.3, 0.1)
        J = 10.0 * (1 - pit.params.m2)
        removed = Q * J / (J + Q)
        @test pit.last_juvenile_predation_ha ≈ removed
        @test pit.last_maturation ≈ J - removed
        @test pit.N[3] ≈ (J - removed) + 2.0 * (1 - pit.params.m3)
        swamped = COTSHuman(MVector{3,Float64}(0.0, 1000.0, 0.0), 0.8, pit.params)
        cots_timestep!(swamped, 0.3, 0.1)
        @test swamped.last_juvenile_predation_ha / (1000.0 * (1 - pit.params.m2)) < 0.03

        p_tilde = 0.95
        delayed = COTSHuman(copy(initial), 0.8, COTSParams(IMM=0.0, p_tilde=p_tilde,
            preferred_prey_starvation=true, preferred_maturation=true))
        cots_timestep!(delayed, 0.04, 0.3)
        g = sqrt(0.04)
        f = (1 - p_tilde) + p_tilde * (0.04 / 0.05)^3
        J = 10.0 * (1 - delayed.params.m2)
        @test delayed.last_maturation_fraction ≈ g
        @test delayed.last_maturation ≈ J * g * f
        @test delayed.N[2] ≈ J * (1 - g)
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(preferred_maturation=true, juvenile_storage=true)), 0.1, 0.1)
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(juvenile_predation=true, juvenile_predation_capacity=0.0)), 0.1, 0.1)

        for A in (1.0, 3.0, 30.0)
            hill = COTSHuman(MVector{3,Float64}(0.0, 0.0, A), 1.0, COTSParams(IMM=0.0,
                separated_recruitment=true, larval_fecundity=8.0, allee_threshold=3.0))
            stretched = COTSHuman(MVector{3,Float64}(0.0, 0.0, A), 1.0, COTSParams(IMM=0.0,
                separated_recruitment=true, larval_fecundity=8.0, allee_threshold=3.0,
                allee_stretched=true))
            cots_timestep!(hill, 0.4, 0.1)
            cots_timestep!(stretched, 0.4, 0.1)
            ratio = stretched.last_local_fecundity / hill.last_local_fecundity
            @test ratio ≈ exp(-(3.0 / A)^0.5) / (A^2 / (9.0 + A^2))
        end
        zero_adults = COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 1.0, COTSParams(IMM=0.0,
            separated_recruitment=true, allee_stretched=true))
        cots_timestep!(zero_adults, 0.4, 0.1)
        @test zero_adults.last_local_fecundity == 0.0
    end

    @testset "forced grazing applies prescribed per-capita preferred-first demand" begin
        prey = COTSPreyMap([1, 2], [3])
        cover = zeros(3, 2, 3)
        cover[1, :, :] .= 0.02
        cover[2, :, :] .= 0.01
        cover[3, :, :] .= 0.10
        original = copy(cover)
        consumed_F, consumed_S = forced_grazing!(cover, [0.0, 20.0, 100.0], prey;
            habitable_fraction=[0.5, 0.5, 0.5], consumption_m2_per_adult_year=10.0)
        @test cover[:, :, 1] == original[:, :, 1]
        F, S = 0.06, 0.20
        demand2 = 20.0 * 10.0 / 5_000.0
        @test consumed_F[2] ≈ demand2
        @test consumed_S[2] == 0.0
        @test sum(cover[1:2, :, 2]) ≈ F - demand2
        @test cover[3, :, 2] == original[3, :, 2]
        demand3 = 100.0 * 10.0 / 5_000.0
        @test consumed_F[3] ≈ F
        @test consumed_S[3] ≈ demand3 - F
        @test sum(cover[1:2, :, 3]) ≈ 0.0 atol = 1e-15
        @test sum(cover[3, :, 3]) ≈ S - (demand3 - F)
        # Same arithmetic as the per-capita path of cots_mortality!: with m3 = 0 and
        # ample food the post-transition adults equal the forced density.
        dynamic = copy(original)
        models = [COTSHuman(MVector{3,Float64}(0.0, 0.0, A), 0.8, COTSParams(IMM=0.0, m3=0.0,
            per_capita_consumption=true, consumption_m2_per_adult_year=10.0))
            for A in (0.0, 20.0, 100.0)]
        cots_mortality!(dynamic, models, prey; habitable_fraction=[0.5, 0.5, 0.5])
        @test [m.N[3] for m in models] ≈ [0.0, 20.0, 100.0]
        @test dynamic ≈ cover
        @test_throws DimensionMismatch forced_grazing!(copy(original), [1.0, 2.0], prey)
        @test_throws ArgumentError forced_grazing!(copy(original), [1.0, -2.0, 0.0], prey)
    end

    @testset "starvation grace delays preferred-prey starvation" begin
        base = (IMM=0.0, p_tilde=0.95, preferred_prey_starvation=true,
            starvation_memory_years=0.0)
        initial = MVector{3,Float64}(0.0, 5.0, 20.0)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(; base...))
        zero_grace = COTSHuman(copy(initial), 0.8, COTSParams(; base..., starvation_grace_years=0))
        graced = COTSHuman(copy(initial), 0.8, COTSParams(; base..., starvation_grace_years=1))
        f_low = 0.05 + 0.95 * (0.02 / 0.05)^3
        for (step, F) in enumerate((0.02, 0.02, 0.2, 0.02))
            cots_timestep!(legacy, F, 0.3)
            cots_timestep!(zero_grace, F, 0.3)
            cots_timestep!(graced, F, 0.3)
            @test zero_grace.N == legacy.N
            @test zero_grace.last_food_survival == legacy.last_food_survival
            expected = step in (1, 3, 4) ? 1.0 : f_low
            @test graced.last_food_survival ≈ expected
        end
        @test graced.low_food_years == 1
        @test cots_mechanism_diagnostics([graced]).low_food_years == [1.0]
        reset_adult_ages!(graced)
        @test graced.low_food_years == 0
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(IMM=0.0, starvation_grace_years=1)), 0.1, 0.1)
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(; base..., starvation_grace_years=-1)), 0.1, 0.1)
    end

    @testset "post-grazing starvation uses preferred cover left after demand" begin
        initial = MVector{3,Float64}(0.0, 5.0, 60.0)
        base = (IMM=0.0, p_tilde=0.95, per_capita_consumption=true,
            consumption_m2_per_adult_year=10.0, preferred_prey_starvation=true,
            starvation_memory_years=0.0)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(; base...))
        cots_timestep!(legacy, 0.08, 0.3; habitable_fraction=0.6)
        @test legacy.preferred_food_memory == 0.08
        @test legacy.last_food_survival == 1.0
        treated = COTSHuman(copy(initial), 0.8, COTSParams(; base...,
            starvation_post_grazing=true))
        cots_timestep!(treated, 0.08, 0.3; habitable_fraction=0.6)
        left = 0.08 - 60.0 * 10.0 / (10_000.0 * 0.6)
        @test treated.preferred_food_memory ≈ max(left, 0.0)
        @test treated.last_food_survival ≈ 0.05 + 0.95 * (max(left, 0.0) / 0.05)^3
        exhausted = COTSHuman(MVector{3,Float64}(0.0, 0.0, 200.0), 0.8, treated.params)
        cots_timestep!(exhausted, 0.08, 0.3; habitable_fraction=0.6)
        @test exhausted.preferred_food_memory == 0.0
        @test exhausted.last_food_survival ≈ 0.05
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(IMM=0.0, starvation_post_grazing=true,
                preferred_prey_starvation=true)), 0.1, 0.1)
    end

    @testset "post-outbreak collapse is opt-in, state-triggered and mass-accounted" begin
        initial = MVector{3,Float64}(5.0, 40.0, 200.0)
        base = (IMM=0.0, adult_senescence=true)
        legacy = COTSHuman(copy(initial), 0.8, COTSParams(; base...))
        off = COTSHuman(copy(initial), 0.8, COTSParams(; base...,
            collapse_threshold_ha=1.0, collapse_duration_years=1))
        for model in (legacy, off)
            reset_adult_ages!(model)
            cots_timestep!(model, 0.4, 0.1)
        end
        @test legacy.N == off.N
        @test off.outbreak_clock == 0
        @test off.last_collapse_removed_ha == 0.0

        params(; kwargs...) = COTSParams(; base..., outbreak_collapse=true,
            collapse_duration_years=2, kwargs...)
        for juveniles in (false, true)
            control = COTSHuman(copy(initial), 0.8, COTSParams(; base...))
            model = COTSHuman(copy(initial), 0.8, params(collapse_juveniles=juveniles))
            reset_adult_ages!(control)
            reset_adult_ages!(model)
            cots_timestep!(control, 0.4, 0.1)
            cots_timestep!(model, 0.4, 0.1)
            @test model.N == control.N
            @test model.outbreak_clock == 1
            @test model.last_collapse_removed_ha == 0.0
            cots_timestep!(control, 0.4, 0.1)
            cots_timestep!(model, 0.4, 0.1)
            @test control.N[3] >= 67.5
            dieback = juveniles ? control.N[2] + control.N[3] : control.N[3]
            scale = 0.25 / dieback
            @test model.outbreak_clock == 0
            @test model.last_collapse_removed_ha ≈ dieback - 0.25
            @test model.N[3] ≈ control.N[3] * scale
            @test model.N[2] ≈ (juveniles ? control.N[2] * scale : control.N[2])
            @test model.N[1] == control.N[1]
            @test sum(model.adult_ages) ≈ model.N[3]
            @test model.adult_ages ≈ control.adult_ages .* scale
            @test cots_mechanism_diagnostics([model]).collapse_removed_ha ≈
                [dieback - 0.25]
        end

        # A year below the threshold resets the clock, so no collapse occurs.
        dipping = COTSHuman(MVector{3,Float64}(0.0, 0.0, 150.0), 0.8, params())
        reset_adult_ages!(dipping)
        cots_timestep!(dipping, 0.4, 0.1)
        @test dipping.outbreak_clock == 1
        dipping.N[3] = 10.0
        reset_adult_ages!(dipping)
        @test dipping.outbreak_clock == 0
        dipping.N[3] = 10.0
        dipping.outbreak_clock = 1
        cots_timestep!(dipping, 0.4, 0.1)
        @test dipping.N[3] < 67.5
        @test dipping.outbreak_clock == 0
        @test dipping.last_collapse_removed_ha == 0.0

        # Without senescence the adult total is rescaled directly.
        plain = COTSHuman(MVector{3,Float64}(0.0, 0.0, 300.0), 0.8, COTSParams(IMM=0.0,
            outbreak_collapse=true, collapse_duration_years=1))
        cots_timestep!(plain, 0.4, 0.1)
        @test plain.N[3] ≈ 0.25
        @test plain.outbreak_clock == 0

        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(outbreak_collapse=true, collapse_duration_years=0)), 0.1, 0.1)
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(outbreak_collapse=true, collapse_threshold_ha=0.0)), 0.1, 0.1)
        @test_throws ArgumentError cots_timestep!(COTSHuman(copy(initial), 0.8,
            COTSParams(outbreak_collapse=true, collapse_background_ha=100.0)), 0.1, 0.1)
    end

    @testset "structural switches replay deterministically" begin
        p = COTSParams(IMM=0.0, preferred_prey_starvation=true,
            starvation_memory_years=0.5, adult_senescence=true)
        runs = map(1:2) do _
            model = COTSHuman(MVector{3,Float64}(1.0, 1.0, 2.0), 0.8, p)
            for F in (0.3, 0.1, 0.02, 0.0, 0.08)
                cots_timestep!(model, F, 0.2)
            end
            (copy(model.N), copy(model.adult_ages), model.preferred_food_memory)
        end
        @test runs[1] == runs[2]
    end

    @testset "Allee effect suppresses recruitment at low density" begin
        p = test_cots_params()
        model_low = COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.1), 0.8, p)
        model_high = COTSHuman(MVector{3,Float64}(0.0, 0.0, 3.0), 0.8, p)
        cots_timestep!(model_low, 0.5, 0.2)
        cots_timestep!(model_high, 0.5, 0.2)
        @test model_high.N[1] > model_low.N[1] * 5.0
    end

    @testset "cots_mortality! reduces cover" begin
        p = test_cots_params()
        C_cover_t = fill(0.05, 5, 7, 3)
        prey_map = COTSPreyMap([1, 2, 3], [4, 5])
        models = [COTSHuman(MVector{3,Float64}(0.0, 0.0, 2.0), 0.8, p) for _ in 1:3]
        cover_before = sum(C_cover_t)
        cots_mortality!(C_cover_t, models, prey_map)
        @test sum(C_cover_t) < cover_before
        @test all(0.0 .<= C_cover_t .<= 1.0)
    end

    @testset "disperse_cots_larvae!" begin
        p = test_cots_params()
        models = [
            COTSHuman(MVector{3,Float64}(10.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p)
        ]
        conn = sparse([1, 1, 2, 3], [2, 3, 2, 3], [0.5, 0.3, 0.1, 0.1], 4, 4)
        disperse_cots_larvae!(models, conn)
        @test models[2].N[1] > 0.0
        @test models[3].N[1] > 0.0
        @test all(m -> all(isfinite.(m.N)) && all(m.N .>= 0.0), models)
    end

    @testset "init_cots_populations" begin
        p = test_cots_params()
        seed = Set([1, 5, 10])
        models = init_cots_populations(20, p; seed_locs=seed, init_density=2.0)
        @test length(models) == 20
        @test models[1].N[3] == 2.0
        @test models[5].N[3] == 2.0
        @test models[10].N[3] == 2.0
        @test models[2].N[3] == 0.0
        models_rand = init_cots_populations(20, p; rng=Random.MersenneTwister(42))
        @test count(m -> m.N[3] > 0.0, models_rand) >= 1
    end

    @testset "inject_upstream_pulse!" begin
        p = test_cots_params()
        models = [COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p) for _ in 1:10]
        inject_upstream_pulse!(models, Set([2, 5, 8]); pulse_val=2.0)
        @test models[2].N[1] == 2.0
        @test models[5].N[1] == 2.0
        @test models[8].N[1] == 2.0
        @test models[1].N[1] == 0.0
    end

    @testset "cots_mortality! preserves non-prey groups and applies proportional mortality" begin
        p = test_cots_params()
        prey_map = COTSPreyMap([1], [2])
        C_cover_t = zeros(3, 2, 1)
        C_cover_t[1, :, 1] .= [0.2, 0.4]
        C_cover_t[2, :, 1] .= [0.1, 0.3]
        C_cover_t[3, :, 1] .= [0.7, 0.9]
        non_prey_before = copy(C_cover_t[3, :, 1])
        models = [COTSHuman(MVector{3,Float64}(0.0, 0.0, 1.5), 0.8, p)]
        cots_mortality!(C_cover_t, models, prey_map)
        @test isapprox(C_cover_t[1, 1, 1] / 0.2, C_cover_t[1, 2, 1] / 0.4)
        @test isapprox(C_cover_t[2, 1, 1] / 0.1, C_cover_t[2, 2, 1] / 0.3)
        @test C_cover_t[3, :, 1] == non_prey_before
        @test sum(C_cover_t[1:2, :, 1]) < 1.0
        @test all(0.0 .<= C_cover_t .<= 1.0)
    end

    @testset "disperse_cots_larvae! skips self-connectivity and applies scalar" begin
        p = test_cots_params()
        models = [
            COTSHuman(MVector{3,Float64}(10.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(1.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p)
        ]
        conn = sparse([1, 1, 1, 2, 2], [1, 2, 3, 2, 4], [0.9, 0.5, 0.3, 0.1, 0.25], 4, 4)
        disperse_cots_larvae!(models, conn; immigration_scalar=2.0)
        @test models[1].N[1] == 10.0
        @test models[2].N[1] == 11.0
        @test models[3].N[1] == 6.0
        @test models[4].N[1] == 0.5
    end

    @testset "inject_upstream_pulse! vector pulse values" begin
        p = test_cots_params()
        models = [COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p) for _ in 1:10]
        pulse_vals = collect(0.1:0.1:1.0)
        inject_upstream_pulse!(models, Set([2, 5, 8, 99]), pulse_vals)
        @test models[2].N[1] == pulse_vals[2]
        @test models[5].N[1] == pulse_vals[5]
        @test models[8].N[1] == pulse_vals[8]
        @test models[1].N[1] == 0.0
        @test models[3].N[1] == 0.0
    end

    @testset "package-shaped API wrappers" begin
        p = test_cots_params()
        cots_state = initialize_cots(4, p; seed_locs=Set([1, 3]), init_density=1.2)
        @test cots_state isa Vector{COTSHuman}
        @test cots_state[1].N[3] == 1.2
        @test cots_state[2].N[3] == 0.0
        @test cots_state[3].N[3] == 1.2
        disabled_state = initialize_cots(3, p; enabled=false)
        @test all(m -> all(m.N .== 0.0), disabled_state)
        C_cover_t = fill(0.05, 5, 2, 4)
        cover_before = sum(C_cover_t)
        apply_predation!(C_cover_t, cots_state, COTSPreyMap([1, 2, 3], [4, 5]))
        @test sum(C_cover_t) < cover_before
        conn = sparse([1], [2], [0.5], 4, 4)
        before_recruits = cots_state[2].N[1]
        disperse_larvae!(cots_state, conn; scalar=2.0)
        @test cots_state[2].N[1] >= before_recruits
        pulse_vals = [0.0, 0.2, 0.4, 0.6]
        apply_external_supply!(cots_state, Set([2, 4]), pulse_vals)
        @test cots_state[4].N[1] >= pulse_vals[4]
    end

    @testset "juvenile storage is opt-in" begin
        base = COTSParams(IMM=0.0, juvenile_storage=false)
        stored = COTSParams(
            IMM=0.0,
            juvenile_storage=true,
            juvenile_maturation_min=0.1,
            juvenile_maturation_max=1.0,
            juvenile_maturation_cover=0.4
        )
        legacy_model = COTSHuman(MVector{3,Float64}(0.0, 1.0, 0.0), 0.8, base)
        stored_model = COTSHuman(MVector{3,Float64}(0.0, 1.0, 0.0), 0.8, stored)
        cots_timestep!(legacy_model, 0.01, 0.0)
        cots_timestep!(stored_model, 0.01, 0.0)
        @test legacy_model.N[2] == 0.0
        @test stored_model.N[2] > 0.0
        @test stored_model.N[3] < legacy_model.N[3]
        @test stored_model.last_retained_juveniles == stored_model.N[2]
    end

    @testset "reef-scale dispersal and source survival" begin
        p = COTSParams(IMM=0.0)
        models = [
            COTSHuman(MVector{3,Float64}(10.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p)
        ]
        reef_conn = [0.0 0.5; 0.2 0.0]
        disperse_larvae_by_reef!(models, reef_conn, [1, 1, 2, 2]; source_survival=[1.0, 0.0])
        @test models[3].N[1] == 2.5
        @test models[4].N[1] == 2.5
        @test sum(cots_flow_diagnostics(models).internal_immigration) == 5.0

        base_recruits = [1.0, 2.0, 3.0, 4.0]
        site_models = [COTSHuman(MVector{3,Float64}(value, 0.0, 0.0), 0.8, p) for value in base_recruits]
        reef_models = [COTSHuman(MVector{3,Float64}(value, 0.0, 0.0), 0.8, p) for value in base_recruits]
        site_conn = zeros(4, 4)
        mapping = [1, 1, 2, 2]
        for source in 1:4, sink in 1:4
            site_conn[source, sink] = reef_conn[mapping[source], mapping[sink]] / 2
        end
        disperse_larvae!(site_models, sparse(site_conn))
        disperse_larvae_by_reef!(reef_models, reef_conn, mapping)
        @test [m.N[1] for m in reef_models] ≈ [m.N[1] for m in site_models]
    end

    @testset "absolute external supply is logged" begin
        p = COTSParams(IMM=0.0)
        models = [COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p) for _ in 1:3]
        apply_external_supply!(models, [0.1, 0.2, 0.3])
        @test [m.N[1] for m in models] == [0.1, 0.2, 0.3]
        @test cots_flow_diagnostics(models).external_immigration == [0.1, 0.2, 0.3]
    end

    @testset "separated recruitment defers production until dispersal" begin
        p = COTSParams(
            IMM=0.0,
            separated_recruitment=true,
            larval_fecundity=8.0,
            settlement_probability=0.25,
            allee_threshold=3.0,
            b_ricker=0.1
        )
        model = COTSHuman(MVector{3,Float64}(0.0, 0.0, 3.0), 1.0, p)
        cots_timestep!(model, 0.4, 0.1)
        expected = 8.0 * model.body_condition^2 * 3.0 * exp(-0.3) * 0.5
        @test model.last_local_fecundity ≈ expected
        @test model.N[1] == 0.0
        @test model.last_settled_recruits == 0.0
    end

    @testset "separated reef dispersal preserves production, survival, and settlement stages" begin
        p = COTSParams(
            IMM=0.0,
            separated_recruitment=true,
            settlement_probability=0.5
        )
        models = [
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p)
        ]
        models[1].last_local_fecundity = 10.0
        models[2].last_local_fecundity = 4.0
        reef_conn = [0.5 0.25; 0.1 0.4]
        disperse_larvae_by_reef!(
            models,
            reef_conn,
            [1, 2];
            source_survival=[0.2, 0.5],
            reef_areas=[2.0, 1.0]
        )
        flows = cots_flow_diagnostics(models)
        @test flows.pelagic_survivors ≈ [1.1, 1.8]
        @test flows.local_retention ≈ [0.5, 0.4]
        @test flows.internal_immigration ≈ [0.05, 0.5]
        @test flows.settled_recruits ≈ [0.55, 0.9]
        @test [model.N[1] for model in models] ≈ [0.55, 0.9]
    end

    @testset "external larval supply applies settlement exactly once" begin
        p = COTSParams(
            IMM=0.0,
            separated_recruitment=true,
            settlement_probability=0.5
        )
        models = [
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p),
            COTSHuman(MVector{3,Float64}(0.0, 0.0, 0.0), 0.8, p)
        ]
        apply_external_larval_supply!(models, [2.0, 3.0], [1.0, 2.0])
        flows = cots_flow_diagnostics(models)
        @test flows.external_potential == [2.0, 3.0]
        @test flows.external_pelagic == [1.0, 2.0]
        @test flows.external_immigration == [0.5, 1.0]
        @test flows.settled_recruits == [0.5, 1.0]
        @test [model.N[1] for model in models] == [0.5, 1.0]
    end
end
