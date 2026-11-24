# ETDRK2: stiff diagonal linear part Λ, nonlinear part N(u) = sin(u) + 0.3 shift(u),
# whose linearisation along an orbit u is J(u) y = cos(u) y + 0.3 shift(y).

_Λ  = [-1.0, -10.0, -50.0, -300.0]
_N2!(t, u, du)       = (du .= sin.(u) .+ 0.3.*circshift(u, 1) .+ cos(t); du)
_J2!(t, u, y, dy)    = (dy .= cos.(u).*y .+ 0.3.*circshift(y, 1); dy)    # tangent
_full2!(t, u, du)     = (_N2!(t, u, du); du .+= _Λ.*u; du)
_Jᵀ2!(t, u, y, dy)   = (dy .= cos.(u).*y .+ 0.3.*circshift(y, -1); dy)   # adjoint

@testset "ETDRK2 order of convergence            " begin
    u₀ = [1.0, -0.5, 0.3, 0.8]
    T  = 1.0

    # ---- reference: explicit RK4 on the full right-hand side, tiny step ----
    uref = copy(u₀)
    flow(_full2!, RK4(zeros(4)), TimeStepConstant(2e-5))(uref, (0, T))

    # ---- errors for halved steps ----
    Δts  = [0.02, 0.01, 0.005, 0.0025, 0.00125]
    errs = map(Δts) do Δt
        u = copy(u₀)
        flow(_N2!, Diagonal(_Λ), ETDRK2(zeros(4)), TimeStepConstant(Δt))(u, (0, T))
        norm(u - uref, Inf)
    end
    orders = log2.(errs[1:end-1] ./ errs[2:end])
    @test all(orders .> 1.5)
    @test 1.9 < orders[end] < 2.1
end

@testset "ETDRK2 exact for linear diagonal        " begin
    Δt = 0.02
    λ  = [-100/Δt, -1000.0, -3.0, -1e-8, 0.0, 0.5]
    u₀ = [1.0, 2.0, -1.0, 0.5, 3.0, 0.1]
    u  = copy(u₀)
    flow((t, u, du) -> (du .= 0; du), Diagonal(λ), ETDRK2(zeros(6)), TimeStepConstant(Δt))(u, (0, 10Δt))
    @test u ≈ exp.(λ.*10Δt).*u₀  rtol=1e-12  atol=1e-14
end

@testset "ETDRK2 continuous tangent and adjoint   " begin
    n  = 4
    Δt = 0.002
    u₀ = [1.0, -0.5, 0.3, 0.8]
    y₀ = [0.3, 1.0, -0.7, 0.2]
    T  = 1.0
    A  = Diagonal(_Λ)

    # ---- orbit in a storage ----
    store = RAMStorage(zeros(n); degree=5)
    flow(_N2!, A, ETDRK2(zeros(n)), TimeStepConstant(Δt))(copy(u₀), (0, T), store)

    # ---- tangent, from t=0 to T ----
    y = copy(y₀)
    flow(_J2!, A, ETDRK2(zeros(n), ContinuousMode(false)), TimeStepFromStorage(Δt))(y, store, (0, T))

    # ---- finite differences of the nonlinear flow ----
    ε  = 1e-6
    ϕ  = flow(_N2!, A, ETDRK2(zeros(n)), TimeStepConstant(Δt))
    up = copy(u₀) .+ ε.*y₀; ϕ(up, (0, T))
    um = copy(u₀) .- ε.*y₀; ϕ(um, (0, T))
    yfd = (up .- um) ./ (2ε)
    @test y ≈ yfd  rtol=1e-4

    # ---- CB3R2R3e at a tiny step ----
    ycb = copy(y₀)
    flow(_J2!, A, CB3R2R3e(zeros(n), ContinuousMode(false)), TimeStepFromStorage(Δt/10))(ycb, store, (0, T))
    @test y ≈ ycb  rtol=1e-4

    # ---- adjoint: <ψ y₀, w> = <y₀, ψᵀ w> ----
    w₀ = [0.5, -1.0, 0.4, 0.9]
    w  = copy(w₀)
    flow(_Jᵀ2!, A, ETDRK2(zeros(n), ContinuousMode(true)), TimeStepFromStorage(Δt))(w, store, (T, 0))
    @test dot(y, w₀) ≈ dot(y₀, w)  rtol=1e-4
end

@testset "ETDRK2 coupled systems                  " begin
    Δt = 0.005
    T  = 1.0
    u₀ = [1.0, -0.5, 0.3, 0.8]

    # ---- no linear part at all: Heun's RK2 ----
    f!(t, u, du) = (du .= sin.(u) .- 0.5.*u; du)
    ur = copy(u₀)
    k1 = zeros(4)
    k2 = zeros(4)
    for t in 0:Δt:(T - Δt/2)
        f!(t, ur, k1)
        f!(t + Δt, ur .+ Δt.*k1, k2)
        ur .+= Δt./2 .* (k1 .+ k2)
    end
    ue = copy(u₀)
    flow(f!, nothing, ETDRK2(zeros(4)), TimeStepConstant(Δt))(ue, (0, T))
    @test ue ≈ ur  rtol=1e-12
    uc = couple(copy(u₀), [0.0])
    q!(t, u, du, q, dq) = (dq .= sum(u); dq)
    g1 = (t, u, du) -> f!(t, u, du)
    flow(couple(g1, q!), couple(nothing, nothing), CallDependency((1,), (1, 2)),
         ETDRK2(couple(zeros(4), zeros(1))), TimeStepConstant(Δt))(uc, (0, T))
    @test uc[1] ≈ ur  rtol=1e-12

    # ---- Diagonal and nothing: u as alone, q = ∫ u₁, second order ----
    ua = copy(u₀)
    flow(_N2!, Diagonal(_Λ), ETDRK2(zeros(4)), TimeStepConstant(Δt))(ua, (0, T))
    z  = couple(copy(u₀), [0.0])
    flow(couple(_N2!, q!), couple(Diagonal(_Λ), nothing), CallDependency((1,), (1, 2)),
         ETDRK2(couple(zeros(4), zeros(1))), TimeStepConstant(Δt))(z, (0, T))
    @test z[1] ≈ ua  rtol=1e-14
    qref = [0.0]
    zz = couple(copy(u₀), [0.0])
    flow(couple((t, u, du) -> (_full2!(t, u, du)), q!), couple(nothing, nothing),
         CallDependency((1,), (1, 2)), RK4(couple(zeros(4), zeros(1))),
         TimeStepConstant(1e-4))(zz, (0, T)) # explicit with stiff term is stable at 1e-4
    @test z[2][1] ≈ zz[2][1]  rtol=2e-3

    # ---- k coupled tangent vectors equal k separate runs ----
    store = RAMStorage(zeros(4); degree=5)
    flow(_N2!, Diagonal(_Λ), ETDRK2(zeros(4)), TimeStepConstant(0.01))(copy(u₀), (0, T), store)
    ys = [[0.3, 1.0, -0.7, 0.2], [1.0, 0.0, 0.5, -1.0]]
    yc = couple(copy.(ys)...)
    flow(couple(_J2!, _J2!), couple(Diagonal(_Λ), Diagonal(_Λ)), CallDependency((1,), (2,)),
         ETDRK2(couple(zeros(4), zeros(4)), ContinuousMode(false)),
         TimeStepFromStorage(0.05))(yc, store, (0, T))
    for i in 1:2
        y = copy(ys[i])
        flow(_J2!, Diagonal(_Λ), ETDRK2(zeros(4), ContinuousMode(false)),
             TimeStepFromStorage(0.05))(y, store, (0, T))
        @test yc[i] ≈ y  rtol=1e-13
    end
end

@testset "ETDRK2 cache and errors                 " begin
    A = Diagonal(_Λ)
    m = ETDRK2(zeros(4))
    sys = Flows.System(_N2!, A)

    # ---- the coefficients follow the step ----
    x = [1.0, 2.0, 3.0, 4.0]
    Flows.step!(m, sys, 0.0, 0.1, x, nothing)
    @test m.Δt[] == 0.1
    @test m.coef[1] ≈ exp.(0.1.*_Λ)
    Flows.step!(m, sys, 0.0, 0.05, x, nothing)
    @test m.Δt[] == 0.05
    @test m.coef[1] ≈ exp.(0.05.*_Λ)
    @test m.coef[2] ≈ 0.05 .* (exp.(0.05.*_Λ) .- 1) ./ (0.05.*_Λ)

    # ---- results after a change equal those of a fresh method ----
    xa = [1.0, 2.0, 3.0, 4.0]; xb = copy(xa)
    Flows.step!(m, sys, 0.0, 0.1, xa, nothing)
    Flows.step!(ETDRK2(zeros(4)), sys, 0.0, 0.1, xb, nothing)
    @test xa == xb

    # ---- another operator ----
    sys2 = Flows.System(_N2!, Diagonal(2 .* _Λ))
    xa = [1.0, 2.0, 3.0, 4.0]; xb = copy(xa)
    Flows.step!(m, sys2, 0.0, 0.1, xa, nothing)
    Flows.step!(ETDRK2(zeros(4)), sys2, 0.0, 0.1, xb, nothing)
    @test xa == xb

    # ---- unsupported ----
    @test_throws ArgumentError ETDRK2(zeros(4), DiscreteMode(false))
    @test_throws ErrorException Flows.step!(ETDRK2(zeros(2)), Flows.System(_N2!, [1.0 0.0; 0.0 1.0]),
                                            0.0, 0.1, zeros(2), nothing)
end

# A step allocates nothing after warm-up, also when the coefficients are cached
@testset "ETDRK2 allocations                      " begin
    n  = 4
    A  = Diagonal(_Λ)
    u₀ = [1.0, -0.5, 0.3, 0.8]
    # right-hand sides that do not allocate
    N!  = (t, u, du) -> (du .= sin.(u) .- 0.5.*u; du)
    J!  = (t, u, y, dy) -> (dy .= cos.(u).*y; dy)
    q!  = (t, u, du, q, dq) -> (dq[1] = u[1]; dq)
    store = RAMStorage(zeros(n); degree=5)
    flow(N!, A, ETDRK2(zeros(n)), TimeStepConstant(0.01))(copy(u₀), (0, 1.0), store)

    function allocs(m, sys, x, arg)
        Flows.step!(m, sys, 0.3, 0.01, x, arg)        # warm-up, computes coefficients
        a1 = @allocated Flows.step!(m, sys, 0.3, 0.01, x, arg)   # cache hit
        return a1
    end

    # normal
    @test allocs(ETDRK2(zeros(n)), Flows.System(N!, A), copy(u₀), nothing) == 0

    # continuous
    @test allocs(ETDRK2(zeros(n), ContinuousMode(false)), Flows.System(J!, A),
                 copy(u₀), store) == 0

    # coupled, with a component without linear part
    z  = couple(copy(u₀), [0.0])
    gc = couple(N!, q!)
    for Ac in (couple(A, nothing), couple(nothing, nothing))
        sys = Flows.System(gc, Ac, CallDependency((1,), (1, 2)))
        @test allocs(ETDRK2(couple(zeros(n), zeros(1))), sys, z, nothing) == 0
    end

    # coupled tangent
    sys = Flows.System(couple(J!, J!), couple(A, A), CallDependency((1,), (2,)))
    @test allocs(ETDRK2(couple(zeros(n), zeros(n)), ContinuousMode(false)), sys,
                 couple(copy(u₀), copy(u₀)), store) == 0
end
