include("mms.jl")

let

    year_seconds = 31556952
    # 1 is dynamic 0 is static
    test_type = 0
    # Domain length
    Lw = 40
    # Basin Depth
    D = .25
    # simulation timespan
    t_span = (0, .001)
    
    # mesh refinement
    ns = 2 * 2 .^ (6:6)
    @show ns
    # order of operators
    p = [2]

    # Basin Params
    B_p = (μ_out = 36.0,
           ρ_out = 2.8,
           μ_in = 8.0,
           ρ_in = 2.0,
           c = (Lw/2)/D,
           r̄ = (Lw/2)^2,
           r_w = 1 + (Lw/2)/D)

    # Rate-and-State Friction Params
    RS = (Hvw = 12,
          Ht = 6,
          σn = 50.0,
          a = .015,
          b0 = .02,
          bmin = 0.0,
          Dc = 1e6,
          f0 = .6,
          V0 = 1e-6,
          τ_inf = 24.82,
          Vp = 1e-9)

    
    # MMS params
    MMS = (Lw = Lw,
           wl = Lw/2,
           amp = .5,
           ϵ = .01,
           t̄ = 35*year_seconds,
           t_w = 10,
           t_f = 70*year_seconds,
           τ∞ = 31.73,
           Vp = 1e-9,
           H = 8,
           Vmin = 1e-12,
           δ_e = 1e-9*(35*year_seconds)/2 -
               (1e-12*(35*year_seconds)))

    refine(p, ns, t_span, Lw, D, B_p, RS, (-1, 0, 0, 1), MMS, test_type)

end
