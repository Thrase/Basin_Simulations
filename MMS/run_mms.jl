include("mms.jl")

let
    # Domain length
    Lw = 1.0
    # Basin Depth
    D = .25
    # simulation timespan
    t_span = (0, .01)
    
    # mesh refinement
    ns = 8 * 2 .^ (1:4)
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
          Dc = 10e6,
          f0 = .6,
          V0 = 1e-6)
    
    # MMS params
    MMS = (wl = Lw/2,
           amp = .5,
           ϵ = .01)

    refine(p, ns, t_span, Lw, D, B_p, RS, (-1, 0, 0, 1), MMS)

end
