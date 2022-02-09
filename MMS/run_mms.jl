include("mms.jl")

let

    year_seconds = 31556952
    
    # Domain length
    Lw = 1
    # Basin width
    W = 24/40
    # Basin Depth
    D = 6/40
    # simulation timespan
    
    
    # mesh refinement

    ns = 2 .^ (4:9)
    
    # order of operators
    p = [4]

    # Basin Params
    B_p = (μ_out = 24.0,
           ρ_out = 3.0,
           μ_in = 18.0,
           ρ_in = 2.6,
           c = (W/2)/D,
           r̄ = (W/2)^2,
           r_w = .01,
           on = 1)

    # Rate-and-State Friction Params`
    RS = (σn = 50.0,
          a = .015,
          b0 = .02,
          Dc = 1e9,
          f0 = .6,
          V0 = 1e-6,
          τ_inf = 24.82,
          Vp = 1e-9)

     MMS = (wl = Lw/2,
           amp = .5,
           ϵ = .01)
    # MMS params
 #=   MMS = (Lw = Lw,
           t̄ = 35*year_seconds,
           t_w = 10,
           t_f = 70*year_seconds,
           τ∞ = 31.73,
           Vp = 1e-9,
           H = 8/40,
           Vmin = 1e-12,
           δ_e = 1e-9*(35*year_seconds) -
               (1e-12*(35*year_seconds)),         
           ϵ = 2.0)
=#
    refine(p, ns, Lw, D, B_p, RS, (-1, 0, 0, 1), MMS)

end
