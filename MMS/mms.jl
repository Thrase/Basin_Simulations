using Plots
using Printf

include("../physical_params.jl")
include("mms_funcs.jl")
include("../domain.jl")
include("../numerical.jl")
include("../solvers.jl")

function refine(ps, ns, t_span, Lw, D, B_p, RS, MMS)
    
    # expand to (0,Lw) × (0, Lw)
    (x1, x2, x3, x4) = (0, Lw, 0, Lw)
    (y1, y2, y3, y4) = (0, 0, Lw, Lw)

    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)
    
    for p in ps
        err = Vector{Float64}(undef, length(ns))
        for (iter, N) in enumerate(ns)
            
            nn = N + 1
            Nn = nn^2
            mt = @elapsed begin
                metrics = create_metrics(N, N, B_p, μ, xt, yt)
            end

            @printf "Got metrics: %s s\n" mt
       
            x = metrics.coord[1]
            y = metrics.coord[2]

            LFtoB = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN]
            ot = @elapsed begin

                lop = locoperator(p, N, N, B_p, μ, ρ, metrics, LFtoB)
                b = b_fun(metrics.facecoord[2][1], RS)
                
                dynamic_operators = (nn = nn,
                                     Nn = Nn,
                                     R = (-1, 0, 1, 0),
                                     L = lop.L,
                                     Ã = lop.Ã,
                                     JHP = lop.JI * lop.H̃I * lop.P̃I,
                                     P̃I = lop.P̃I,
                                     sJ = metrics.sJ,
                                     Γ = lop.Γ,
                                     Z̃f = lop.Z̃f,
                                     H = lop.H,
                                     Cf = lop.Cf,
                                     B = lop.B,
                                     n = lop.n,
                                     fcs = metrics.facecoord,
                                     coord = metrics.coord,
                                     τ̃ =  Array{Any, 1}(undef, 4),
                                     û =  Array{Any, 1}(undef, 4),
                                     dû = Array{Any, 1}(undef, 4),
                                     τ̂ = Array{Any, 1}(undef, 4),
                                     τf = Array{Float64, 1}(undef, nn),
                                     fault_v = Array{Float64, 1}(undef, nn),
                                     B_p = B_p,
                                     RS = RS,
                                     MMS = MMS,
                                     b = b,
                                     Forcing = Forcing,
                                     Char_Source = S_c,
                                     RS_Source = S_rs)
            end

            @printf "Got Operators: %s s\n" ot


            it = @elapsed begin
                u0 = ue(x[:], y[:], 0.0, MMS)
                v0 = ue_t(x[:], y[:], 0.0, MMS)
                q = [u0;v0]
                for i in 1:4
                    q = vcat(q, lop.L[i]*u0)
                end
                q = vcat(q, ψe(metrics.facecoord[1][1],
                                 metrics.facecoord[2][1],
                                 0, B_p, RS, MMS))
                
                @assert length(q) == 2nn^2 + 5nn
            end

            @printf "Got initial conditions: %s s\n" it
            st = @elapsed begin
                dt_scale = .01
                dt = dt_scale * 2 * lop.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))
                @printf "Running simulation with %s nodes...\n" nn
                timestep!(q, ODE_RHS_D_MMS!, dynamic_operators, dt, t_span)
            end
            @printf "Ran simulation to time %s: %s s\n\n" t_span[2] st
            
            u_end = @view q[1:Nn]
            diff_u = u_end - ue(x[:], y[:], t_span[2], MMS)
            err[iter] = sqrt(diff_u' * lop.JH * diff_u)
            @printf "error: %e\n" err[iter]
            if iter > 1
            @printf "rate: %f\n\n" log(2, err[iter - 1]/err[iter])
            end
        end
    end
end
