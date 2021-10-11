using Plots
using Printf

include("../physical_params.jl")
include("mms_funcs.jl")
include("../domain.jl")
include("../numerical.jl")
include("../solvers.jl")

function refine(ps, ns, t_span, Lw, D, B_p, RS, R, MMS)
    

    #xt, yt = transforms_e(Lw, .1, .05)
    # expand to (0,Lw) × (0, Lw)
    (x1, x2, x3, x4) = (0, 1, 0, 1)
    (y1, y2, y3, y4) = (0, 0, 1, 1)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)
    
    for p in ps
        err1 = Vector{Float64}(undef, length(ns))
        err2 = Vector{Float64}(undef, length(ns))
        err3 = Vector{Float64}(undef, length(ns))
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
            faces = [0 2 3 4]
            ot = @elapsed begin

                d_ops = operators_dynamic(p, N, N, B_p, μ, ρ, R, faces, metrics, LFtoB)

                b = b_fun(metrics.facecoord[2][1], RS)

                τ̃f = Array{Float64, 1}(undef, nn)
                vf = Array{Float64, 1}(undef, nn)
                v̂_fric = Array{Float64, 1}(undef, nn)
                
                block_solve_operators = (d_ops = d_ops,
                                         nn = nn,
                                         fc = metrics.facecoord,
                                         coord = metrics.coord,
                                         R = R,
                                         B_p = B_p,
                                         MMS = MMS,
                                         RS = RS,
                                         b = b,
                                         sJ = metrics.sJ,
                                         τ̃f = τ̃f,
                                         vf = vf,
                                         v̂_fric = v̂_fric,
                                         CHAR_SOURCE = S_c,
                                         STATE_SOURCE = S_rs,
                                         FORCE = Forcing)

            end

            @printf "Got Operators: %s s\n" ot

            it = @elapsed begin
                u0 = ue(x[:], y[:], 0.0, MMS)
                v0 = ue_t(x[:], y[:], 0.0, MMS)
                q1 = [u0;v0]
                for i in 1:4
                    q1 = vcat(q1, d_ops.L[i]*u0)
                end
                q1 = vcat(q1, ψe(metrics.facecoord[1][1],
                                metrics.facecoord[2][1],
                                 0, B_p, RS, MMS))

                q2 = deepcopy(q1)
                q3 = deepcopy(q1)
                @assert length(q1) == 2nn^2 + 5nn
            end

            @printf "Got initial conditions: %s s\n" it
            @printf "Running simulations with %s nodes...\n" nn
            @printf "\n___________________________________\n"
            
            dt_scale = .0001
            dt = dt_scale * 2 * d_ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))
            
            st2 = @elapsed begin
                timestep!(q2, ODE_RHS_BLOCK_CPU_MMS_FAULT!, block_solve_operators, dt, t_span)
            end

            @printf "Ran block fault simulation to time %s: %s s \n\n" t_span[2] st2
            
            u_end2 = @view q2[1:Nn]
            diff_u2 = u_end2 - ue(x[:], y[:], t_span[2], MMS)
            err2[iter] = sqrt(diff_u2' * d_ops.JH * diff_u2)

            @printf "block fault error: %e\n\n" err2[iter]

            if iter > 1
                @printf "block fault rate: %f\n" log(2, err2[iter - 1]/err2[iter])
            end
         
            @printf "___________________________________\n\n"
        end
    end

end
