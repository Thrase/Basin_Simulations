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

                d_op = operators_dynamic(p, N, N, B_p, μ, ρ, R, faces, metrics, LFtoB)

                display(d_op.Λ)
                quit()
                b = b_fun(metrics.facecoord[2][1], RS)
                
                block_solve_operators = (nn = nn,
                                         fc = metrics.facecoord,
                                         coord = metrics.coord,
                                         R = R,
                                         B_p = B_p,
                                         MMS = MMS,
                                         Λ = d_op.Λ,
                                         sJ = metrics.sJ,
                                         Z̃ = d_op.Z̃f,
                                         L = d_op.L,
                                         H = d_op.H,
                                         P̃I = d_op.P̃I,
                                         JIHP = d_op.JIHP,
                                         CHAR_SOURCE = S_c,
                                         FORCE = Forcing)

            end

            @printf "Got Operators: %s s\n" ot

            it = @elapsed begin
                u0 = ue(x[:], y[:], 0.0, MMS)
                v0 = ue_t(x[:], y[:], 0.0, MMS)
                q1 = [u0;v0]
                for i in 1:4
                    q1 = vcat(q1, d_op.L[i]*u0)
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
            
            dt_scale = .01
            dt = dt_scale * 2 * d_op.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))
            
            st2 = @elapsed begin
                timestep!(q2, ODE_RHS_BLOCK_CPU_FAULT!, block_solve_operators, dt, t_span)
            end

            @printf "Ran block simulation 2N store to time %s: %s s \n\n" t_span[2] st2
            
            u_end2 = @view q2[1:Nn]
            diff_u2 = u_end2 - ue(x[:], y[:], t_span[2], MMS)
            err2[iter] = sqrt(diff_u2' * d_op.JH * diff_u2)

            @printf "block 2n error: %e\n\n" err2[iter]

            if iter > 1
                @printf "block 2n rate: %f\n" log(2, err2[iter - 1]/err2[iter])
            end
         
            @printf "___________________________________\n\n"
        end
    end

end
