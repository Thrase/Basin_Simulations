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
                
                # sources for GPU
                dt_scale = .0001
                dt = dt_scale * 2 * d_ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))
                nstep = ceil(Int, (t_span[2] - t_span[1]) / dt)
                dt = (t_span[2] - t_span[1]) / nstep

                volume_source = Array{Float64,2}(undef, nn^2, nstep)
                source_1 = Array{Float64, 2}(undef, nn, nstep)
                source_2 = Array{Float64, 2}(undef, nn, nstep)
                source_3 = Array{Float64, 2}(undef, nn, nstep)
                source_4 = Array{Float64, 2}(undef, nn, nstep)
                
                x = metrics.coord[1]
                y = metrics.coord[2]
                fc = metrics.facecoord

                for step in 1:nstep
                    time = t_span[1] + (step - 1) * dt
                    volume_source[:, step] .= d_ops.P̃I * Forcing(x[:], y[:], time, B_p, MMS)
                    source_1[:, step] .= S_rs(fc[1][1], fc[2][1], b, time, B_p, RS, MMS)
                    source_2[:, step] .= S_c(fc[1][2], fc[2][2], time, 2, R[2], B_p, MMS)
                    source_3[:, step] .= S_c(fc[1][3], fc[2][3], time, 3, R[3], B_p, MMS)
                    source_4[:, step] .= S_c(fc[1][4], fc[2][4], time, 4, R[4], B_p, MMS)
                end

                times = t_span[1]: dt : t_span[2]
                #@show times
                #@show length(times)
                #quit()
                #Volume_source = Forcing

                GPU_operators = (nn = nn,
                                 Λ = d_ops.Λ,
                                 Z̃f = d_ops.Z̃f,
                                 L = d_ops.L,
                                 H = d_ops.H,
                                 P̃I = d_ops.P̃I,
                                 JIHP = d_ops.JIHP,
                                 nCnΓ1 = d_ops.nCnΓ1,
                                 nBBCΓL1 = d_ops.nBBCΓL1,
                                 sJ = metrics.sJ,
                                 RS = RS,
                                 b = b,
                                 vf = vf,
                                 τ̃f = τ̃f,
                                 v̂_fric = v̂_fric,
                                 source_1 = source_1,
                                 source_2 = source_2,
                                 source_3 = source_3,
                                 source_4 = source_4,
                                 volume_source = volume_source)
                


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
                @assert length(q1) == 2nn^2 + 5nn

            end

            @printf "Got initial conditions: %s s\n" it
            @printf "Running simulations with %s nodes...\n" nn
            @printf "\n___________________________________\n"
            
            
            st1 = @elapsed begin
               ODE_RHS_GPU_FAULT!(q1, GPU_operators, dt, t_span)
            end
            
             @printf "Ran GPU to time %s in: %s s \n\n" t_span[2] st1

            st2 = @elapsed begin
                euler!(q2, ODE_RHS_BLOCK_CPU_MMS_FAULT!, block_solve_operators, dt, t_span)
            end

            @printf "Ran CPU to time %s in: %s s \n\n" t_span[2] st2
            
            u_end1 = @view Array(q1)[1:Nn]
            diff_u1 = u_end1 - ue(x[:], y[:], t_span[2], MMS)
            err1[iter] = sqrt(diff_u1' * d_ops.JH * diff_u1)


            contour(x[:,1], y[1,:],
                    (reshape(u_end1, (nn, nn)) .- ue(x, y, t_span[2], MMS))',
                    xlabel="off fault", ylabel="depth", fill=true, yflip=true)

            gui()
            sleep(10)

            u_end2 = @view q2[1:Nn]
            diff_u2 = u_end2 - ue(x[:], y[:], t_span[2], MMS)
            err2[iter] = sqrt(diff_u2' * d_ops.JH * diff_u2)
            
            @printf "GPU error: %e\n\n" err1[iter]
            @printf "CPU error: %e\n\n" err2[iter]

            if iter > 1
                @printf "GPU rate: %f\n" log(2, err1[iter - 1]/err1[iter])
                @printf "CPU rate: %f\n" log(2, err2[iter - 1]/err2[iter])
            end
         
            @printf "___________________________________\n\n"
        end
    end

end
