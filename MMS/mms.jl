using Plots
using Printf
using CUDA
using CUDA.CUSPARSE
using OrdinaryDiffEq

include("../physical_params.jl")
include("mms_funcs.jl")
include("../domain.jl")
include("../numerical.jl")
include("../solvers.jl")

function refine(ps, ns, t_span, Lw, D, B_p, RS, R, MMS, test_type)
    

    #xt, yt = transforms_e(Lw, .1, .05)
    # expand to (0,Lw) × (0, Lw)
    (x1, x2, x3, x4) = (-Lw, Lw, -Lw, Lw)
    (y1, y2, y3, y4) = (-Lw, -Lw, Lw, Lw)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)
    
    for p in ps
        err = Vector{Float64}(undef, length(ns))
        err3 = Vector{Float64}(undef, length(ns))
        err4 = Vector{Float64}(undef, length(ns))
        err5 = Vector{Float64}(undef, length(ns))
        for (iter, N) in enumerate(ns)
            
            nn = N + 1
            Nn = nn^2
            mt = @elapsed begin
                metrics = create_metrics(N, N, B_p, μ, ρ, xt, yt)
            end

            @printf "Got metrics: %s s\n" mt
       
            x = metrics.coord[1]
            y = metrics.coord[2]

                faces_fault = [0 2 3 4]
                @time d_ops = operators(p, N, N, μ, ρ, R, B_p, faces_fault, metrics)
                
                b = repeat([.02], nn)
                τ̃f = Array{Float64, 1}(undef, nn)
                vf = Array{Float64, 1}(undef, nn)
                v̂_fric = Array{Float64, 1}(undef, nn)
                
                cpu_operators = (d_ops = d_ops,
                                 nn = nn,
                                 R = R,
                                 fc = metrics.facecoord,
                                 coord = metrics.coord,
                                 B_p = B_p,
                                 MMS = MMS,
                                 RS = RS,
                                 b = b,
                                 sJ = metrics.sJ,
                                 τ̃f = τ̃f,
                                 CHAR_SOURCE = S_c,
                                 STATE_SOURCE = S_rs,
                                 FORCE = Forcing)
                

            # Dynamic MMS
            if test_type == 1

                    threads = 512
                    GS = 0.0
                    GS += (length(d_ops.Λ.nzval) * 8)/1e9
                    GS += length(metrics.sJ[1]) * 8/1e9
                    GS += length(d_ops.Z̃f[1]) * 8/1e9
                    GS += length(b) * 8/1e9
                    GS += length(d_ops.L[1].nzval) * 8/1e9
                    GS += length(d_ops.H[1].nzval) * 8/1e9
                    GS += length(d_ops.JIHP.nzval) * 8/1e9
                    GS += length(d_ops.nCnΓ1.nzval) * 8/1e9
                    GS += length(d_ops.nBBCΓL1.nzval) * 8/1e9
                    
                    

                    #@printf "Estimated Gigabytes allocating to the GPU %f\n" GS

                    #quit()
                    ot = @elapsed begin
                                    
                    GPU_operators = (nn = nn,
                                     threads = threads,
                                     blocks = cld(nn, threads),
                                     Λ = CuSparseMatrixCSC(d_ops.Λ),
                                     sJ = CuArray(metrics.sJ[1]),
                                     Z̃f = CuArray(d_ops.Z̃f[1]),
                                     L = CuSparseMatrixCSC(d_ops.L[1]),
                                     H = CuArray(diag(d_ops.H[1])),
                                     JIHP = CuSparseMatrixCSC(d_ops.JIHP),
                                     nCnΓ1 = CuSparseMatrixCSC(d_ops.nCnΓ1),
                                     nBBCΓL1 = CuSparseMatrixCSC(d_ops.nBBCΓL1),
                                     RS = CuArray([RS.a, RS.σn, RS.V0, RS.Dc, RS.f0, nn]),
                                     b = CuArray(b),
                                     τ̃f = CuArray(zeros(nn)))
                    
                    
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

                    q3 = CuArray(deepcopy(q1))
                    q4 = deepcopy(q1)
                    q5 = deepcopy(q1)
                    @assert length(q1) == 2nn^2 + 5nn

                end
                

                dt_scale = .0001
                dt = dt_scale * 2 * d_ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))

                @printf "Got initial conditions: %s s\n" it
                @printf "Running simulations with %s nodes...\n" nn
                @printf "\n___________________________________\n"
                
                
                st3 = @elapsed begin
                    timestep!(q3, FAULT_GPU!, GPU_operators, dt, t_span)
                end
                

                @printf "Ran GPU to time %s in: %s s \n\n" t_span[2] st3

                st4 = @elapsed begin
                    timestep!(q4, MMS_FAULT_CPU!, cpu_operators, dt, t_span)
                end
                
                #@printf "Ran CPU MMS to time %s in: %s s \n\n" t_span[2] st4
                
                
                st5 = @elapsed begin
                    timestep!(q5, FAULT_CPU!, cpu_operators, dt, t_span)
                end

                @printf "Ran CPU to time %s in: %s s \n\n" t_span[2] st5
                
                
                u_end3 = @view Array(q3)[1:Nn]
                u_end5 = @view q5[1:Nn]


                @printf "L2 error displacements between CPU and GPU: %e\n\n" norm(u_end5 - u_end3)
                

                @printf "___________________________________\n\n"

            #quasi dynamic MMS
            else
                #=
                year_seconds = 31556952
                
                ψ0 = ψe_2(metrics.facecoord[2][1], 0, B_p, RS, MMS)
                u0 = h_e(x[:], y[:], 0, MMS)
                δ0 = 2 * d_ops.L[1] * u0

                #=
                for t in 0:year_seconds: 100 * year_seconds
                    contour(x[:, 1], y[1, :],
                            h_e(x[:], y[:], t, MMS),
                            fill=true,
                            yflip=true)
                    sleep(.1)
                    gui()
                end
                =#
                
                ψδ = [ψ0 ; δ0]
                vars = (Δτ = zeros(nn),
                        τ = zeros(nn),
                        u = u0,
                        ge = zeros(nn^2),
                        vf = zeros(nn))
                static_params = (reject_step = [false],
                                 Lw = Lw,
                                 nn = nn,
                                 vars = vars,
                                 ops = d_ops,
                                 metrics = metrics,
                                 RS = RS,
                                 b = b,
                                 MMS = MMS,
                                 B_p = B_p)

                t_span = (0.0, 50 * year_seconds)
                prob = ODEProblem(Q_DYNAMIC_MMS!, ψδ, t_span, static_params)
                sol = solve(prob, Tsit5(); isoutofdomain=stepcheck, dt=1,
                            atol = 1e-12, rtol = 1e-12, save_everystep=true,
                            internalnorm=(x, _)->norm(x, Inf))
                =#

                u = zeros(nn^2)
                ge = zeros(nn^2)
                vf = zeros(nn)

                params = (u = u,
                          ge = ge,
                          vf = vf,
                          M = d_ops.M̃,
                          K = d_ops.K,
                          H̃ = d_ops.H̃,
                          MMS = MMS,
                          B_p = B_p,
                          metrics = metrics)
                
                u1 = Pe(metrics.facecoord[1][1],
                        metrics.facecoord[2][1],
                        0,
                        MMS)

                t_span = (0, 100.5)
                
                prob = ODEProblem(POISSON_MMS!, u1, t_span, params)
                sol = solve(prob, Tsit5();
                            atol = 1e-12,
                            rtol = 1e-12,
                            internalnorm=(x,_)->norm(x, Inf))
                
                
                diff =  params.u[:] .- Pe(x[:], y[:], t_span[2], MMS)

                err[iter] = sqrt(diff' * d_ops.JH * diff)

                
                plt1 = contour(x[:, 1], y[1, :],
                               (reshape(u, (nn, nn)) .- Pe(x, y, t_span[2], MMS))',
                               title = "error", fill=true)
                plt2 = contour(x[:, 1], y[1, :], title = "exact",
                               Pe(x, y, t_span[2], MMS)' , fill = true)
                plt3 = contour(x[:, 1], y[1, :], title = "forcing",
                               P_FORCE(x, y, t_span[2], B_p, MMS)', fill=true)
                plt4 = contour(x[:, 1], y[1, :], title = "numerical",
                               reshape(u, (nn,nn))', fill=true)
                
                plot(plt1, plt2, plt3, plt4, layout=4)
                gui()
                
                @printf "\n\nerror with manufactured solution: %e\n\n" err[iter]
                if iter > 1
                    @printf "rate: %f\n\n" log(2, err[iter - 1]/err[iter])
                end

            end
        end
    end
end
