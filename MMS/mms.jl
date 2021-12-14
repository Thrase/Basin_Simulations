using Plots
using Printf
using CUDA
using CUDA.CUSPARSE

include("../physical_params.jl")
include("mms_funcs.jl")
include("../domain.jl")
include("../numerical.jl")
include("../solvers.jl")

function refine(ps, ns, t_span, Lw, D, B_p, RS, R, MMS, test_type)
    

    #xt, yt = transforms_e(Lw, .1, .05)
    # expand to (0,Lw) × (0, Lw)
    (x1, x2, x3, x4) = (0, 1, 0, 1)
    (y1, y2, y3, y4) = (0, 0, 1, 1)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)
    
    for p in ps
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

            ot = @elapsed begin
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



                
            end
        end
    end
end
