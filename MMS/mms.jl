using Plots
using Printf
using CUDA
using CUDA.CUSPARSE
using OrdinaryDiffEq
using ODE

include("../physical_params.jl")
include("mms_funcs.jl")
include("../domain.jl")
include("../numerical.jl")
include("../solvers.jl")

function refine(ps, ns, Lw, D, B_p, RS, R, MMS)
    
    year_seconds = 31556952
    #xt, yt = transforms_e(Lw, .1, .05)
    # expand to (0,Lw) × (0, Lw)
    (x1, x2, x3, x4) = (0, Lw, 0, Lw)
    (y1, y2, y3, y4) = (0, 0, Lw, Lw)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)
    
    for p in ps

        @printf "Operator order: %d\n\n" p
        err = Vector{Float64}(undef, length(ns))
        err3 = Vector{Float64}(undef, length(ns))
        err4 = Vector{Float64}(undef, length(ns))
        err5 = Vector{Float64}(undef, length(ns))
        for (iter, N) in enumerate(ns)
            
            nn = N + 1

            @printf "\tNodes per Dimension: %d\n\n" nn
            
            Nn = nn^2
            mt = @elapsed begin
                metrics = create_metrics(N, N, B_p, μ, ρ, xt, yt)
            end

            #@printf "Got metrics: %s s\n" mt
            facecoord = metrics.facecoord
            x = metrics.coord[1]
            y = metrics.coord[2]
            #=
            for time in 0:.1:3
            #@show minimum(Pe(x,v y, 1.1, MMS))
                contour(x[:, 1], y[1, :],
                        Pe(x, y, time, MMS),
                        fill = true, yflip=true, title = "μ")
                gui()
                sleep(.1)
            end
            
            quit()
            =#
            faces_fault = [0 2 3 4]
            d_ops = operators(p, N, N, μ, ρ, R, B_p, faces_fault, metrics)
            
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
                #=
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
                
                =#
            end
            #@printf "Got Operators: %s s\n" ot
            #=
            it = @elapsed begin
                u0 = he(x[:], y[:], 0.0, MMS)
                v0 = he_t(x[:], y[:], 0.0, MMS)
                q1 = [u0;v0]
                for i in 1:4
                    q1 = vcat(q1, d_ops.L[i]*u0)
                end
                q1 = vcat(q1, ψe_2(metrics.facecoord[1][1],
                                   metrics.facecoord[2][1],
                                   0, B_p, RS, MMS))

                #                    q3 = CuArray(deepcopy(q1))
                q4 = deepcopy(q1)
                q5 = deepcopy(q1)
                @assert length(q1) == 2nn^2 + 5nn

            end
            

            dt_scale = 1
            dt = dt_scale * 2 * d_ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))
            t_span = (0, .1)

            #@printf "Got initial conditions: %s s\n" it
            
            
            
            #=
            st3 = @elapsed begin
            timestep!(q3, FAULT_GPU!, GPU_operators, dt, t_span)
            end
            =#

            #@printf "Ran GPU to time %s in: %s s \n\n" t_span[2] st3
            
            st4 = @elapsed begin
                timestep!(q4, MMS_FAULT_CPU!, cpu_operators, dt, t_span)
            end
            
            #@printf "Ran CPU MMS to time %s in: %s s \n\n" t_span[2] st4
            
            #=
            st5 = @elapsed begin
            timestep!(q5, FAULT_CPU!, cpu_operators, dt, t_span)
            end
            =#

            x = metrics.coord[1]
            y = metrics.coord[2]
            
            u_end4 = @view q4[1:Nn]
            diff_u4 = u_end4 - he(x[:], y[:], t_span[2], MMS)
            err4[iter] = sqrt(diff_u4' * d_ops.JH * diff_u4)
            

            #@printf "Ran CPU to time %s in: %s s \n\n" t_span[2] st5
            
            
            #u_end3 = @view Array(q3)[1:Nn]
            #u_end5 = @view q5[1:Nn]

            @printf "\t\tdynamic error with MS: %e\n" err4[iter]
            if iter > 1
                @printf "\t\tdynamic rate: %f\n" log(2, err4[iter - 1]/err4[iter])
            end
            

            #@printf "L2 error displacements between CPU and GPU: %e\n\n" norm(u_end5 - u_end3)
            

            =#

            
          
            
            xf1 = metrics.facecoord[1][1]
            yf1 = metrics.facecoord[2][1]

            
            #=
            #for time in 0:year_seconds:34*year_seconds
                plot(ψe_t(xf1, yf1, time, B_p, RS, MMS), yf1, yflip=true, legend=false)
                gui()
                @show time/year_seconds
            #end
            #for time in 34*year_seconds:86400:34.99999*year_seconds
                plot(ψe_t(xf1, yf1, time, B_p, RS, MMS), yf1, yflip=true, legend=false)
                gui()
                @show time/year_seconds
            #end
            
            for time in 34.99999*year_seconds:1:35*year_seconds + 100
                plot(ψe_t(xf1, yf1, time, B_p, RS, MMS), yf1, yflip=true, legend=false)
                gui()
                sleep(.1)
                @show time/year_seconds
            end
            #=
            for time in 35*year_seconds + 100:86400:37 * year_seconds
                plot(ψe_t(xf1, yf1, time, B_p, RS, MMS), yf1, yflip=true, legend=false)
                gui()
                @show time/year_seconds
            end
            for time in 37 * year_seconds:year_seconds:70 * year_seconds
                plot(ψe_t(xf1, yf1, time, B_p, RS, MMS), yf1, yflip=true, legend=false)
                gui()
                @show time/year_seconds
            end
            =#
            =#
            #@show minimum(τPe(xf1, yf1, 0, 1, B_p, MMS))
            
            u = zeros(nn^2)
            ge = zeros(nn^2)
            vf = zeros(nn)

            t_final = 40 * year_seconds
            t_begin = 0#30 * year_seconds
            params = (t_final = t_final,
                      year_seconds = year_seconds,
                      reject_step = [false],
                      nn = nn,
                      Δτ = zeros(nn),
                      u = u,
                      ge = ge,
                      vf = vf,
                      RS = RS,
                      MMS = MMS,
                      B_p = B_p,
                      ops = d_ops,
                      b = b,
                      metrics = metrics,
                      counter = [0])
            
            δ = 2 * he(xf1,
                       yf1,
                       t_begin,
                       MMS)

            ψ = ψe(xf1,
                     yf1,
                     t_begin,
                     B_p,
                     RS,
                     MMS)

            ψδ = [ψ ; δ]

            
            t_span = (t_begin, t_final)
            
            prob = ODEProblem(Q_DYNAMIC_MMS!, ψδ, t_span, params)
            plotter = DiscreteCallback(PLOTFACE, terminate!)
            sol = solve(prob, RK4();
                        isoutofdomain=stepcheck,
                        atol = 1e-12,
                        rtol = 1e-12,
                        internalnorm=(x,_)->norm(x, Inf))
            #callback=plotter)
            
            #@show sol.t[end]
            diff = params.u[:] .- he(x[:], y[:], sol.t[end], MMS)

            err[iter] = sqrt(diff' * d_ops.JH * diff)


            #=
            plt1 = contour(x[:, 1], y[1, :],
                           (reshape(u, (nn, nn)) .- he(x, y, t_span[2], MMS))',
                           title = "error", fill=true, yflip=true)
            plt2 = contour(x[:, 1], y[1, :],
                           he(x, y, t_span[2], MMS)',
                           fill = true, yflip=true, title = "exact")
            plt3 = contour(x[:, 1], y[1, :], 
                           h_FORCE(x, y, t_span[2], B_p, MMS)',
                           fill=true, yflip=true, title = "forcing")
            plt4 = contour(x[:, 1], y[1, :], 
                           reshape(u, (nn,nn))',
                           fill=true, yflip=true, title = "numerical")
            
            plot(plt1, plt2, plt3, plt4, layout=4)
            
            gui()
            =#
            
            
            plt5 = plot((d_ops.L[1] * u - he(xf1, yf1, sol.t[end], MMS)), yf1,
                        yflip = true, title = "face 1 error", legend=false)
            
            plt6 = plot(he(xf1, yf1, sol.t[end], MMS), yf1,
                        yflip = true, title = "face 1 exact", legend=false, xlims=(0, 1.1))
            
            plt7 = plot(d_ops.L[1] * u, yf1,
                        yflip = true, title = "face 1 numerical", legend=false, xlims=(0, 1.1))
            plot(plt5, plt6, plt7, layout=3)
            gui()
            

            #=
            plt1 = contour(x[:, 1], y[1, :],
                           (reshape(u, (nn, nn)) .- Pe(x, y, t_span[2], MMS))',
                           title = "error", fill=true, yflip=true)
            plt2 = contour(x[:, 1], y[1, :],
                           Pe(x, y, t_span[2], MMS)',
                           fill = true, yflip=true, title = "exact")
            plt3 = contour(x[:, 1], y[1, :], 
                           P_FORCE(x, y, t_span[2], B_p, MMS)',
                           fill=true, yflip=true, title = "forcing")
            plt4 = contour(x[:, 1], y[1, :], 
                           reshape(u, (nn,nn))',
                           fill=true, yflip=true, title = "numerical")
            
            plot(plt1, plt2, plt3, plt4, layout=4)
            gui()
            =#

            
            #=
            plt5 = plot((d_ops.L[face_view] * u - Pe(xb, yb, time, MMS)), yb,
                        yflip = false, title = string("face", face_view,"  error"), legend=false)
            
            plt6 = plot(xb, Pe(xb, yb, time, MMS),
                        yflip = false,  title = string("face", face_view,"  exact"), legend=false)
            
            plt7 = plot(xb, d_ops.L[face_view] * u,
                        yflip = false, title = string("face", face_view,"  numerical"), legend=false)

            plot(plt5, plt6, plt7, layout=3)
            gui()
            =#
            
            @printf "\n\t\tquasi-dynamic error with MS: %e\n" err[iter]
            if iter > 1
                @printf "\t\tquasi-dynamic rate: %f\n" log(2, err[iter - 1]/err[iter])
            end


            @printf "\t___________________________________\n\n"
            
        end
    end
end
