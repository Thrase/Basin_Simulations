using Plots
using CUDA
using CUDA.CUSPARSE
using Printf


CUDA.allowscalar(false)

function Q_DYNAMIC!(dψδ, ψδ, p, t)

    #@printf "\r\t%f" t/p.year_seconds
    
    reject_step = p.reject_step
    if reject_step[1]
        return
    end
    
    nn = p.nn
    Lw = p.Lw
    Δτ = p.vars.Δτ
    u = p.vars.u
    ge = p.vars.ge
    vf = p.vars.vf
    M = p.ops.M̃
    K = p.ops.K
    H̃ = p.ops.H̃
    JI = p.ops.JI
    RS = p.RS
    metrics = p.metrics
    ops = p.ops
    η = metrics.η
    b = p.RS.b
    

    xf1 = metrics.facecoord[1][1]
    yf1 = metrics.facecoord[2][1] 

    
    ψ  = @view ψδ[1:nn]
    δ =  @view ψδ[nn + 1 : 2nn]
    dψ = @view dψδ[1:nn]
    V = @view dψδ[nn + 1 : 2nn]

    mod_data!(δ, ge, K, vf, RS, metrics, Lw, t)

    u[:] = M \ ge

    HI = ops.HI[1]
    G = ops.G[1]
    Γ = ops.Γ[1]
    L = ops.L[1]
    sJ = metrics.sJ[1]
    
    Δτ .= - (HI * G * u + Γ * (δ ./ 2 - L * u)) ./ sJ

    for n in 1:nn
        
        ψn = ψ[n]
        bn = b[n]
        τn = Δτ[n]
        ηn = η[n]

        
        VR = abs(τn / ηn)
        VL = -VR
        Vn = V[n]
        obj_rs(V) = rateandstateQ(V, ψn, RS.σn, τn, ηn, RS.a, RS.V0)
        (Vn, _, iter) = newtbndv(obj_rs, VL, VR, Vn; ftol = 1e-14,
                                 atolx = 1e-14, rtolx = 1e-14)

        if !isfinite(Vn)
            #@printf "Reject V\n"
            flush(stdout)
            reject_step[1] = true
            return
        end

        V[n] = Vn
        
        if bn != 0
            dψ[n] = (bn * RS.V0 / RS.Dc) * (exp((RS.f0 - ψn) / bn) - abs(Vn) / RS.V0)
        else
            dψ[n] = 0
        end
        
        if !isfinite(dψ[n])
            #@printf "\nReject dψ\n"
            dψ .= 0
            reject_step[1] = true
            return
        end

    end
    
    nothing
    
end

function Q_STATIC_MMS!(p)

    nn = p.nn
    Δτ = p.Δτ
    u = p.u
    ge = p.ge
    vf = p.vf
    M = p.ops.M̃
    K = p.ops.K
    JH = p.ops.JH
    RS = p.RS
    MMS = p.MMS
    B_p = p.B_p
    metrics = p.metrics
    ops = p.ops
    η = metrics.η
    b = p.b
    δ = 0
    ys = p.year_seconds
    t = p.t_final
    
    static_data_mms!(δ, ge, K, JH, vf, MMS, B_p, RS, metrics, t)

    u[:] = M \ ge
    

end

function Q_DYNAMIC_MMS_NOROOT!(dψδ, ψδ, p, t)



    @printf "\r\t%f" t/31556952

    reject_step = p.reject_step
    if reject_step[1]
        return
    end

    nn = p.nn
    Δτ = p.Δτ
    u = p.u
    ge = p.ge
    vf = p.vf
    M = p.ops.M̃
    K = p.ops.K
    H̃ = p.ops.H̃
    JH = p.ops.JH
    RS = p.RS
    MMS = p.MMS
    B_p = p.B_p
    metrics = p.metrics
    ops = p.ops
    η = metrics.η
    b = p.b
    f1_source = p.f1_source
    

    xf1 = metrics.facecoord[1][1]
    yf1 = metrics.facecoord[2][1] 

    
    ψ  = @view ψδ[1:nn]
    δ =  @view ψδ[nn + 1 : 2nn]
    dψ = @view dψδ[1:nn]
    V = @view dψδ[nn + 1 : 2nn]

    mod_data_mms!(δ, ge, K, H̃, JH, vf, MMS, B_p, RS, metrics, t)

    u[:] = M \ ge

    HI = ops.HI[1]
    G = ops.G[1]
    Γ = ops.Γ[1]
    L = ops.L[1]
    sJ = metrics.sJ[1]
   
    #V .= 2 .* he_t(xf1, yf1, t, MMS)
    #Δτ .= - τhe(xf1, yf1, t, 1, B_p, MMS)
    Δτ .= - (HI * G * u + Γ * (δ ./ 2- L * u)) ./ sJ
    
    for n in 1:nn
        
        ψn = ψ[n]
        bn = b[n]
        τn = Δτ[n]
        ηn = η[n]

        Vn = (2 * RS.V0 * sinh(τn / (RS.σn * RS.a))) / (exp(ψn/RS.a))

        if !isfinite(Vn)
            reject_step[1] = true
            return
        end

        V[n] = Vn

        
        if bn != 0
            dψ[n] = (bn * RS.V0 / RS.Dc) * (exp((RS.f0 - ψn) / bn) - abs(V[n]) / RS.V0)
            #dψ[n] = ψe_thd(xf1[n], yf1[n], t, B_p, RS, MMS)[1]
            dψ[n] += S_rsh(xf1[n], yf1[n], t, bn, B_p, RS, MMS)[1]
        else
            dψ[n] = 0
        end
        
        if !isfinite(dψ[n])
            dψ .= 0
            reject_step[1] = true
            return
        end
        
    end
    
    nothing
    
end


function Q_DYNAMIC_MMS!(dψδ, ψδ, p, t)


    #@printf "%f\n" t
    @printf "\r\t%f" t/31556952
    #=
    reject_step = p.reject_step
    if reject_step[1]
        return
    end
    =#
    nn = p.nn
    Δτ = p.Δτ
    u = p.u
    ge = p.ge
    vf = p.vf
    M = p.ops.M̃
    K = p.ops.K
    H̃ = p.ops.H̃
    JH = p.ops.JH
    RS = p.RS
    MMS = p.MMS
    B_p = p.B_p
    metrics = p.metrics
    ops = p.ops
    η = metrics.η
    b = p.b
    f1_source = p.f1_source
    

    xf1 = metrics.facecoord[1][1]
    yf1 = metrics.facecoord[2][1] 

    
    ψ  = @view ψδ[1:nn]
    δ =  @view ψδ[nn + 1 : 2nn]
    dψ = @view dψδ[1:nn]
    V = @view dψδ[nn + 1 : 2nn]

    mod_data_mms!(δ, ge, K, H̃, JH, vf, MMS, B_p, RS, metrics, t)

    u[:] = M \ ge

    HI = ops.HI[1]
    G = ops.G[1]
    Γ = ops.Γ[1]
    L = ops.L[1]
    sJ = metrics.sJ[1]

   
    #V .= 2 .* he_t(xf1, yf1, t, MMS)
     
    #Δτ .= - τhe(xf1, yf1, t, 1, B_p, MMS)
    Δτ .= - (HI * G * u + Γ * (δ ./ 2- L * u)) ./ sJ

    for n in 1:nn
        
        ψn = ψ[n]
        bn = b[n]
        τn = Δτ[n]
        ηn = η[n]

        
        VR = abs(τn / ηn)
        VL = -VR
        Vn = V[n]
        obj_rs(V) = rateandstateQ(V, ψn, RS.σn, τn, ηn, RS.a, RS.V0)
        (Vn, _, iter) = newtbndv(obj_rs, VL, VR, Vn; ftol = 1e-14,
                                 atolx = 1e-14, rtolx = 1e-14)
        #=
        if !isfinite(Vn)
            @printf "Reject V\n"
            @show Vn
            flush(stdout)
            reject_step[1] = true
            return
        end
        =#
        V[n] = Vn
        

        if bn != 0
            dψ[n] = (bn * RS.V0 / RS.Dc) * (exp((RS.f0 - ψn) / bn) - abs(Vn) / RS.V0)
            #dψ[n] = ψe_th(xf1[n], yf1[n], t, B_p, RS, MMS)
            dψ[n] += S_rsh(xf1[n], yf1[n], t, bn, B_p, RS, MMS)
        else
            dψ[n] = 0
        end
        
        #=
        if !isfinite(dψ[n])
            @printf "\nReject dψ\n"
            dψ .= 0
            reject_step[1] = true
            return
        end
        =#

    end
    
    nothing
    
end


# timestepping rejection function
function stepcheck(_, p, _)
    if p.reject_step[1]
        p.reject_step[1] = false
        return true
    end
    return false
end

function break_con(t_now, sim_seconds, cycle_flag, cycles, num_cycles)
    
    if cycle_flag == false
        return t_now < sim_seconds
    elseif cycle_flag == true
        return cycles <= num_cycles
    end

end

function write_breaks(io)
    
    temp_io = open(io.slip_file, "a")
    writedlm(temp_io, ["BREAK"])
    close(temp_io)
    temp_io = open(io.slip_rate_file, "a")
    writedlm(temp_io, ["BREAK"])
    close(temp_io)
    temp_io = open(io.stress_file, "a")
    writedlm(temp_io, ["BREAK"])
    close(temp_io)
    temp_io = open(io.state_file, "a")
    writedlm(temp_io, ["BREAK"])
    close(temp_io)
    temp_io = open(io.u_file, "a")
    writedlm(temp_io, ["BREAK"])
    close(temp_io)
    temp_io = open(io.v_file, "a")
    writedlm(temp_io, ["BREAK"])
    close(temp_io)
    
    for file_name in io.station_names
        temp_io = open(file_name, "a")
        writedlm(temp_io, ["BREAK"])
        close(temp_io)
    end
end


function PLOTFACE(ψδ,t,i)

    if isdefined(i,:fsallast)
        year_seconds = i.p.year_seconds
        yf1 = i.p.metrics.facecoord[2][1]
        xf1 = i.p.metrics.facecoord[1][1]
        MMS = i.p.MMS
        B_p = i.p.B_p
        nn = i.p.nn
        dψV = i.fsallast
        V = @view dψV[nn .+ (1:nn)]
        plot(V, yf1, legend=false, color =:blue, yflip=true)
        plot!(2 * he_t(xf1, yf1, t, MMS), yf1, legend=false, color =:red, yflip=true)
        gui()
    end

    return false
    
end


# function for every accepted timstep with integrator stopping condition
function STOPFUN_Q(ψδ,t,i)
    
    if isdefined(i,:fsallast)

        nn = i.p.nn
        RS = i.p.RS
        τ = i.p.vars.Δτ
        t_prev = i.p.vars.t_prev
        year_seconds = i.p.year_seconds
        u_prev = i.p.vars.u_prev
        u = i.p.vars.u
        fault_coord = i.p.metrics.facecoord[2][1]
        Lw = i.p.Lw
        io = i.p.io
        pf = i.p.io.pf
        η = i.p.metrics.η
        cycles = i.p.cycles[1]
        
        dψV = i.fsallast
        ψ = @view ψδ[(1:nn)]
        δ = @view ψδ[nn .+ (1:nn)]
        V = @view dψV[nn .+ (1:nn)]
        Vmax = maximum(abs.(V))
        
        if pf[1] % 10 == 0

            #=    
            plt1 = plot(V[1:nn], fault_coord[1:nn], yflip = true, ylabel="Depth",
            xlabel="Slip-Rate", linecolor=:blue, linewidth=.1,
            legend=false)
            
            
            plt2 = plot(τ[1:nn], fault_coord[1:nn], yflip = true, ylabel="Depth",
            xlabel="Slip", linecolor=:blue, linewidth=.1,
            legend=false)
            plot(plt1, plt2, layout=2)
            gui()
            =#
            
            plot!(δ[1:nn], fault_coord[1:nn], yflip = true, ylabel="Depth",
            xlabel="Slip", linecolor=:blue, linewidth=.1,
            legend=false)
            gui()
            

            write_out(δ, V, τ, ψ, t,
                  fault_coord,
                  Lw,
                  io.station_names,
                  η)
            
            write_out_ss(δ, V, τ, ψ, t,
                         io.slip_file,
                         io.stress_file,
                         io.slip_rate_file,
                         io.state_file)
        end

        
        year_count = t/year_seconds
        
        if Vmax >= 1e-2 
            return true
        end
        
        pf[1] += 1
        u_prev .= u
        t_prev[1] = t
        
    end
    
    return false
        
end


function MMS_WAVEPROP_CPU!(dq, q, p, t)

    @printf "\r\t%f" t
    
    nn = p.nn
    fc = p.fc
    coord = p.coord
    R = p.R
    B_p = p.B_p
    MMS = p.MMS
    Λ = p.d_ops.Λ
    sJ = p.sJ
    Z̃f = p.d_ops.Z̃f
    L = p.d_ops.L
    H = p.d_ops.H
    P̃I = p.d_ops.P̃I
    JIHP = p.d_ops.JIHP
    CHAR_SOURCE = p.CHAR_SOURCE
    FORCE = p.FORCE

    u = q[1:nn^2]

    # compute all temporal derivatives
    dq .= Λ * q

    for i in 1:4
        fx = fc[1][i]
        fy = fc[2][i]
        S̃_c = sJ[i] .* CHAR_SOURCE(fx, fy, t, i, R[i], B_p, MMS)
        dq[2nn^2 + (i-1)*nn + 1 : 2nn^2 + i*nn] .+= S̃_c ./ (2*Z̃f[i])
        dq[nn^2 + 1:2nn^2] .+= L[i]' * H[i] * S̃_c ./ 2
    end
    dq[nn^2 + 1 : 2nn^2] .= JIHP * dq[nn^2 + 1 : 2nn^2]
    dq[nn^2 + 1:2nn^2] .+= P̃I * FORCE(coord[1][:], coord[2][:], t, B_p, MMS)
end


function WAVEPROP!(dq, q, p, t)
    nn = p.nn
    Λ = p.d_ops.Λ
    JIHP = p.JIHP
    dq .= Λ * q
    dq[nn^2 + 1 : 2nn^2] .= JIHP * dq[nn^2 + 1 : 2nn^2]
end


function MMS_FAULT_CPU!(dq, q, p, t)

    @printf "\r\t%f" t
    
    nn = p.nn
    R = p.R
    fc = p.fc
    coord = p.coord
    B_p = p.B_p
    RS = p.RS
    b = p.b
    MMS = p.MMS
    Λ = p.d_ops.Λ
    sJ = p.sJ
    Z̃f = p.d_ops.Z̃f
    L = p.d_ops.L
    H = p.d_ops.H
    P̃I = p.d_ops.P̃I
    JIHP = p.d_ops.JIHP
    nCnΓ1 = p.d_ops.nCnΓ1
    HIGΓL1 = p.d_ops.HIGΓL1

    τ̃f = p.τ̃f
    
    u = @view q[1:nn^2]
    v = @view q[nn^2 + 1: 2nn^2]
    û1 = @view q[2nn^2 + 1 : 2nn^2 + nn]
    ψ = @view q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]
    dv = @view dq[nn^2 + 1: 2nn^2]
    dû1 = @view dq[2nn^2 + 1 : 2nn^2 + nn]
    dψ = @view dq[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]

    # compute all temporal derivatives
    dq .= Λ * q
    
    # compute numerical traction on face 1
    #τ̃f .= sJ[1] .* τhe(fc[1][1], fc[2][1], t, 1, B_p, MMS)
    τ̃f .= HIGΓL1 * u + nCnΓ1 * û1
    #ψ .= ψe_d(fc[1][1], fc[2][1], t, B_p, RS, MMS)
    
    # Root find for RS friction
    for n in 1:nn
        
        vn = v[1 + nn * (n-1)]

        v̂_root(v̂) = rateandstateD(v̂,
                                  Z̃f[1][n],
                                  vn,
                                  sJ[1][n],
                                  ψ[n],
                                  RS.a,
                                  τ̃f[n],
                                  RS.σn,
                                  RS.V0)

        left = vn - τ̃f[n]/Z̃f[1][n]
        right = -left
        
        if left > right  
            tmp = left
            left = right
            right = tmp
        end
        
        (v̂n, _, _) = newtbndv(v̂_root, left, right, vn; ftol = 1e-14,
                              atolx = 1e-14, rtolx = 1e-14)

        if isnan(v̂n)
            #println("Not bracketing root")
        end
        
        dû1[n] = v̂n
        #dû1[n] = he_t(fc[1][1][n], fc[2][1][n], t, MMS)
        dv[1 + (n - 1)*nn] +=  H[1][n, n] * (Z̃f[1][n] * v̂n)
        dψ[n] = (b[n] .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ[n]) ./ b[n]) .- abs.(2 .* v̂n) ./ RS.V0)
    end


    # Non-fault Source injection
    for i in 2:4
        SOURCE = sJ[i] .* S_ch(fc[1][i], fc[2][i], t, i, R[i], B_p, MMS)
        dq[2nn^2 + (i-1)*nn + 1 : 2nn^2 + i*nn] .+= SOURCE ./ (2*Z̃f[i])
        dq[nn^2 + 1:2nn^2] .+= L[i]' * H[i] * SOURCE ./ 2
    end

    # psi source
    dψ .+= S_rsdh(fc[1][1], fc[2][1], b, t, B_p, RS, MMS)


    dq[nn^2 + 1:2nn^2] .= JIHP * dq[nn^2 + 1:2nn^2]
    dq[nn^2 + 1:2nn^2] .+= P̃I * Forcing_hd(coord[1][:], coord[2][:], t, B_p, MMS)


end


function FAULT_CPU!(dq, q, p, t)

    nn = p.nn
    RS = p.RS
    b = p.b
    Λ = p.d_ops.Λ
    sJ = p.sJ[1]
    Z̃f = p.d_ops.Z̃f[1]
    H = p.d_ops.H[1]
    JIHP = p.d_ops.JIHP
    nCnΓ1 = p.d_ops.nCnΓ1
    nBBCΓL1 = p.d_ops.nBBCΓL1
    τ̃f = p.τ̃f
    
    u = @view q[1:nn^2]
    v = @view q[nn^2 + 1 : 2nn^2]
    û1 = @view q[2nn^2 + 1 : 2nn^2 + nn]
    ψ = @view q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]
    dv = @view dq[nn^2 + 1 : 2nn^2]
    dû1 = @view dq[2nn^2 + 1 : 2nn^2 + nn]
    dψ = @view dq[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]

    dq .= Λ * q
    # compute numerical traction on face 1
    τ̃f .= nBBCΓL1 * u + nCnΓ1 * û1

    # Root find for RS friction
    for n in 1:nn

        vn = v[1 + nn * (n-1)]

        v̂_root(v̂) = rateandstateD(v̂,
                                  Z̃f[n],
                                  vn,
                                  sJ[n],
                                  ψ[n],
                                  RS.a,
                                  τ̃f[n],
                                  RS.σn,
                                  RS.V0)

        left = vn - τ̃f[n]/Z̃f[n]
        right = -left

        if left > right  
            tmp = left
            left = right
            right = tmp
        end
        
        (v̂n, _, _) = newtbndv(v̂_root, left, right, vn; ftol = 1e-12,
                              atolx = 1e-12, rtolx = 1e-12)

        if isnan(v̂n)
            #println("Not bracketing root")
        end
        dû1[n] = v̂n
        dv[1 + (n - 1)*nn] +=  H[n, n] * (Z̃f[n] * v̂n)
        dψ[n] = (b[n] .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ[n]) ./ b[n]) .- abs.(2 .* v̂n) ./ RS.V0)
    end
         
    dv .= JIHP * dq[nn^2 + 1:2nn^2]

end


function FAULT_GPU!(dq, q, p, t)
    
    #@printf "\r\t%f" t

    nn = p.nn
    RS = p.RS
    b = p.b
    Λ = p.Λ
    sJ = p.sJ
    Z̃f1 = p.Z̃f1
    Z̃f2 = p.Z̃f2
    Z̃f3 = p.Z̃f3
    L2 = p.L2
    L3 = p.L3
    H = p.H
    JIHP = p.JIHP
    nCnΓ1 = p.nCnΓ1
    HIGΓL1 = p.HIGΓL1
    τ̃f = p.τ̃f
    
    source2 = p.source2
    source3 = p.source3

    threads = p.threads
    blocks = p.blocks

    u = @view q[1:nn^2]
    vf = @view q[nn^2 + 1: nn : 2nn^2]
    û1 = @view q[2nn^2 + 1 : 2nn^2 + nn]
    ψ = @view q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]
    dv = @view dq[nn^2 + 1 : 2nn^2]
    dû1 = @view dq[2nn^2 + 1 : 2nn^2 + nn]
    dψ = @view dq[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]
    dvf = @view dq[nn^2 + 1: nn : 2nn^2]

   
    dq .= Λ * q

    # compute numerical traction on face 1
    τ̃f .= HIGΓL1 * u + nCnΓ1 * û1
    
    
    @cuda blocks=blocks threads=threads FAULT_PROBLEM!(dû1, dvf, vf, τ̃f, Z̃f1, H, sJ, ψ, dψ, b, RS)

    dq[2nn^2 + nn + 1 : 2nn^2 + 2nn] .+= source2 ./ (2*Z̃f2)
    dq[nn^2 + 1:2nn^2] .+= L2' * (H .* source2 ./ 2)

    dq[2nn^2 + 2nn + 1 : 2nn^2 + 3nn] .+= source3 ./ (2*Z̃f3)
    dq[nn^2 + 1:2nn^2] .+= L3' * (H .* source3 ./ 2)

    dv .= JIHP * dq[nn^2 + 1:2nn^2]

end


function FAULT_PROBLEM!(dû1, dvf, vf, τ̃f, Z̃f, H, sJ, ψ, dψ, b, RS)

    a = RS[1]
    σn = RS[2]
    V0 = RS[3]
    Dc = RS[4]
    f0 = RS[5]
    nn = RS[6]

    n = blockDim().x * (blockIdx().x - 1) + threadIdx().x  
    #@cuprintln("hello")
    if n <= nn
        
        vn = vf[n]
        Z̃n = Z̃f[n]
        sJn = sJ[n]
        ψn = ψ[n]
        τ̃n = τ̃f[n]
        bn = b[n]
        Hn = H[n]
        v̂nL = vn - τ̃n/Z̃n
        v̂nR = -v̂nL

        (fL, _) = rateandstateD_GPU(v̂nL, Z̃n, vn, sJn, ψn, a, τ̃n, σn, V0)
        (fR, _) = rateandstateD_GPU(v̂nR, Z̃n, vn, sJn, ψn, a, τ̃n, σn, V0)
        (f, df) = rateandstateD_GPU(vn, Z̃n, vn, sJn, ψn, a, τ̃n, σn, V0)

        dv̂nlr = v̂nR - v̂nL

        v̂n = vn
        
        count = 0
        for iter in 1:500

            dv̂n = -f / df
            v̂n  = v̂n + dv̂n

            if v̂n < v̂nL || v̂n > v̂nR || abs(dv̂n) / dv̂nlr < 0
                v̂n = (v̂nR + v̂nL) / 2
                dv̂n = (v̂nR - v̂nL) / 2
            end
            
            (f, df) = rateandstateD_GPU(v̂n, Z̃n, vn, sJn, ψn, a, τ̃n, σn, V0)

            if f * fL > 0
                (fL, v̂nL) = (f, v̂n)
            else
                (fR, v̂nR) = (f, v̂n)
            end
            dv̂nlr = v̂nR - v̂nL

            if abs(f) < 1e-12 && abs(dv̂n) < 1e-12 + 1e-12 * (abs(dv̂n) + abs(v̂n))
                break
            end
            count += 1
        end
        
        dû1[n] = v̂n

        dvf[n] += Hn * (Z̃n * v̂n)

        dψ[n] = (bn .* V0 ./ Dc) .* (exp.((f0 .- ψn) ./ bn) .- abs.(2 .* v̂n) ./ V0)
    end
    
    nothing

end


# bracketed newton method
function newtbndv(func, xL, xR, x; ftol = 1e-6, maxiter = 500, minchange=0,
                  atolx = 1e-4, rtolx = 1e-4)

    (fL, _) = func(xL)
    (fR, _) = func(xR)
    if fL .* fR > 0
        return (typeof(x)(NaN), typeof(x)(NaN), -maxiter)
    end

    (f, df) = func(x)
    dxlr = xR - xL

    for iter = 1:maxiter
        dx = -f / df
        x  = x + dx

        if x < xL || x > xR || abs(dx) / dxlr < minchange
            x = (xR + xL) / 2
            dx = (xR - xL) / 2
        end

        (f, df) = func(x)

        if f * fL > 0
            (fL, xL) = (f, x)
        else
            (fR, xR) = (f, x)
        end
        dxlr = xR - xL

        if abs(f) < ftol && abs(dx) < atolx + rtolx * (abs(dx) + abs(x))
            return (x, f, iter)
        end
    end
    return (x, f, -maxiter)
end


# Dynamic roofinding problem on the fault
function rateandstateD(v̂, z̃, v, sJ, ψ, a, τ̃, σn, V0)
    Y = (1 / (2 * V0)) * exp(ψ / a)
    f = a * asinh(2v̂ * Y)
    dfdv̂  = a * (1 / sqrt(1 + (2v̂ * Y)^2)) * 2Y
    g = sJ * σn * f + τ̃ + z̃*(v̂ - v)
    dgdv̂ = z̃ + sJ * σn * dfdv̂
    
    return (g, dgdv̂)
end


# Dynamic roofinding problem on the fault
function rateandstateD_GPU(v̂, z̃, v, sJ, ψ, a, τ̃, σn, V0)
    Y = (1 / (2 * V0)) * exp(ψ / a)
    f = a * CUDA.asinh(2v̂ * Y)
    dfdv̂  = a * (1 / sqrt(1 + (2v̂ * Y)^2)) * 2Y
    g = sJ * σn * f + τ̃ + z̃*(v̂ - v)
    dgdv̂ = z̃ + sJ * σn * dfdv̂
    
    return (g, dgdv̂)
end


function timestep_write!(q, f!, p, dt, (t0, t1), Δq = similar(q), Δq2 = similar(q))
    T = eltype(q)
    
    nn = p.nn
    Lw = p.Lw
    fc = p.fc
    pf = p.io.pf
    τ̃f = p.τ̃f
    sJ = p.sJ
    Z̃f = p. Z̃f1
    io = p.io
    v̂ = p.v̂
    d_to_s = p.d_to_s
    RS = p.RS_cpu
    vf = @view q[nn^2 + 1: nn : 2nn^2]
    v = @view q[nn^2 + 1 : 2nn^2]
    u = @view q[1 : nn^2]
    ûf = @view q[2nn^2 + 1 : 2nn^2 + nn]
    ψ = @view q[2nn^2 + 4nn + 1 : 2nn^2 + 5*nn]
    
    RKA = [
        T(0),
        T(-567301805773 // 1357537059087),
        T(-2404267990393 // 2016746695238),
        T(-3550918686646 // 2091501179385),
        T(-1275806237668 // 842570457699),
    ]

    RKB = [
        T(1432997174477 // 9575080441755),
        T(5161836677717 // 13612068292357),
        T(1720146321549 // 2090206949498),
        T(3134564353537 // 4481467310338),
        T(2277821191437 // 14882151754819),
    ]

    RKC = [
        T(0),
        T(1432997174477 // 9575080441755),
        T(2526269341429 // 6820363962896),
        T(2006345519317 // 3224310063776),
        T(2802321613138 // 2924317926251),
    ]
    

    nstep = ceil(Int, (t1 - t0) / dt)
    dt = (t1 - t0) / nstep

    pf[1] = .1
    pf[2] = .5
    

    fill!(Δq, 0)
    fill!(Δq2, 0)
    for step in 1:nstep
        t = t0 + (step - 1) * dt
        for s in 1:length(RKA)
            f!(Δq2, q, p, t + RKC[s] * dt)
            v̂ .= Δq2[2nn^2 + 1 : 2nn^2 + nn]
            Δq .+= Δq2
            q .+= (RKB[s] * dt) .* Δq
            Δq .*= RKA[s % length(RKA) + 1]
        end
        
       

        if step == ceil(Int, pf[1]/dt)
            
            v̂_cpu = Array(v̂)
            δ = Array(2ûf)
            τ̂ = Array(-τ̃f ./ sJ .- Z̃f .* (v̂ - vf) ./ sJ)
            ψ_cpu = Array(ψ)
            

            if any(isnan, v̂_cpu)
                @printf "nan from dynamic rootfinder"
                exit()
            end
        
            write_out(δ,
                      2v̂_cpu,
                      τ̂,
                      ψ_cpu,
                      t,
                      fc,
                      Lw,
                      io.station_names)

            write_out_ss(δ,
                         2v̂_cpu,
                         τ̂,
                         ψ_cpu,
                         t,
                         io.slip_file,
                         io.stress_file,
                         io.slip_rate_file,
                         io.state_file)
            
            plot!(δ, fc, yflip = true, ylabel="Depth",
                  xlabel="Slip-Rate", linecolor=:red, linewidth=.1,
                  legend=false)
            gui()
            

            if io.vp == 1
                write_out_uv(Array(u), Array(v), nn, nn, io.u_file, io.v_file)
            end

            pf[1] +=.1
        end
    
        if (2 * maximum(v̂)) < d_to_s
            return t
        end

    end

    nothing

end


function timestep!(q, f!, p, dt, (t0, t1), Δq = similar(q), Δq2 = similar(q))
    T = eltype(q)

    RKA = (
        T(0),
        T(-567301805773 // 1357537059087),
        T(-2404267990393 // 2016746695238),
        T(-3550918686646 // 2091501179385),
        T(-1275806237668 // 842570457699),
    )

    RKB = (
        T(1432997174477 // 9575080441755),
        T(5161836677717 // 13612068292357),
        T(1720146321549 // 2090206949498),
        T(3134564353537 // 4481467310338),
        T(2277821191437 // 14882151754819),
    )

    RKC = (
        T(0),
        T(1432997174477 // 9575080441755),
        T(2526269341429 // 6820363962896),
        T(2006345519317 // 3224310063776),
        T(2802321613138 // 2924317926251),
    )

    nstep = ceil(Int, (t1 - t0) / dt)
    dt = (t1 - t0) / nstep

    fill!(Δq, 0)
    fill!(Δq2, 0)
    for step in 1:nstep
        t = t0 + (step - 1) * dt
        for s in 1:length(RKA)
            f!(Δq2, q, p, t + RKC[s] * dt)
            Δq .+= Δq2
            q .+= RKB[s] * dt * Δq
            Δq .*= RKA[s % length(RKA) + 1]
        end
    end

    nothing
    
end


function euler!(q, f!, p, dt, t_span, Δq = similar(q))
    
    nstep = ceil(Int, (t_span[2] - t_span[1]) / dt)
    dt = (t_span[2] - t_span[1]) / nstep

    for step in 1:nstep
        t = t_span[1] + (step - 1) * dt
        fill!(Δq, 0)
        f!(Δq, q, p, t)
        q .+= dt * Δq
    end

end


# Quasi-Dynamic rootfinding problem on the fault
function rateandstateQ(V, ψ, σn, τn, ηn, a, V0)
    
    Y = (1 ./ (2 .* V0)) .* exp.(ψ ./ a)
    f = a .* asinh.(V .* Y)
    dfdV  = a .* (1 ./ sqrt.(1 + (V .* Y).^2)) .* Y
    g = σn .* f + ηn .* V - τn
    dgdV = σn .* dfdV + ηn
    (g, dgdV)
end
