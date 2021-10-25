using Plots
using CUDA
using CUDA.CUSPARSE

CUDA.allowscalar(false)

function MMS_WAVEPROP_CPU!(dq, q, p, t)

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

    nn = p.nn
    fc = p.fc
    coord = p.coord
    R = p.R
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
    nBBCΓL1 = p.d_ops.nBBCΓL1
    CHAR_SOURCE = p.CHAR_SOURCE
    STATE_SOURCE = p.STATE_SOURCE
    FORCE = p.FORCE
    vf = p.vf
    τ̃f = p.τ̃f
    v̂_fric = p.v̂_fric
    
    u = @view q[1:nn^2]
    û1 = @view q[2nn^2 + 1 : 2nn^2 + nn]
    ψ = @view q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]
    
    # compute all temporal derivatives
    dq .= Λ * q
    # get velocity on fault
    vf .= L[1] * q[nn^2 + 1 : 2nn^2]
    # compute numerical traction on face 1
    τ̃f .= nBBCΓL1 * u + nCnΓ1 * û1

    # Root find for RS friction
    for n in 1:nn
        
        v̂_root(v̂) = rateandstateD(v̂,
                                  Z̃f[1][n],
                                  vf[n],
                                  sJ[1][n],
                                  ψ[n],
                                  RS.a,
                                  τ̃f[n],
                                  RS.σn,
                                  RS.V0)

        left = -1e5
        right = 1e5
        
        if left > right  
            tmp = left
            left = right
            right = tmp
        end
        
        (v̂n, _, _) = newtbndv(v̂_root, left, right, vf[n]; ftol = 1e-12,
                              atolx = 1e-12, rtolx = 1e-12)

        if isnan(v̂n)
            #println("Not bracketing root")
        end
        
        dq[2nn^2 + n] = v̂n
        dq[nn^2 + 1 + (n - 1)*nn] +=  H[1][n, n] * (Z̃f[1][n] * v̂n)
        
    end
                 

    # Non-fault Source injection
    for i in 2:4
        SOURCE = sJ[i] .* CHAR_SOURCE(fc[1][i], fc[2][i], t, i, R[i], B_p, MMS)
        dq[2nn^2 + (i-1)*nn + 1 : 2nn^2 + i*nn] .+= SOURCE ./ (2*Z̃f[i])
        dq[nn^2 + 1:2nn^2] .+= L[i]' * H[i] * SOURCE ./ 2
    end
    
    dq[nn^2 + 1:2nn^2] .= JIHP * dq[nn^2 + 1:2nn^2]
    dq[nn^2 + 1:2nn^2] .+= P̃I * FORCE(coord[1][:], coord[2][:], t, B_p, MMS)

    # psi source
    dq[2nn^2 + 4nn + 1 : 2nn^2 + 5nn] .= (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ) ./ b) .- abs.(2 .* v̂_fric) ./ RS.V0)
    dq[2nn^2 + 4nn + 1 : 2nn^2 + 5nn] .+= STATE_SOURCE(fc[1][1], fc[2][1], b, t, B_p, RS, MMS)

end


function FAULT_CPU!(dq, q, p, t)

    nn = p.nn
    fc = p.fc
    R = p.R
    B_p = p.B_p
    RS = p.RS
    b = p.b
    Λ = p.d_ops.Λ
    sJ = p.sJ
    Z̃f = p.d_ops.Z̃f
    L = p.d_ops.L
    H = p.d_ops.H
    P̃I = p.d_ops.P̃I
    JIHP = p.d_ops.JIHP
    nCnΓ1 = p.d_ops.nCnΓ1
    nBBCΓL1 = p.d_ops.nBBCΓL1
    vf = p.vf
    τ̃f = p.τ̃f
    v̂_fric = p.v̂_fric
    
    u = @view q[1:nn^2]
    û1 = @view q[2nn^2 + 1 : 2nn^2 + nn]
    ψ = @view q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]
    
    # compute all temporal derivatives
    dq .= Λ * q
    # get velocity on fault
    vf .= L[1] * q[nn^2 + 1 : 2nn^2]
    # compute numerical traction on face 1
    τ̃f .= nBBCΓL1 * u + nCnΓ1 * û1

    # Root find for RS friction
    for n in 1:nn
        
        v̂_root(v̂) = rateandstateD(v̂,
                                  Z̃f[1][n],
                                  vf[n],
                                  sJ[1][n],
                                  ψ[n],
                                  RS.a,
                                  τ̃f[n],
                                  RS.σn,
                                  RS.V0)

        left = -1e10#vf[n] - τ̃f[n]/Z̃f[1][n]
        right = 1e10#-left
        
        if left > right  
            tmp = left
            left = right
            right = tmp
        end
        
         (v̂n, _, _) = newtbndv(v̂_root, left, right, vf[n]; ftol = 1e-12,
                              atolx = 1e-12, rtolx = 1e-12)

        if isnan(v̂n)
            println("Not bracketing root")
        end
        v̂_fric[n] = v̂n
    end
                 
    # write velocity flux into q
    dq[2nn^2 + 1 : 2nn^2 + nn] .= v̂_fric
    dq[nn^2 + 1 : 2nn^2] .+= L[1]' * H[1] * (Z̃f[1] .* v̂_fric)
    dq[nn^2 + 1:2nn^2] .= JIHP * dq[nn^2 + 1:2nn^2]
    
    # state evolution
    dq[2nn^2 + 4nn + 1 : 2nn^2 + 5nn] .= (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ) ./ b) .- abs.(2 .* v̂_fric) ./ RS.V0)

end






function MMS_GPU_FAULT!(q, p, dt, t_span)
    
    nn = p.nn
    T = eltype(q)
    q = CuArray(q)
    Δq = CuArray(zeros(length(q)))
    Λ = CuSparseMatrixCSR(p.Λ_fault)
    Z̃f1 = CuArray(p.Z̃f[1])
    Z̃f2 = CuArray(p.Z̃f[2])
    Z̃f3 = CuArray(p.Z̃f[3])
    Z̃f4 = CuArray(p.Z̃f[4])
    L1 = CuArray(p.L[1])
    L2 = CuArray(p.L[2])
    L3 = CuArray(p.L[3])
    L4 = CuArray(p.L[4])
    H1 = CuArray(p.H[1])
    H2 = CuArray(p.H[2])
    H3 = CuArray(p.H[3])
    H4 = CuArray(p.H[4])
    P̃I = CuArray(p.P̃I)
    JIHP = CuArray(p.JIHP)
    nCnΓ1 = CuArray(p.nCnΓ1)
    nBBCΓL1 = CuArray(p.nBBCΓL1)
    sJ1 = CuArray(p.sJ[1])
    sJ2 = CuArray(p.sJ[2])
    sJ3 = CuArray(p.sJ[3])
    sJ4 = CuArray(p.sJ[4])
    RS = p.RS
    rootfind = CuArray([p.nn, RS.a, RS.V0, RS.σn, RS.Dc, RS.f0, -1e10, 1e10])
    b = CuArray(p.b)
    vf = CuArray(p.vf)
    store_τ̃v̂ = CuArray(p.τ̃f)
    source_fault = CuArray(p.source_fault)
    source_2 = CuArray(p.source_2)
    source_3 = CuArray(p.source_3)
    source_4 = CuArray(p.source_4)
    volume_source = CuArray(p.volume_source)
    

    #=
    RKA = CuArray([
        T(0),
        T(-567301805773 // 1357537059087),
        T(-2404267990393 // 2016746695238),
        T(-3550918686646 // 2091501179385),
        T(-1275806237668 // 842570457699),
    ])

    RKB = CuArray([
        T(1432997174477 // 9575080441755),
        T(5161836677717 // 13612068292357),
        T(1720146321549 // 2090206949498),
        T(3134564353537 // 4481467310338),
        T(2277821191437 // 14882151754819),
    ])

    RKC = CuArray([
        T(0),
        T(1432997174477 // 9575080441755),
        T(2526269341429 // 6820363962896),
        T(2006345519317 // 3224310063776),
        T(2802321613138 // 2924317926251),
    ])
    =#

    v = @view q[nn^2 + 1: 2nn^2]
    u = @view q[1: nn^2]
    û1 = @view q[2nn^2 + 1 : 2nn^2 + nn]
    ψ = @view q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn]
    
    
    nstep = ceil(Int, (t_span[2] - t_span[1]) / dt)
    dt = (t_span[2] - t_span[1]) / nstep

    threads = 1024
    blocks = cld(nn, 1024)
    
    
    for step in 1:nstep
        
        Δq .= Λ * q
        # get velocity on fault
        vf .= L1 * v
        # compute numerical traction on face 1
        store_τ̃v̂ .= nBBCΓL1 * u + nCnΓ1 * û1
        @cuda blocks=blocks threads=threads dynamic_rootfind_d!(Δq,                                                                                  vf,
                                                                store_τ̃v̂,
                                                                ψ,
                                                                b,
                                                                Z̃f1,
                                                                sJ1,
                                                                H1,
                                                                L1,
                                                                rateandstateD_device,
                                                                rootfind)
        
        synchronize()
        
        
        Δq[2nn^2 + 1 : 2nn^2 + nn] .= store_τ̃v̂
        Δq[nn^2 + 1 : 2nn^2] .+= L1' * H1 * (Z̃f1 .* store_τ̃v̂)
        
        Δq[2nn^2 + nn + 1 : 2nn^2 + 2nn] .+= source_2[:, step] ./ (2*Z̃f2)
        Δq[nn^2 + 1 : 2nn^2] .+= L2' * H2 * source_2[:, step] ./ 2
        Δq[2nn^2 + 2nn + 1 : 2nn^2 + 3nn] .+= source_3[:,step] ./ (2*Z̃f3)
        Δq[nn^2 + 1 : 2nn^2] .+= L3' * H3 * source_3[:, step] ./ 2
        Δq[2nn^2 + 3nn + 1 : 2nn^2 + 4nn] .+= source_4[:,step] ./ (2*Z̃f4)
        Δq[nn^2 + 1 : 2nn^2] .+= L4' * H4 * source_4[:, step] ./ 2


        Δq[nn^2 + 1:2nn^2] .= JIHP * Δq[nn^2 + 1:2nn^2]
        Δq[nn^2 + 1:2nn^2] .+= volume_source[:, step]

        Δq[2nn^2 + 4nn + 1 : 2nn^2 + 5nn] .= vf + source_fault[:, step]

        q .+=  dt * Δq

        #for s in 1:length(RKA)
            #Δq .+= Λ * q
            #f!(Δq, RKA[s % length(RKA)], q, p, t + RKC[s] * dt)
            
            #q .+= RKB[s] * dt * Δq
        #end
    end
    
    nothing
end
