using SparseArrays
using LinearAlgebra
using CUDA
using CUDA.CUSPARSE

include("DiagonalSBP.jl")

const BC_DIRICHLET = 1
const BC_NEUMANN = 2
const BC_LOCKED_INTERFACE = 0
const BC_JUMP_INTERFACE   = 7


# Quasi-Dynamic rootfinding problem on the fault
function rateandstateQ(V, ψ, σn, τn, ηn, a, V0)
    Y = (1 ./ (2 .* V0)) .* exp.(ψ ./ a)
    f = a .* asinh.(V .* Y)
    dfdV  = a .* (1 ./ sqrt.(1 + (V .* Y).^2)) .* Y
    g    = σn .* f + ηn .* V - τn
    dgdV = σn .* dfdV + ηn
    (g, dgdV)
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
function rateandstateD_device(v̂::Float64, z̃::Float64, v::Float64, sJ::Float64,
                              ψ::Float64, a::Float64, τ̃::Float64, σn::Float64, V0::Float64)
    Y = (1 / (2 * V0)) * exp(ψ / a)
    f = a * CUDA.asinh(2v̂ * (1 / (2 * V0)) * exp(ψ / a))
    dfdv̂  = a * (1 / sqrt(1 + (2v̂ * Y)^2)) * 2Y
    g = sJ * σn * f .+ τ̃ .+ z̃*(v̂ .- v)
    dgdv̂ = z̃ .+ sJ * σn * dfdv̂
    
    return (g, dgdv̂)
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

function dynamic_rootfind_d!(Δq,
                             vf,
                             τ̃f,
                             ψ,
                             Z̃f,
                             sJ,
                             H,
                             L,
                             root_func, 
                             rootfind)
    
    node = blockDim().x * blockIdx().x + threadIdx().x

    #unpack params
    
    nn = rootfind[1]
    a = rootfind[2]
    V0 = rootfind[3]
    σn = rootfind[4]
    Dc = rootfind[5]
    f0 = rootfind[6]
    vL = rootfind[7]
    vR = rootfind[8]
    
    if node < nn
        
        v_iter = vf[node]
        z̃n = Z̃f[node]
        vn = vf[node]
        sJn = sJ[node]
        ψn = ψ[node] 
        τ̃n = τ̃f[node]
        
        Y = (1 / (2 * V0)) * exp(ψn / a)
        
        gL = sJn * σn * a * CUDA.asinh(2vL * Y) .+
            τ̃n .+ z̃n*(vL .- vn)
        gR = sJn * σn * a * CUDA.asinh(2vR * Y) .+
            τ̃n .+ z̃n*(vR .- vn)
        

        #if gL * gR > 0
        #    Δq[2nn^2 + node] = Float64(NaN)
        #end

        g = sJn * σn * a * CUDA.asinh(2v_iter * Y) .+
            τ̃n .+ z̃n*(v_iter .- vn)
        dg = z̃n .+ sJn * σn * a * (1 / sqrt(1 + (2v_iter * Y)^2)) * 2Y

        dvlr = vR - vL

        for iter = 1:1000
            dv = -g / dg
            v_iter  = v_iter + dv

            if v < vL || v > vR || abs(dv) / dvlr < 0
                v = (vR + vL) / 2
                dv = (vR - vL) / 2
            end
        
            g = sJn * σn * a * CUDA.asinh(2v_iter * Y) .+
            τ̃n .+ z̃n*(v_iter .- vn)
            dg = z̃n .+ sJn * σn * a * (1 / sqrt(1 + (2v_iter * Y)^2)) * 2Y

            
            if g * gL > 0
                (gL, vL) = (g, v_iter)
            else
                (gR, vR) = (g, v_iter)
            end
            dxlr = vR - vL
            
            if abs(g) < ftol && abs(dv) < atolx + rtolx * (abs(dv) + abs(v_iter))
                Δq[2nn^2 + node] = v_iter
            end
        end
    end
    nothing
end



function locbcarray_mod!(ge, lop, LFToB, bc_Dirichlet, bc_Neumann)
    F = lop.F
    (xf, yf) = lop.facecoord
    Hf = lop.H
    sJ = lop.sJ
    nx = lop.nx
    ny = lop.ny
    τ = lop.τ
    ge[:] .= 0
    for lf = 1:4
        if LFToB[lf] == BC_DIRICHLET
            vf = bc_Dirichlet(lf, xf[lf], yf[lf])
        elseif LFToB[lf] == BC_NEUMANN
            gN = bc_Neumann(lf, xf[lf], yf[lf], nx[lf], ny[lf])
            vf = sJ[lf] .* gN ./ diag(τ[lf])
        elseif LFToB[lf] == BC_LOCKED_INTERFACE
            continue 
        else
            error("invalid bc")
        end
        ge[:] -= F[lf] * vf
    end
end

function computetraction_mod(lop, lf, u, δ)
    HfI_FT = lop.HfI_FT[lf]
    τf = lop.τ[lf]
    sJ = lop.sJ[lf]
    return (HfI_FT * u + τf * (δ .- δ / 2)) ./ sJ
end

function operators_dynamic(p, Nr, Ns, B_p, μ, ρ, R, faces, metrics, LFToB, 
                     τscale = 2,
                     crr = metrics.crr,
                     css = metrics.css,
                     crs = metrics.crs)


    
    csr = crs
    J = metrics.J

    hr = 2/Nr
    hs = 2/Ns

    hmin = min(hr, hs)
    
    r = -1:hr:1
    s = -1:hs:1
    
    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp * Nsp
    Nn = Np
    nn = Nrp
    # Derivative operators for the rest of the computation
    (Dr, HrI, Hr, r) = D1(p, Nr; xc = (-1,1))
    Qr = Hr * Dr
    QrT = sparse(transpose(Qr))

    (Ds, HsI, Hs, s) = D1(p, Ns; xc = (-1,1))
    Qs = Hs * Ds
    QsT = sparse(transpose(Qs))

    # Identity matrices for the comuptation
    Ir = sparse(I, Nrp, Nrp)
    Is = sparse(I, Nsp, Nsp)

    #{{{ Set up the rr derivative matrix
    ISr0 = Array{Int64,1}(undef,0)
    JSr0 = Array{Int64,1}(undef,0)
    VSr0 = Array{Float64,1}(undef,0)
    ISrN = Array{Int64,1}(undef,0)
    JSrN = Array{Int64,1}(undef,0)
    VSrN = Array{Float64,1}(undef,0)

    (_, S0e, SNe, _, _, Ae, _) = variable_D2(p, Nr, rand(Nrp))
    IArr = Array{Int64,1}(undef,Nsp * length(Ae.nzval))
    JArr = Array{Int64,1}(undef,Nsp * length(Ae.nzval))
    VArr = Array{Float64,1}(undef,Nsp * length(Ae.nzval))
    stArr = 0

    ISr0 = Array{Int64,1}(undef,Nsp * length(S0e.nzval))
    JSr0 = Array{Int64,1}(undef,Nsp * length(S0e.nzval))
    VSr0 = Array{Float64,1}(undef,Nsp * length(S0e.nzval))
    stSr0 = 0

    ISrN = Array{Int64,1}(undef,Nsp * length(SNe.nzval))
    JSrN = Array{Int64,1}(undef,Nsp * length(SNe.nzval))
    VSrN = Array{Float64,1}(undef,Nsp * length(SNe.nzval))
    stSrN = 0
    for j = 1:Nsp
        rng = (j-1) * Nrp .+ (1:Nrp)
        (_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Nr, crr[rng])
        (Ie, Je, Ve) = findnz(Ae)
        IArr[stArr .+ (1:length(Ve))] = Ie .+ (j-1) * Nrp
        JArr[stArr .+ (1:length(Ve))] = Je .+ (j-1) * Nrp
        VArr[stArr .+ (1:length(Ve))] = Hs[j,j] * Ve
        stArr += length(Ve)

        (Ie, Je, Ve) = findnz(S0e)
        ISr0[stSr0 .+ (1:length(Ve))] = Ie .+ (j-1) * Nrp
        JSr0[stSr0 .+ (1:length(Ve))] = Je .+ (j-1) * Nrp
        VSr0[stSr0 .+ (1:length(Ve))] =  Hs[j,j] * Ve
        stSr0 += length(Ve)

        (Ie, Je, Ve) = findnz(SNe)
        ISrN[stSrN .+ (1:length(Ve))] = Ie .+ (j-1) * Nrp
        JSrN[stSrN .+ (1:length(Ve))] = Je .+ (j-1) * Nrp
        VSrN[stSrN .+ (1:length(Ve))] =  Hs[j,j] * Ve
        stSrN += length(Ve)
    end
    Ãrr = sparse(IArr[1:stArr], JArr[1:stArr], VArr[1:stArr], Np, Np)
    Sr0 = sparse(ISr0[1:stSr0], JSr0[1:stSr0], VSr0[1:stSr0], Np, Np)
    SrN = sparse(ISrN[1:stSrN], JSrN[1:stSrN], VSrN[1:stSrN], Np, Np)
    Sr0T = sparse(JSr0[1:stSr0], ISr0[1:stSr0], VSr0[1:stSr0], Np, Np)
    SrNT = sparse(JSrN[1:stSrN], ISrN[1:stSrN], VSrN[1:stSrN], Np, Np)

    (_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Ns, rand(Nsp))
    IAss = Array{Int64,1}(undef,Nrp * length(Ae.nzval))
    JAss = Array{Int64,1}(undef,Nrp * length(Ae.nzval))
    VAss = Array{Float64,1}(undef,Nrp * length(Ae.nzval))
    stAss = 0

    ISs0 = Array{Int64,1}(undef,Nrp * length(S0e.nzval))
    JSs0 = Array{Int64,1}(undef,Nrp * length(S0e.nzval))
    VSs0 = Array{Float64,1}(undef,Nrp * length(S0e.nzval))
    stSs0 = 0

    ISsN = Array{Int64,1}(undef,Nrp * length(SNe.nzval))
    JSsN = Array{Int64,1}(undef,Nrp * length(SNe.nzval))
    VSsN = Array{Float64,1}(undef,Nrp * length(SNe.nzval))
    stSsN = 0
    for i = 1:Nrp
        rng = i .+ Nrp * (0:Ns)
        (_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Ns, css[rng])

        (Ie, Je, Ve) = findnz(Ae)
        IAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
        JAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
        VAss[stAss .+ (1:length(Ve))] = Hr[i,i] * Ve
        stAss += length(Ve)

        (Ie, Je, Ve) = findnz(S0e)
        ISs0[stSs0 .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
        JSs0[stSs0 .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
        VSs0[stSs0 .+ (1:length(Ve))] = Hr[i,i] * Ve
        stSs0 += length(Ve)

        (Ie, Je, Ve) = findnz(SNe)
        ISsN[stSsN .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
        JSsN[stSsN .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
        VSsN[stSsN .+ (1:length(Ve))] = Hr[i,i] * Ve
        stSsN += length(Ve)
    end
    Ãss = sparse(IAss[1:stAss], JAss[1:stAss], VAss[1:stAss], Np, Np)
    Ss0 = sparse(ISs0[1:stSs0], JSs0[1:stSs0], VSs0[1:stSs0], Np, Np)
    SsN = sparse(ISsN[1:stSsN], JSsN[1:stSsN], VSsN[1:stSsN], Np, Np)
    Ss0T = sparse(JSs0[1:stSs0], ISs0[1:stSs0], VSs0[1:stSs0], Np, Np)
    SsNT = sparse(JSsN[1:stSsN], ISsN[1:stSsN], VSsN[1:stSsN], Np, Np)

    Ãsr = (QsT ⊗ Ir) * sparse(1:length(crs), 1:length(crs), view(crs, :)) * (Is ⊗ Qr)
    Ãrs = (Is ⊗ QrT) * sparse(1:length(csr), 1:length(csr), view(csr, :)) * (Qs ⊗ Ir)

    Ã = Ãrr + Ãss + Ãrs + Ãsr
   
    # volume quadrature
    H̃ = kron(Hr, Hs)
    H̃inv = spdiagm(0 => 1 ./ diag(H̃))
    
    # diagonal rho
    rho = ρ(metrics.coord[1], metrics.coord[2], B_p)
    rho = reshape(rho, Nrp*Nsp)
    P̃ = spdiagm(0 => rho)
    P̃inv = spdiagm(0 => (1 ./ rho))
    JI = spdiagm(0 => reshape(metrics.JI, Nrp*Nsp))
    
    er0 = sparse([1  ], [1], [1], Nrp, 1)
    erN = sparse([Nrp], [1], [1], Nrp, 1)
    es0 = sparse([1  ], [1], [1], Nsp, 1)
    esN = sparse([Nsp], [1], [1], Nsp, 1)

    er0T = sparse([1], [1  ], [1], 1, Nrp)
    erNT = sparse([1], [Nrp], [1], 1, Nrp)
    es0T = sparse([1], [1  ], [1], 1, Nsp)
    esNT = sparse([1], [Nsp], [1], 1, Nsp)

    # Store coefficient matrices as matrices
    crs0 = sparse(Diagonal(crs[1:Nrp]))
    crsN = sparse(Diagonal(crs[Nrp*Ns .+ (1:Nrp)]))
    csr0 = sparse(Diagonal(csr[1   .+ Nrp*(0:Ns)]))
    csrN = sparse(Diagonal(csr[Nrp .+ Nrp*(0:Ns)]))

    cmax = maximum([maximum(crr), maximum(crs), maximum(css)])

    # Surface quadtrature matrices
    H1 = H2 = Hs 
    H3 = H4 = Hr

    H = (Hs, Hs, Hr, Hr)
    # Volume to Face Operators (transpose of these is face to volume)
    L = (convert(Array{Float64, 2}, kron(Ir, es0)'),
         convert(Array{Float64, 2}, kron(Ir, esN)'),
         convert(Array{Float64, 2}, kron(er0, Is)'),
         convert(Array{Float64, 2}, kron(erN, Is)'))

    # coefficent matrices
    Crr1 = spdiagm(0 => crr[1, :])
    Crs1 = spdiagm(0 => crs[1, :])
    Csr1 = spdiagm(0 => crs[1, :])
    Css1 = spdiagm(0 => css[1, :])
    
    Crr2 = spdiagm(0 => crr[Nrp, :])
    Crs2 = spdiagm(0 => crs[Nrp, :])
    Csr2 = spdiagm(0 => crs[Nrp, :])
    Css2 = spdiagm(0 => css[Nrp, :])
    
    Css3 = spdiagm(0 => css[:, 1])
    Crs3 = spdiagm(0 => crs[:, 1])
    Csr3 = spdiagm(0 => crs[:, 1])
    Crr3 = spdiagm(0 => crr[:, 1])
    
    Css4 = spdiagm(0 => css[:, Nsp])
    Crs4 = spdiagm(0 => crs[:, Nsp])
    Csr4 = spdiagm(0 => crs[:, Nsp])
    Crr4 = spdiagm(0 => crr[:, Nsp])

    (_, S0, SN, _, _) = D2(p, Nr, xc=(-1,1))[1:5]
    S0 = sparse(Array(S0[1,:])')
    SN = sparse(Array(SN[end, :])')
    
    # Boundars Derivatives
    B1r =  Crr1 * kron(Is, S0)
    B1s = Crs1 * L[1] * kron(Ds, Ir)
    B2r = Crr2 * kron(Is, SN)
    B2s = Crs2 * L[2] * kron(Ds, Ir)
    B3s = Css3 * kron(S0, Ir)
    B3r = Csr3 * L[3] * kron(Is, Dr)
    B4s = Css4 * kron(SN, Ir)
    B4r = Csr4 * L[4] * kron(Is, Dr)
    

    (xf1, xf2, xf3, xf4) = metrics.facecoord[1]
    (yf1, yf2, yf3, yf4) = metrics.facecoord[2]

    Z̃f = (metrics.sJ[1] .* sqrt.(ρ(xf1, yf1, B_p) .* μ(xf1, yf1, B_p)),
          metrics.sJ[2] .* sqrt.(ρ(xf2, yf2, B_p) .* μ(xf2, yf2, B_p)),
          metrics.sJ[3] .* sqrt.(ρ(xf3, yf3, B_p) .* μ(xf3, yf3, B_p)),
          metrics.sJ[4] .* sqrt.(ρ(xf4, yf4, B_p) .* μ(xf4, yf4, B_p)))

    # Penalty terms
        if p == 2
        l = 2
        β = 0.363636363
        α = 1 / 2
        θ_R = 1.0
    elseif p == 4
        l = 4
        β = 0.2505765857
        α = 17 / 48
        θ_R = 0.5776
    elseif p == 6
        l = 7
        β = 0.1878687080
        α = 13649 / 43200
        θ_R = 0.3697
    else
        error("unknown order")
    end

    ψmin_r = reshape(crr, Nrp, Nsp)
    ψmin_s = reshape(css, Nrp, Nsp)
    @assert minimum(ψmin_r) > 0
    @assert minimum(ψmin_s) > 0
    
    hr = 2 / Nr
    hs = 2 / Ns

    ψ1 = ψmin_r[  1, :]
    ψ2 = ψmin_r[Nrp, :]
    ψ3 = ψmin_s[:,   1]
    ψ4 = ψmin_s[:, Nsp]
    
    for k = 2:l
        ψ1 = min.(ψ1, ψmin_r[k, :])
        ψ2 = min.(ψ2, ψmin_r[Nrp+1-k, :])
        ψ3 = min.(ψ3, ψmin_s[:, k])
        ψ4 = min.(ψ4, ψmin_s[:, Nsp+1-k])
    end
    
    τR1 = (1/(θ_R*hr))*Is
    τR2 = (1/(θ_R*hr))*Is
    τR3 = (1/(θ_R*hs))*Ir
    τR4 = (1/(θ_R*hs))*Ir
    
    p1 = ((crr[  1, :]) ./ ψ1)
    p2 = ((crr[Nrp, :]) ./ ψ2)
    p3 = ((css[:,   1]) ./ ψ3)
    p4 = ((css[:, Nsp]) ./ ψ4)
   
    P1 = sparse(1:Nsp, 1:Nsp, p1)
    P2 = sparse(1:Nsp, 1:Nsp, p2)
    P3 = sparse(1:Nrp, 1:Nrp, p3)
    P4 = sparse(1:Nrp, 1:Nrp, p4)

    # dynamic penalty matrices
    Γ = ((2/(α*hr))*Is + τR1 * P1,
         (2/(α*hr))*Is + τR2 * P2,
         (2/(α*hs))*Ir + τR3 * P3,
         (2/(α*hs))*Ir + τR4 * P4)

    JH = sparse(1:Np, 1:Np, view(J, :)) * (Hs ⊗ Hr)
    
    JIHP = JI * H̃inv * P̃inv
    
    Cf = ((Crr1, Crs1), (Crr2, Crs2), (Css3, Csr3), (Css4, Csr4))
    B = ((B1r, B1s), (B2r, B2s), (B3s, B3r), (B4s, B4r))
    nl = (-1, 1, -1, 1)
    
    # accleration blocks
    dv_u = -Ã
    
    for i in 1:4
        if faces[i] == 0
            dv_u .+= (L[i]' * H[i] * (nl[i] * (B[i][1] + B[i][2]) - Cf[i][1] * Γ[i] * L[i])) +
                nl[i] * (B[i][1]' + B[i][2]') * H[i] * L[i]
        else
            dv_u .+= (L[i]' * H[i] * ((1 - R[i])/2 .* (nl[i] * (B[i][1] + B[i][2]) - Cf[i][1] * Γ[i] * L[i]))) +
                nl[i] * (B[i][1]' + B[i][2]') * H[i] * L[i]
        end
    end

    dv_v = spzeros(Nn, Nn)
    for i in 1:4
        if faces[i] == 0
            dv_v .+=  L[i]' * H[i] * (-Z̃f[i] .* L[i])
        else
            dv_v .+=  L[i]' * H[i] * (-(1 - R[i])/2 .* Z̃f[i] .* L[i])
        end
    end

    dv_û = spzeros(Nn, 4nn)
    dû_u = spzeros(4nn, Nn)
    dû_v = spzeros(4nn, Nn)

    for i in 1:4
        if faces[i] == 0
                        dv_û[ : , (i-1) * nn + 1 : i * nn] .=
                (L[i]' * H[i] * Cf[i][1] * Γ[i]) -
                nl[i] * (B[i][1]' + B[i][2]') * H[i]
        else
            dv_û[ : , (i-1) * nn + 1 : i * nn] .=
                (L[i]' * H[i] * ((1 - R[i])/2 .* Cf[i][1] * Γ[i])) -
                nl[i] * (B[i][1]' + B[i][2]') * H[i]
        
            dû_u[(i-1) * nn + 1 : i * nn , : ] .=
                -(1 + R[i])/2 .* (nl[i] * (B[i][1] + B[i][2]) -
                Cf[i][1] * Γ[i] * L[i])./Z̃f[i]
        
            dû_v[(i-1) * nn + 1 : i * nn , : ] .= (1 + R[i])/2 .* L[i]
        end
    end

    
    dû_û = spzeros(4nn, 4nn)
    for i in 1:4
        if faces[i] != 0
            dû_û[(i-1) * nn + 1 : i * nn, (i-1) * nn + 1 : i * nn] .= -(1 + R[i])/2 .* (Cf[i][1] * Γ[i])./Z̃f[i]
        end
    end

    dû_ψ = spzeros(4nn, nn)

    Λ = [ spzeros(Nn, Nn) sparse(I, Nn, Nn) spzeros(Nn, 5nn)
          dv_u dv_v dv_û spzeros(Nn, nn)
          dû_u dû_v dû_û dû_ψ
          spzeros(nn, 2Nn + 5nn) ]

    nCnΓ1 = Crr1 * Γ[1]
    nBBCΓL1 = nl[1] * (B[1][1] + B[1][2]) - nCnΓ1 * L[1]
    
    (Λ = Λ,
     Ã = Ã,
     P̃I = P̃inv,
     H̃I = H̃inv,
     JI = JI,
     JIHP = JIHP,
     nCnΓ1 = nCnΓ1,
     nBBCΓL1 = nBBCΓL1,
     hmin = hmin,
     cmax = cmax,
     JH = JH,
     sJ = metrics.sJ,
     nx = metrics.nx,
     ny = metrics.ny,
     L = L,
     H = H,
     Z̃f = Z̃f)
    
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


function rk4!(q, f!, p, dt, tspan)

    nstep = ceil(Int, (tspan[2] - tspan[1]) / dt)
    dt = (tspan[2] - tspan[1]) / nstep
    Δq = similar(q)
    Δq2 = similar(q)
    fill!(Δq, 0)
    fill!(Δq2, 0)
    
    for step in 1:nstep
        fill!(Δq, 0)
        t = tspan[1] + (step - 1) * dt
        f!(Δq2, q, p, t)
        Δq .+= 1/6 * dt * Δq2
        f!(Δq2, q + (1/2) * dt * Δq2, p, t + (1/2) * dt)
        Δq .+= 1/6 * dt * 2Δq2
        f!(Δq2, q + (1/2) * dt * Δq2, p, t + (1/2) * dt)
        Δq .+= 1/6 * dt * 2Δq2
        f!(Δq2, q + dt * Δq2, p, t + dt)
        Δq .+= 1/6 * dt * Δq2
        q .+= Δq
        
    end
    nothing
end


function rk4!(q, Λ, dt, tspan)

    nstep = ceil(Int, (tspan[2] - tspan[1]) / dt)
    dt = (tspan[2] - tspan[1]) / nstep
    Δq = similar(q)
    Δq2 = similar(q)
    for step in 1:nstep
        Δq .= 0
        t = tspan[1] + (step - 1) * dt
        Δq2 .= Λ * q
        Δq .+= 1/6 * dt * Δq2
        Δq2 .= Λ * (q + (1/2) * dt * Δq2)
        Δq .+= 1/6 * dt * 2Δq2
        Δq2 .= Λ * (q + (1/2) * dt * Δq2)
        Δq .+= 1/6 * dt * 2Δq2
        Δq2 .= Λ * (q + dt * Δq2)
        Δq .+= 1/6 * dt * Δq2
        q .+= Δq
    end
    nothing
end

