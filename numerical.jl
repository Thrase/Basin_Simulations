using SparseArrays
using LinearAlgebra
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

function locoperator(p, Nr, Ns, B_p, μ, ρ, metrics, LFToB, 
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
        R = Ae - Dr' * Hr * Diagonal(css[rng]) * Dr

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
    
    # Boundary point matrices
    Er0 = sparse([1], [1], [1], Nrp, Nrp)
    ErN = sparse([Nrp], [Nrp], [1], Nrp, Nrp)
    Es0 = sparse([1], [1], [1], Nsp, Nsp)
    EsN = sparse([Nsp], [Nsp], [1], Nsp, Nsp)

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
    H1 = Hs
    H1I = HsI

    H2 = Hs
    H2I = HsI

    H3 = Hr
    H3I = HrI

    H4 = Hr
    H4I = HrI

    # Volume to Face Operators (transpose of these is face to volume)
    L1= kron(Ir, es0)'
    L2 = kron(Ir, esN)'
    L3 = kron(er0, Is)'
    L4 = kron(erN, Is)'

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

    (_, S0, SN, HI, H) = D2(p, Nr, xc=(-1,1))[1:5]
    S0 = sparse(Array(S0[1,:])')
    SN = sparse(Array(SN[end, :])')
    
    # Boundars Derivatives
    B1r =  Crr1 * kron(Is, S0)
    #display(sparse(B1r'))
    B1s = Crs1 * L1 * kron(Ds, Ir)
    B2r = Crr2 * kron(Is, SN)
    #display(sparse(B2r'))
    B2s = Crs2 * L2 * kron(Ds, Ir)
    B3s = Css3 * kron(S0, Ir)
    B3r = Csr3 * L3 * kron(Is, Dr)
    B4s = Css4 * kron(SN, Ir)
    B4r = Csr4 * L4 * kron(Is, Dr)


    
    (xf1, xf2, xf3, xf4) = metrics.facecoord[1]
    (yf1, yf2, yf3, yf4) = metrics.facecoord[2]

    # Shear impedence on faces
    Z1 = sqrt.(ρ(xf1, yf1, B_p) .* μ(xf1, yf1, B_p))
    Z2 = sqrt.(ρ(xf2, yf2, B_p) .* μ(xf2, yf2, B_p))
    Z3 = sqrt.(ρ(xf3, yf3, B_p) .* μ(xf3, yf3, B_p))
    Z4 = sqrt.(ρ(xf4, yf4, B_p) .* μ(xf4, yf4, B_p))
    Z̃1 = metrics.sJ[1] .* Z1
    Z̃2 = metrics.sJ[2] .* Z2
    Z̃3 = metrics.sJ[3] .* Z3
    Z̃4 = metrics.sJ[4] .* Z4

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

    τ1 = (2τscale / hr) * (crr[  1, :].^2 / β + crs[  1, :].^2 / α) ./ ψ1
    τ2 = (2τscale / hr) * (crr[Nrp, :].^2 / β + crs[Nrp, :].^2 / α) ./ ψ2
    τ3 = (2τscale / hs) * (css[:,   1].^2 / β + crs[:,   1].^2 / α) ./ ψ3
    τ4 = (2τscale / hs) * (css[:, Nsp].^2 / β + crs[:, Nsp].^2 / α) ./ ψ4

    # static penalty matrices
    τ1 = sparse(1:Nsp, 1:Nsp, τ1)
    τ2 = sparse(1:Nsp, 1:Nsp, τ2)
    τ3 = sparse(1:Nrp, 1:Nrp, τ3)
    τ4 = sparse(1:Nrp, 1:Nrp, τ4)

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
    Γ1 = (2/(α*hr))*Is + τR1 * P1
    Γ2 = (2/(α*hr))*Is + τR2 * P2
    Γ3 = (2/(α*hs))*Ir + τR3 * P3
    Γ4 = (2/(α*hs))*Ir + τR4 * P4

    nCnΓ1 = Crr1 * Γ1
    nCnΓ2 = Crr2 * Γ2
    nCnΓ3 = Css3 * Γ3
    nCnΓ4 = Css4 * Γ4
    
    nBBCΓL1 = -(B1r + B1s) - nCnΓ1 * L1
    nBBCΓL2 = (B2r + B2s) - nCnΓ2 * L2
    nBBCΓL3 = -(B3r + B3s) - nCnΓ3 * L3
    nBBCΓL4 = (B4r + B4s) - nCnΓ4 * L4

    BCTH1 = -(-(B1r' + B1s')) * H1
    BCTH2 = -(B2r' + B2s') * H2
    BCTH3 = -(-(B3r' + B3s')) * H3
    BCTH4 = -(B4r'+ B4s') * H4

    BCTHL1 = -(B1r' + B1s') * H1 * L1
    BCTHL2 = (B2r' + B2s') * H2 * L2
    BCTHL3 = -(B3r' + B3s') * H3 * L3
    BCTHL4 = (B4r' + B4s') * H4 * L4

    
    
    
    C̃1 =  (Sr0 + Sr0T) + ((csr0 * Qs + QsT * csr0) ⊗ Er0) + ((τ1 * H1) ⊗ Er0)
    C̃2 = -(SrN + SrNT) - ((csrN * Qs + QsT * csrN) ⊗ ErN) + ((τ2 * H2) ⊗ ErN)
    C̃3 =  (Ss0 + Ss0T) + (Es0 ⊗ (crs0 * Qr + QrT * crs0)) + (Es0 ⊗ (τ3 * H3))
    C̃4 = -(SsN + SsNT) - (EsN ⊗ (crsN * Qr + QrT * crsN)) + (EsN ⊗ (τ4 * H4))

    G1 = -(Is ⊗ er0T) * Sr0 - ((csr0 * Qs) ⊗ er0T)
    G2 = +(Is ⊗ erNT) * SrN + ((csrN * Qs) ⊗ erNT)
    G3 = -(es0T ⊗ Ir) * Ss0 - (es0T ⊗ (crs0 * Qr))
    G4 = +(esNT ⊗ Ir) * SsN + (esNT ⊗ (crsN * Qr))

    F1 = G1' - ((τ1 * H1) ⊗ er0)
    F2 = G2' - ((τ2 * H2) ⊗ erN)
    F3 = G3' - (es0 ⊗ (τ3 * H3))
    F4 = G4' - (esN ⊗ (τ4 * H4))

    HfI_F1T = H1I * G1 - (τ1 ⊗ er0')
    HfI_F2T = H2I * G2 - (τ2 ⊗ erN')
    HfI_F3T = H3I * G3 - (es0' ⊗ τ3)
    HfI_F4T = H4I * G4 - (esN' ⊗ τ4)

    HfI_G1 = H1I * G1
    HfI_G2 = H2I * G2
    HfI_G3 = H3I * G3
    HfI_G4 = H4I * G4

    M̃ = Ã + C̃1 + C̃2 + C̃3 + C̃4

    # Modify the operator to handle the boundary conditions
    bctype=(BC_LOCKED_INTERFACE, BC_LOCKED_INTERFACE, BC_LOCKED_INTERFACE, BC_LOCKED_INTERFACE)
    F = (F1, F2, F3, F4)
    τ = (τ1, τ2, τ3, τ4)
    HfI = (H1I, H2I, H3I, H4I)

    # Modify operators for the BC
    for lf = 1:4
        if LFToB[lf] == BC_NEUMANN
            M̃ -= F[lf] * (Diagonal(1 ./ (diag(τ[lf]))) * HfI[lf]) * F[lf]'
        elseif !(LFToB[lf] == BC_DIRICHLET ||
                 LFToB[lf] == BC_LOCKED_INTERFACE ||
                 LFToB[lf] >= BC_JUMP_INTERFACE)
            error("invalid bc")
        end
    end
    bctype=(LFToB[1], LFToB[2], LFToB[3], LFToB[4])
    JH = sparse(1:Np, 1:Np, view(J, :)) * (Hs ⊗ Hr)

    JIHP = JI * H̃inv * P̃inv
    
    (M̃ = cholesky(Symmetric(M̃)),
     Ã = Ã,
     P̃I = P̃inv,
     H̃I = H̃inv,
     JI = JI,
     JIHP = JIHP,
     F = (F1, F2, F3, F4),
     HfI_FT = (HfI_F1T, HfI_F2T, HfI_F3T, HfI_F4T),
     HfI_G = (HfI_G1, HfI_G2, HfI_G3, HfI_G4),
     coord = metrics.coord,
     facecoord = metrics.facecoord,
     hmin = hmin,
     cmax = cmax,
     JH = JH,
     sJ = metrics.sJ,
     nx = metrics.nx,
     ny = metrics.ny,
     HfI = (H1I, H2I, H3I, H4I),
     τ = (τ1, τ2, τ3, τ4),
     L = (L1, L2, L3, L4),
     H = (H1, H2, H3, H4),
     Z̃f = (Z̃1, Z̃2, Z̃3, Z̃4),
     Cf = ((Crr1, Crs1), (Crr2, Crs2), (Css3, Csr3), (Css4, Csr4)),
     B = ((B1r, B1s), (B2r, B2s), (B3s, B3r), (B4s, B4r)),
     BCTH = (BCTH1, BCTH2, BCTH3, BCTH4),
     BCTHL = (BCTHL1,  BCTHL2,  BCTHL3,  BCTHL4),
     nBBCΓL = (nBBCΓL1,  nBBCΓL2,  nBBCΓL3,  nBBCΓL4),
     nCnΓ = (nCnΓ1, nCnΓ2, nCnΓ3, nCnΓ4),
     #RZ̃L = (RZ̃L1, RZ̃L2, RZ̃L3, RZ̃L4),
     Γ = (Γ1, Γ2, Γ3, Γ4),
     n = (-1, 1, -1, 1),
     bctype=bctype)
end

function sbpA(p, Nr, Ns, metrics, 
                     τscale = 2,
                     crr = metrics.crr,
                     css = metrics.css,
                     crs = metrics.crs)

    csr = crs

    hr = 2/Nr
    hs = 2/Ns

    hmin = min(hr, hs)
    
    r = -1:hr:1
    s = -1:hs:1
    
    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp * Nsp

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
        R = Ae - Dr' * Hr * Diagonal(css[rng]) * Dr

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

    return Ã
    
end



function dynamicblock(ops)

    Nn = ops.Nn
    nn = ops.nn
    Ã = ops.Ã
    L = ops.L
    H = ops.H
    R = ops.R
    Z̃f = ops.Z̃f
    nBBCΓL = ops.nBBCΓL
    BCTHL = ops.BCTHL
    nCnΓ = ops.nCnΓ
    BCTH = ops.BCTH
    JIHP = ops.JIHP
    
    @show Nn, nn
    # velocity blocks
    du_u = spzeros(Nn, Nn)
    du_v = sparse(I, Nn, Nn)
    du_û = spzeros(Nn, 4nn)
    du_ψ = spzeros(Nn, nn)

    
    # accleration blocks
    dv_u = -Ã
    for i in 1:4
        dv_u .+= (L[i]' * H[i] * ((1 - R[i])/2 .* nBBCΓL[i])) + BCTHL[i]
    end
    
    dv_v = spzeros(Nn, Nn)
    for i in 1:4
        dv_v .+= L[i]' * H[i] * (-(1 - R[i])/2 .* Z̃f[i] .* L[i])
    end

    dv_û1 = (L[1]' * H[1] * ((1 - R[1])/2 .* nCnΓ[1])) + BCTH[1]
    dv_û2 = (L[2]' * H[2] * ((1 - R[2])/2 .* nCnΓ[2])) + BCTH[2]
    dv_û3 = (L[3]' * H[3] * ((1 - R[3])/2 .* nCnΓ[3])) + BCTH[3]
    dv_û4 = (L[4]' * H[4] * ((1 - R[4])/2 .* nCnΓ[4])) + BCTH[4]

    dv_û = [ dv_û1 dv_û2 dv_û3 dv_û4 ]

    dv_ψ = spzeros(Nn, nn)

    # flux blocks
    
    dû1_u = -(1 + R[1])/2 .* nBBCΓL[1]./Z̃f[1]
    dû2_u = -(1 + R[2])/2 .* nBBCΓL[2]./Z̃f[2]
    dû3_u = -(1 + R[3])/2 .* nBBCΓL[3]./Z̃f[3]
    dû4_u = -(1 + R[4])/2 .* nBBCΓL[4]./Z̃f[4]

    dû_u = [ dû1_u
             dû2_u
             dû3_u
             dû4_u ]

    #display(dû1_u)
    #display(dû2_u)
    #display(dû3_u)
    #display(dû4_u)
    #display(dû_u)
    
    dû1_v = (1 + R[1])/2 .* L[1]
    dû2_v = (1 + R[2])/2 .* L[2]
    dû3_v = (1 + R[3])/2 .* L[3]
    dû4_v = (1 + R[4])/2 .* L[4]

    dû_v = [ dû1_v
             dû2_v
             dû3_v
             dû4_v ]

    #display(dû1_v)
    #display(dû2_v)
    #display(dû3_v)
    #display(dû4_v)
    #display(dû_v)
    
    dû1_û1 = -(1 + R[1])/2 .* nCnΓ[1]./Z̃f[1]
    dû1_û2 = spzeros(nn, nn)
    dû1_û3 = spzeros(nn, nn)
    dû1_û4 = spzeros(nn, nn)

    
    dû2_û1 = spzeros(nn, nn)
    dû2_û2 = -(1 + R[2])/2 .* nCnΓ[2]./Z̃f[2]
    dû2_û3 = spzeros(nn, nn)
    dû2_û4 = spzeros(nn, nn)

    
    dû3_û1 = spzeros(nn, nn)
    dû3_û2 = spzeros(nn, nn)
    dû3_û3 = -(1 + R[3])/2 .* nCnΓ[3]./Z̃f[3]
    dû3_û4 = spzeros(nn, nn)

    dû4_û1 = spzeros(nn, nn)
    dû4_û2 = spzeros(nn, nn)
    dû4_û3 = spzeros(nn, nn)
    dû4_û4 = -(1 + R[4])/2 .* nCnΓ[4]./Z̃f[4]

    dû_û = [ dû1_û1 dû1_û2 dû1_û3 dû1_û4
             dû2_û1 dû2_û2 dû2_û3 dû2_û4
             dû3_û1 dû3_û2 dû3_û3 dû3_û4
             dû4_û1 dû4_û2 dû4_û3 dû4_û4 ]
    
    dû1_ψ = spzeros(nn, nn)
    dû2_ψ = spzeros(nn, nn)
    dû3_ψ = spzeros(nn, nn)
    dû4_ψ = spzeros(nn, nn)

    dû_ψ = [ dû1_ψ
             dû2_ψ
             dû3_ψ
             dû4_ψ ]
    
    # state blocks
    dψ_u = spzeros(nn, Nn)
    dψ_v = spzeros(nn, Nn)
    dψ_û = spzeros(nn, 4nn)
    dψ_ψ = spzeros(nn, nn)
    
    Λ = [ du_u du_v du_û du_ψ
          #=JIHP*=#dv_u #=JIHP*=#dv_v dv_û dv_ψ
          dû_u dû_v dû_û dû_ψ
          dψ_u dψ_v dψ_û dψ_ψ ]

    
    #display(Λ)
    return Λ

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
