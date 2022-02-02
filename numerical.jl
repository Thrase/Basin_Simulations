using SparseArrays
using LinearAlgebra
using CUDA
using CUDA.CUSPARSE
using Printf

include("DiagonalSBP.jl")

const BC_DIRICHLET = 1
const BC_NEUMANN = 2
const BC_LOCKED_INTERFACE = 0
const BC_JUMP_INTERFACE   = 7

CUDA.allowscalar(false)

⊗(A,B) = kron(A, B)




function mod_data!(δ, ge, K, vf, RS, metrics, Lw, t)

    
    (xf, yf) = metrics.facecoord
    coord = metrics.coord
    sJ = metrics.sJ
    μf2 = metrics.μf2
    
    ge .= 0
    
    for i in 1:4
        if i == 1
            vf .=  δ ./ 2
        elseif i == 2
            # dirichlet h
            vf .= f2_data(RS, μf2, Lw, t)
        elseif i == 3
            # neumann h
            vf .= 0
        elseif i == 4
            # neumann h
            vf .= 0
        end
        
        ge .+= K[i] * vf
    end

end



function static_data_mms!(δ, ge, K, JH, vf, MMS, B_p, RS, metrics, ys)

    (xf, yf) = metrics.facecoord
    coord = metrics.coord
    sJ = metrics.sJ
    μf2 = μ(xf[2], yf[2], B_p)
    
    ge .= 0
    
    for i in 1:4
        if i == 1
            vf .=  he(xf[1], yf[1], 35 * ys, MMS)
        elseif i == 2
            # dirichlet h
            vf .= (MMS.Vp/2 * 35 * ys .+ (RS.τ_inf * MMS.Lw) ./ μf2) .+
                h_face2(xf[2], yf[2], 35 * ys, MMS, RS, μf2)
        elseif i == 3
            # neumann h
            vf .= sJ[3] .* τhe(xf[3], yf[3], 35 * ys, 3, B_p, MMS)
        elseif i == 4
            # neumann h
            vf .= sJ[4] .* τhe(xf[4], yf[4], 35 * ys, 4, B_p, MMS)
        end
        
        ge .+= K[i] * vf
    end

    ge .+= JH * h_FORCE(coord[1][:], coord[2][:], 35 * ys, B_p, MMS)
    
end


function mod_data_mms!(δ, ge, K, H̃, JH, vf, MMS, B_p, RS, metrics, t)

    (xf, yf) = metrics.facecoord
    coord = metrics.coord
    sJ = metrics.sJ
    μf2 = μ(xf[2], yf[2], B_p)
    
    ge .= 0
    
    for i in 1:4
        if i == 1
            vf .=  δ ./ 2
        elseif i == 2
            # dirichlet h
            vf .= (MMS.Vp/2 * t .+ (RS.τ_inf * MMS.Lw) ./ μf2) .+
                h_face2(xf[2], yf[2], t, MMS, RS, μf2)
        elseif i == 3
            # neumann h
            vf .= sJ[3] .* τhe(xf[3], yf[3], t, 3, B_p, MMS)
        elseif i == 4
            # neumann h
            vf .= sJ[4] .* τhe(xf[4], yf[4], t, 4, B_p, MMS)
        end
        
        ge .+= K[i] * vf
    end

    ge .+= JH * h_FORCE(coord[1][:], coord[2][:], t, B_p, MMS)
    
end


function f2_data(RS, μf2, Lw, t)
    return (RS.Vp/2 * t .+ (RS.τ_inf * Lw) ./ μf2)
end


function traction(ops, f, u, û)

    HI = ops.HI[f]
    G = ops.G[f]
    Γ = ops.Γ[f]
    L = ops.L[f]
    
    return (HI * G * u + Γ * (û - L * u))
    
end



            
function operators(p, Nr, Ns, μ, ρ, R, B_p, faces, metrics, 
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

    (_, S0e, SNe, _, _, Ae, _) = variable_D2(p, Nr, rand(Nrp))
    IArr = Array{Int64,1}(undef,Nsp * length(Ae.nzval))
    JArr = Array{Int64,1}(undef,Nsp * length(Ae.nzval))
    VArr = Array{Float64,1}(undef,Nsp * length(Ae.nzval))
    stArr = 0
    A_t = @elapsed begin
        for j = 1:Nsp
            rng = (j-1) * Nrp .+ (1:Nrp)
            (_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Nr, crr[rng])
            (Ie, Je, Ve) = findnz(Ae)
            IArr[stArr .+ (1:length(Ve))] = Ie .+ (j-1) * Nrp
            JArr[stArr .+ (1:length(Ve))] = Je .+ (j-1) * Nrp
            VArr[stArr .+ (1:length(Ve))] = Hs[j,j] * Ve
            stArr += length(Ve)
        end

        Ãrr = sparse(IArr[1:stArr], JArr[1:stArr], VArr[1:stArr], Np, Np)

        (_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Ns, rand(Nsp))
        IAss = Array{Int64,1}(undef,Nrp * length(Ae.nzval))
        JAss = Array{Int64,1}(undef,Nrp * length(Ae.nzval))
        VAss = Array{Float64,1}(undef,Nrp * length(Ae.nzval))
        stAss = 0

        for i = 1:Nrp
            rng = i .+ Nrp * (0:Ns)
            (_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Ns, css[rng])

            (Ie, Je, Ve) = findnz(Ae)
            IAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
            JAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
            VAss[stAss .+ (1:length(Ve))] = Hr[i,i] * Ve
            stAss += length(Ve)

        end

        Ãss = sparse(IAss[1:stAss], JAss[1:stAss], VAss[1:stAss], Np, Np)

        Ãsr = (QsT ⊗ Ir) * sparse(1:length(crs), 1:length(crs), view(crs, :)) * (Is ⊗ Qr)
        Ãrs = (Is ⊗ QrT) * sparse(1:length(csr), 1:length(csr), view(csr, :)) * (Qs ⊗ Ir)

        Ã = Ãrr + Ãss + Ãrs + Ãsr
    end

    #@printf "Got Ã in %f seconds\n" A_t


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

    cmax = maximum([maximum(crr), maximum(crs), maximum(css)])

    # Surface quadtrature matrices
    H1 = H2 = Hs 
    H3 = H4 = Hr

    H = (Hs, Hs, Hr, Hr)
    HI = (HsI, HsI, HrI, HrI)
    # Volume to Face Operators (transpose of these is face to volume)
    L = (convert(SparseMatrixCSC{Float64, Int64}, kron(Ir, es0)'),
         convert(SparseMatrixCSC{Float64, Int64}, kron(Ir, esN)'),
         convert(SparseMatrixCSC{Float64, Int64}, kron(er0, Is)'),
         convert(SparseMatrixCSC{Float64, Int64}, kron(erN, Is)'))
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
        α = 1.0
        θ_R = 1 / 2
    elseif p == 4
        l = 4
        β = 0.2505765857
        α = 0.5776
        θ_R = 17 / 48
    elseif p == 6
        l = 7
        β = 0.1878687080
        α = 0.3697
        θ_R = 13649 / 43200
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
    
    τR1 = (1/(α*hr))*Is
    τR2 = (1/(α*hr))*Is
    τR3 = (1/(α*hs))*Ir
    τR4 = (1/(α*hs))*Ir
    
    p1 = ((crr[  1, :]) ./ ψ1)
    p2 = ((crr[Nrp, :]) ./ ψ2)
    p3 = ((css[:,   1]) ./ ψ3)
    p4 = ((css[:, Nsp]) ./ ψ4)
    
    P1 = sparse(1:Nsp, 1:Nsp, p1)
    P2 = sparse(1:Nsp, 1:Nsp, p2)
    P3 = sparse(1:Nrp, 1:Nrp, p3)
    P4 = sparse(1:Nrp, 1:Nrp, p4)

    # dynamic penalty matrices
    Γ = (Crr1 * ((2/(θ_R*hr))*Is + τR1 * P1),
         Crr2 * ((2/(θ_R*hr))*Is + τR2 * P2),
         Css3 * ((2/(θ_R*hs))*Ir + τR3 * P3),
         Css4 * ((2/(θ_R*hs))*Ir + τR4 * P4))

    JH = sparse(1:Np, 1:Np, view(J, :)) * (Hs ⊗ Hr)
    
    JIHP = JI * H̃inv * P̃inv
    
    B = ((B1r, B1s), (B2r, B2s), (B3s, B3r), (B4s, B4r))
    nl = (-1, 1, -1, 1)
    G = (-H[1] * (B[1][1] + B[1][2]),
         H[2] * (B[2][1] + B[2][2]),
         -H[3] * (B[3][1] + B[3][2]),
         H[4] * (B[4][1] + B[4][2]))



    static_t = @elapsed begin

        K1 = L[1]' * H[1] * Γ[1] - G[1]'
        K2 = L[2]' * H[2] * Γ[2] - G[2]'
        #K3 = L[3]' * H[3] * Γ[3] - G[3]'
        #K4 = L[4]' * H[4] * Γ[4] - G[4]'
        
        #K1 = L[1]' * H2]
        #K2 = L[2]' * H[2]
        K3 = L[3]' * H[3]
        K4 = L[4]' * H[4]

        M̃ = copy(Ã)
        
        for f in 1:2
            M̃ -= L[f]' * G[f]
            M̃ += L[f]' * H[f] * Γ[f] * L[f]
            M̃ -= G[f]' * L[f]
        end
        
        
        
    end
    
    #@printf "Got quasi-dynamic ops in %f seconds\n" static_t
    # accleration blocks
    Λ_t = @elapsed begin
        dv_u = -Ã
        
        for i in 1:4
            if faces[i] == 0
                dv_u .+= (L[i]' * H[i] * (nl[i] * (B[i][1] + B[i][2]) - Γ[i] * L[i])) +
                    nl[i] * (B[i][1]' + B[i][2]') * H[i] * L[i]
            else
                dv_u .+= (L[i]' * H[i] * ((1 - R[i])/2 .* (nl[i] * (B[i][1] + B[i][2]) - Γ[i] * L[i]))) +
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
                    (L[i]' * H[i] * Γ[i]) -
                    nl[i] * (B[i][1]' + B[i][2]') * H[i]
            else
                dv_û[ : , (i-1) * nn + 1 : i * nn] .=
                    (L[i]' * H[i] * ((1 - R[i])/2 .* Γ[i])) -
                    nl[i] * (B[i][1]' + B[i][2]') * H[i]
                
                dû_u[(i-1) * nn + 1 : i * nn , : ] .=
                    -(1 + R[i])/2 .* (nl[i] * (B[i][1] + B[i][2]) -
                    Γ[i] * L[i])./Z̃f[i]
                
                dû_v[(i-1) * nn + 1 : i * nn , : ] .= (1 + R[i])/2 .* L[i]
            end
        end

        
        dû_û = spzeros(4nn, 4nn)
        for i in 1:4
            if faces[i] != 0
                dû_û[(i-1) * nn + 1 : i * nn, (i-1) * nn + 1 : i * nn] .= -(1 + R[i])/2 .* (Γ[i])./Z̃f[i]
            end
        end

        dû_ψ = spzeros(4nn, nn)

        Λ = [ spzeros(Nn, Nn) sparse(I, Nn, Nn) spzeros(Nn, 5nn)
              dv_u dv_v dv_û spzeros(Nn, nn)
              dû_u dû_v dû_û dû_ψ
              spzeros(nn, 2Nn + 5nn) ]

        nCnΓ1 = Γ[1]
        HIGΓL1 = HI[1] * G[1] - Γ[1] * L[1]
        
    end

    #@printf "Got Λ and friends in %f seconds\n" Λ_t

    
    (Λ = Λ,
     M̃ = cholesky(Symmetric(M̃)),
     K = (K1, K2, K3, K4),
     G = G,
     Γ = Γ,
     HI = HI,
     P̃I = P̃inv,
     H̃ = H̃,
     H̃I = H̃inv,
     JI = JI,
     JIHP = JIHP,
     nCnΓ1 = nCnΓ1,
     HIGΓL1 = HIGΓL1,
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
