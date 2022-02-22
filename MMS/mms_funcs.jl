include("../physical_params.jl")
using Plots

##################
#  MMS Function  #
##################

#=
function ϕ(x, y, MMS)
    h = MMS.H
    return (h .* (h .+ x)) ./
        ((h .+ x).^2 .+ y.^2)
end

function ϕ_x(x, y, MMS)
    h = MMS.H
    return (h .* (y.^2 .- (h .+ x).^2)) ./ (((h .+ x).^2 .+ y.^2).^2)
end

function ϕ_xx(x, y, MMS)
    h = MMS.H
    return (2*h .* (h .+ x) .* (h^2 .+ 2*h .* x .+ x.^2 .- 3*y.^2)) ./
        ((h^2 .+ 2*h .* x .+ x.^2 .+ y.^2).^3)
end

function ϕ_y(x, y, MMS)
    h = MMS.H
    return - (2 * h .* y .* (h .+ x)) ./
        ((h .+ x).^2 .+ y.^2).^2
end

function ϕ_yy(x, y, MMS)
    h = MMS.H
    return (2*h .* (h .+ x) .* (3 .* y.^2 .- (h .+ x).^2)) ./
        (((h .+ x).^2 .+ y.^2).^3)
end

function K(t, MMS)
    
    t̄ = MMS.t̄
    tw = MMS.t_w
    δ = MMS.δ_e
    Vm = MMS.Vmin
    
    return 1/π * (atan((t - t̄)/tw) + π/2) + Vm/δ * t
end

function K_t(t, MMS)
    
    t̄ = MMS.t̄
    tw = MMS.t_w
    δ = MMS.δ_e
    Vm = MMS.Vmin
    
    return tw / (π * ((t - t̄)^2 + tw^2)) + Vm/δ
end

function K_tt(t, MMS)
    
    t̄ = MMS.t̄
    tw = MMS.t_w
    δ = MMS.δ_e
    Vm = MMS.Vmin

    return -(2 * tw * (t - t̄)) / (π * ((t - t̄)^2 + tw^2)^2)
end

he(x, y, t, MMS) = MMS.δ_e/2 .* K(t, MMS) .* ϕ(x, y, MMS) .+ MMS.Vp/2 .* t .* (1 .- ϕ(x,y,MMS)) .+
    MMS.τ∞/24 .* x

he_t(x, y, t, MMS) = MMS.δ_e/2 .* K_t(t, MMS) .* ϕ(x, y, MMS) .+ MMS.Vp/2 .* (1 .- ϕ(x,y,MMS))

he_tt(x, y, t, MMS) = MMS.δ_e/2 .* K_tt(t, MMS) .* ϕ(x, y, MMS)

he_x(x, y, t, MMS) =  MMS.δ_e/2 .* K(t, MMS) .* ϕ_x(x, y, MMS) .- MMS.Vp/2 .* t .* ϕ_x(x, y, MMS) .+ MMS.τ∞/24

he_xx(x, y, t, MMS) =  MMS.δ_e/2 .* K(t, MMS) .* ϕ_xx(x, y, MMS) .- MMS.Vp/2 .* t .* ϕ_xx(x, y, MMS)

he_y(x, y, t, MMS) =  MMS.δ_e/2 .* K(t, MMS) .* ϕ_y(x, y, MMS) .- MMS.Vp/2 .* t .* ϕ_y(x, y, MMS)

he_yy(x, y, t, MMS) =  MMS.δ_e/2 .* K(t, MMS) .* ϕ_yy(x, y, MMS) .- MMS.Vp/2 .* t .* ϕ_yy(x, y, MMS)

he_xt(x, y, t, MMS) = MMS.δ_e/2 .* K_t(t, MMS) .* ϕ_x(x, y, MMS) .- MMS.Vp/2 .* ϕ_x(x, y, MMS)


function τhe(fx, fy, t, fnum, B_p, MMS)

    if fnum == 1
        τ = -μ(fx, fy, B_p) .* he_x(fx, fy, t, MMS)
    elseif fnum == 2
        τ = μ(fx, fy, B_p) .* he_x(fx, fy, t, MMS)
    elseif fnum == 3
        τ = -μ(fx, fy, B_p) .* he_y(fx, fy, t, MMS)
    elseif fnum == 4 
        τ = μ(fx, fy, B_p) .* he_y(fx, fy, t, MMS)
    end
    return τ

end

### Dynamic h solution forcing functions

function Forcing_hd(x, y, t, B_p, MMS)
        
    Force = ρ(x, y, B_p) .* he_tt(x, y, t, MMS) .-
        (μ_x(x, y, B_p) .* he_x(x, y, t, MMS) .+ μ(x, y, B_p) .* he_xx(x, y, t, MMS) .+
         μ_y(x, y, B_p) .* he_y(x, y, t, MMS) .+ μ(x, y, B_p) .* he_yy(x, y, t, MMS))

    return Force
end

function S_ch(fx, fy, t, fnum, R, B_p, MMS)
       
    Z = sqrt.(μ(fx, fy, B_p) .* ρ(fx, fy, B_p))
    v = he_t(fx, fy, t, MMS)
    τ = τhe(fx, fy, t, fnum, B_p, MMS)

    return Z .* v .+ τ .- R .* (Z .* v .- τ)
end

function S_rsdh(fx, fy, b, t, B_p, RS, MMS)
    
    ψ = ψe_hd(fx, fy, t, B_p, RS, MMS)
    V = 2*he_t(fx, fy, t, MMS)
    G = (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ) ./ b) .- abs.(V) / RS.V0)
    ψ_t = ψe_thd(fx, fy, t, B_p, RS, MMS)
    return  ψ_t .- G

end


function ψe_hd(x, y, t, B_p, RS, MMS)
    
    τe = τhe(x, y, t, 1, B_p, MMS)
    Ve = 2 .* he_t(x, y, t, MMS)

    return RS.a .* log.((2 * RS.V0 ./ Ve) .* sinh.(-τe ./ (RS.a .* RS.σn)))
end


function ψe_thd(x, y, t, B_p, RS, MMS)

    τe = τhe(x, y, t, 1, B_p, MMS)
    Ve = 2 * he_t(x, y, t, MMS)
    Ve_t = 2 * he_tt(x, y, t, MMS)
    τe_t = - μ(x, y, B_p) .* he_xt(x, y, t, MMS)

    ψ_t = τe_t .* coth.(τe ./ (RS.a * RS.σn)) ./ RS.σn .- RS.a .* Ve_t ./ Ve

    return ψ_t

end

### Quasi-dynamic h solution forcing functions

Forcing_h(x, y, t, B_p, MMS) = -(μ_x(x, y, B_p) .* he_x(x, y, t, MMS) .+
    μ(x, y, B_p) .* he_xx(x, y, t, MMS) .+
    μ_y(x, y, B_p) .* he_y(x, y, t, MMS) .+
    μ(x, y, B_p) .* he_yy(x, y, t, MMS))






=#


f(a, MMS) = MMS.amp * sin.(π.*(a)/MMS.wl)
fp(a, MMS) = MMS.amp * π/MMS.wl*cos.(π.*(a)/MMS.wl)
fpp(a, MMS) = MMS.amp * (-(π/MMS.wl)^2) .* sin.(π.*(a)/MMS.wl)

ue(x, y, t, MMS) = f(x .+ y .- t, MMS) .+ (MMS.amp * π/MMS.wl + MMS.ϵ)*t .+ (MMS.amp * π/MMS.wl 
                                                                             + MMS.ϵ)*x

ue_t(x, y, t, MMS) = -fp(x .+ y .- t, MMS) .+ (MMS.amp * π/MMS.wl + MMS.ϵ)
ue_tt(x, y, t, MMS) = fpp(x .+ y .- t, MMS)

ue_x(x, y, t, MMS) = fp(x .+ y .- t, MMS) .+ (MMS.amp * π/MMS.wl + MMS.ϵ)
ue_y(x, y, t, MMS) = fp(x .+ y .- t, MMS)
ue_xy(x,y, t, MMS) = fpp(x .+ y .- t, MMS)
ue_xx(x, y, t, MMS) = fpp(x .+ y .- t, MMS)
ue_yy(x, y, t, MMS) = fpp(x .+ y .- t, MMS)
ue_xt(x, y, t, MMS) = -fpp(x .+ y .- t, MMS)
ue_yt(x, y, t, MMS) = -fpp(x .+ y .- t, MMS)

function τe(fx, fy, t, fnum, B_p, MMS)

    if fnum == 1
        τ = -μ(fx, fy, B_p) .* ue_x(fx, fy, t, MMS)
    elseif fnum == 2
        τ = μ(fx, fy, B_p) .* ue_x(fx, fy, t, MMS)
    elseif fnum == 3
        τ = -μ(fx, fy, B_p) .* ue_y(fx, fy, t, MMS)
    elseif fnum == 4 
        τ = μ(fx, fy, B_p) .* ue_y(fx, fy, t, MMS)
    end
    return τ

end

function τe_t(fx, fy, t, fnum, B_p, MMS)

    if fnum == 1
        τ = -μ(fx, fy, B_p) .* ue_xt(fx, fy, t, MMS)
    elseif fnum == 2
        τ = μ(fx, fy, B_p) .* ue_xt(fx, fy, t, MMS)
    elseif fnum == 3
        τ = -μ(fx, fy, B_p) .* ue_yt(fx, fy, t, MMS)
    elseif fnum == 4 
        τ = μ(fx, fy, B_p) .* ue_yt(fx, fy, t, MMS)
    end

    return τ   
 
end

### Dynamic u solution forcing functions

function ψe_d(fx, fy, t, B_p, RS, MMS)

    Ve = 2*ue_t(fx, fy, t, MMS)
    τf = τe(fx, fy, t, 1, B_p, MMS)
    
    return RS.a .* log.((2 * RS.V0 ./ Ve) .* sinh.(-τf ./ (RS.a .* RS.σn)))
    
end

function ψe_dt(fx, fy, t, B_p, RS, MMS)

    Ve = 2*ue_t(fx, fy, t, MMS)
    Ve_t = 2*ue_tt(fx, fy, t, MMS)
    τf = τe(fx, fy, t, 1, B_p, MMS)
    τf_t = τe_t(fx, fy, t, 1, B_p, MMS)
    
    return τf_t ./ RS.σn .* coth.(τf ./ (RS.σn .* RS.a)) - RS.a .* Ve_t ./ Ve

end

function Forcing_ud(x, y, t, B_p, MMS)
        
    Force = ρ(x, y, B_p) .* ue_tt(x, y, t, MMS) .-
        (μ_x(x, y, B_p) .* ue_x(x, y, t, MMS) .+ μ(x, y, B_p) .* ue_xx(x, y, t, MMS) .+
         μ_y(x, y, B_p) .* ue_y(x, y, t, MMS) .+ μ(x, y, B_p) .* ue_yy(x, y, t, MMS))

    return Force
end

function S_cd(fx, fy, t, fnum, R, B_p, MMS)
       
    Z = sqrt.(μ(fx, fy, B_p) .* ρ(fx, fy, B_p))
    v = ue_t(fx, fy, t, MMS)
    τ = τe(fx, fy, t, fnum, B_p, MMS)

    return Z .* v .+ τ .- R .* (Z .* v .- τ)
end


function S_rsd(fx, fy, b, t, B_p, RS, MMS)
    
    ψ = ψe(fx, fy, t, B_p, RS, MMS)
    V = 2*ue_t(fx, fy, t, MMS)
    G = (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ) ./ b) .- abs.(V) / RS.V0)
    ψ_t = ψe_t(fx, fy, t, B_p, RS, MMS)
    return  ψ_t .- G

end


### Quasi-dynamic u solution forcing functions

Forcing_u(x, y, t, B_p, MMS) = -(μ_x(x, y, B_p) .* ue_x(x, y, t, MMS) .+
    μ(x, y, B_p) .* ue_xx(x, y, t, MMS) .+
    μ_y(x, y, B_p) .* ue_y(x, y, t, MMS) .+
    μ(x, y, B_p) .* ue_yy(x, y, t, MMS)) 

function ψe(x, y, t, B_p, RS, MMS)

    τ = τe(x, y, t, 1, B_p, MMS)
    Ve = 2 .* ue_t(x, y, t, MMS)

    return RS.a .* log.((2 * RS.V0 ./ Ve) .* sinh.((-τ .- η(y, B_p) .* Ve) ./ (RS.a .* RS.σn)))
    
end

function ψe_t(x, y, t, B_p, RS, MMS)

    τ = τe(x, y, t, 1, B_p, MMS)
    Ve = 2 * ue_t(x, y, t, MMS)
    Ve_t = 2 * ue_tt(x, y, t, MMS)
    τe_t = - μ(x, y, B_p) .* ue_xt(x, y, t, MMS)

    ψ_t = (-τe_t .- η(y, B_p) .* Ve_t) .* coth.((-τ .- η(y, B_p) .* Ve) ./ (RS.a * RS.σn)) ./ RS.σn .- RS.a .* Ve_t ./ Ve

    return ψ_t

end


function S_rs(x, y, t, b, B_p, RS, MMS)

    ψ = ψe(x, y, t, B_p, RS, MMS)
    Ve = 2 * ue_t(x, y, t, MMS)
    G = (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ) ./ b) .- abs.(Ve) / RS.V0)

    s_rs = ψe_t(x, y, t, B_p, RS, MMS) .- G

    return s_rs

end

function u_face2(x, y, t, MMS, RS, μf2)
    return ue(x, y, t, MMS) .- (MMS.Vp/2 * t .+ (RS.τ_inf * MMS.Lw) ./ μf2)
                                
end
