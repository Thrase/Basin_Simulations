include("../physical_params.jl")
using Plots
using ForwardDiff: derivative
##################
#  MMS Function  #
##################

f(a, MMS) = MMS.amp * sin.(π.*(a)/MMS.wl)
fp(a, MMS) = MMS.amp * π/MMS.wl*cos.(π.*(a)/MMS.wl)
fpp(a, MMS) = MMS.amp * (-(π/MMS.wl)^2) .* sin.(π.*(a)/MMS.wl)

ue(x, y, t, MMS) = f(x .+ y .- t, MMS) .+ (MMS.amp * π/MMS.wl + MMS.ϵ)*t .+ (MMS.amp * π/MMS.wl + MMS.ϵ)*x

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


function ψe(fx, fy, t, B_p, RS, MMS)

    Ve = 2*ue_t(fx, fy, t, MMS)
    τf = τe(fx, fy, t, 1, B_p, MMS)
    return RS.a .* log.((2 * RS.V0 ./ Ve) .* sinh.(-τf ./ (RS.a .* RS.σn)))
    
end

function ψe_t(fx, fy, t, B_p, RS, MMS)

    Ve = 2*ue_t(fx, fy, t, MMS)
    Ve_t = 2*ue_tt(fx, fy, t, MMS)
    τf = τe(fx, fy, t, 1, B_p, MMS)
    τf_t = τe_t(fx, fy, t, 1, B_p, MMS)
    
    return τf_t ./ RS.σn .* coth.(τf ./ (RS.σn .* RS.a)) - RS.a .* Ve_t ./ Ve

end

function Forcing(x, y, t, B_p, MMS)
        
    Force = ρ(x, y, B_p) .* ue_tt(x, y, t, MMS) .-
        (μ_x(x, y, B_p) .* ue_x(x, y, t, MMS) .+ μ(x, y, B_p) .* ue_xx(x, y, t, MMS) .+
         μ_y(x, y, B_p) .* ue_y(x, y, t, MMS) .+ μ(x, y, B_p) .* ue_yy(x, y, t, MMS))

    return Force
end

function S_c(fx, fy, t, fnum, R, B_p, MMS)
       
    Z = sqrt.(μ(fx, fy, B_p) .* ρ(fx, fy, B_p))
    v = ue_t(fx, fy, t, MMS)
    τ = τe(fx, fy, t, fnum, B_p, MMS)

    return Z .* v .+ τ .- R .* (Z .* v .- τ)
end


function S_rs(fx, fy, b, t, B_p, RS, MMS)
    
    ψ = ψe(fx, fy, t, B_p, RS, MMS)
    V = 2*ue_t(fx, fy, t, MMS)
    G = (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ) ./ b) .- abs.(V) / RS.V0)
    ψ_t = ψe_t(fx, fy, t, B_p, RS, MMS)
    return  ψ_t .- G

end


function ϕ(x, y, MMS)
    h = MMS.H
    return (h .* (h .+ x)) ./
        ((h .+ x).^2 .+ y.^2)
end

function ϕ_x(x, y, MMS)
    h = MMS.H
    return (h .* (y.^2 .- (h .+ x).^2)) ./ ((h .+ x).^2 .+ y.^2).^2
end

function ϕ_xx(x, y, MMS)
    h = MMS.H
    return (2*h .* (h .+ x) .* (h^2 .+ 2 * h .* x .+ x.^2 .- 3 .* y.^2)) ./
        (h^2 .+ 2 * h .* x .+ x.^2 .+ y.^2).^3
end

function ϕ_y(x, y, MMS)
    h = MMS.H
    return - (2 * h .* y .* (h .+ x)) ./
        ((h .+ x).^2 .+ y.^2).^2
end

function ϕ_yy(x, y, MMS)
    h = MMS.H
    return (2*h .* (h .+ x) .* (3 .* y.^2 .- (h .+ x).^2)) ./
        ((h .+ x).^2 .+ y.^2).^3
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

    return tw / (π * (t̄ - 2t̄ + t̄^2 + t^2)) + Vm/δ 
end

function K_tt(t, MMS)
    t̄ = MMS.t̄
    tw = MMS.t_w
    δ = MMS.δ_e
    Vm = MMS.Vmin

    return (2 * tw * (t̄ -t)) / (π * (t̄ - 2t̄ + t̄^2 + t^2)^2)
end


he(x, y, t, MMS) = MMS.δ_e ./2 .* K(t, MMS) .* ϕ(x, y, MMS) .+ MMS.Vp ./2 .* t .* (1 .- ϕ(x,y,MMS)) .+
    MMS.τ∞/36 .* x

he_t(x, y, t, MMS) = MMS.δ_e ./2 .* K_t(t, MMS) .* ϕ(x, y, MMS) .+ MMS.Vp ./2 .* (1 .- ϕ(x,y,MMS))
he_tt(x, y, t, MMS) = MMS.δ_e ./2 .* K_tt(t, MMS) .* ϕ(x, y, MMS)

he_x(x, y, t, MMS) =  MMS.δ_e .* K(t, MMS) .* ϕ_x(x, y, MMS) .- MMS.Vp/2 .* ϕ_x(x, y, MMS) .+ MMS.τ∞/36
he_xx(x, y, t, MMS) =  MMS.δ_e .* K(t, MMS) .* ϕ_xx(x, y, MMS) .- MMS.Vp/2 .* ϕ_xx(x, y, MMS)

he_y(x, y, t, MMS) =  MMS.δ_e .* K(t, MMS) .* ϕ_y(x, y, MMS) .- MMS.Vp/2 .* ϕ_y(x, y, MMS)
he_yy(x, y, t, MMS) =  MMS.δ_e .* K(t, MMS) .* ϕ_yy(x, y, MMS) .- MMS.Vp/2 .* ϕ_yy(x, y, MMS)

he_xt(x, y, t, MMS) = MMS.δ_e .* K_t(t, MMS) .* ϕ_x(x, y, MMS) .- MMS.Vp/2 .* ϕ_x(x, y, MMS)


h_FORCE(x, y, t, B_p, MMS) = -(μ_x(x, y, B_p) .* he_x(x, y, t, MMS) .+
    μ(x, y, B_p) .* he_xx(x, y, t, MMS) .+
    μ_y(x, y, B_p) .* he_y(x, y, t, MMS) .+
    μ(x, y, B_p) .* he_yy(x, y, t, MMS))


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

function ψe_2(x, y, t, B_p, RS, MMS)

    τe = τhe(x, y, t, 1, B_p, MMS)
    Ve = 2 .* he_t(x, y, t, MMS)

    return RS.a .* log.((2 * RS.V0 ./ Ve) .* sinh.(-τe ./ (RS.a .* RS.σn))) .- η(y, B_p) .* Ve
    
end

function ψe_2t(x, y, t, B_p, RS, MMS)

    τe = τhe(x, y, t, 1, B_p, MMS)
    Ve = 2 * he_t(x, y, t, MMS)
    Ve_t = 2 * he_tt(x, y, t, MMS)
    τe_t = - μ(x, y, B_p) .* he_xt(0, y, t, MMS)
    
    return τe_t ./ RS.σn .* coth.(τe ./ (RS.σn .* RS.a)) - RS.a .* Ve_t ./ Ve .- η(y, B_p) .* Ve_t
end

function fault_force(x, y, t, b, B_p, RS, MMS)

    ψe = ψe_2(x, y, t, B_p, RS, MMS)
    Ve = 2 * he_t(x, y, t, MMS)
    G = (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψe) ./ b) .- abs.(Ve) / RS.V0)
    
    return ψe_2t(x, y, t, B_p, RS, MMS) .- G

end

function h_face2(x, y, t, MMS)
    return he(x, y, t, MMS) .- MMS.Vp/2 * t
end


Pe(x, y, t, MMS) = sin.(π/MMS.Lw .* x) .* cos.(π/MMS.Lw .* y) .* sin.(π/MMS.Lw .* t)

Pe_y(x, y, t, MMS) = - π/MMS.Lw .* sin.(π/MMS.Lw * x) .* sin.(π/MMS.Lw * y) .* sin.(π/MMS.Lw .* t)

Pe_yy(x, y, t, MMS) = - π^2/MMS.Lw^2 .* sin.(π/MMS.Lw * x) .* cos.(π/MMS.Lw * y) .* sin.(π/MMS.Lw .* t)

Pe_x(x, y, t, MMS) = π/MMS.Lw * cos.(π/MMS.Lw * x) .* cos.(π/MMS.Lw * y) .* sin.(π/MMS.Lw .* t)

Pe_xx(x, y, t, MMS) = - π^2/MMS.Lw^2 * sin.(π/MMS.Lw * x) .* cos.(π/MMS.Lw * y) .* sin.(π/MMS.Lw .* t)

Pe_t(x, y, t, MMS) = π/MMS.Lw .* sin.(π/MMS.Lw .* x) .* cos.(π/MMS.Lw .* y) .* cos.(π/MMS.Lw .* t)

Pe_tt(x, y, t, MMS) = - π^2/MMS.Lw^2 .* sin.(π/MMS.Lw .* x) .* cos.(π/MMS.Lw .* y) .* sin.(π/MMS.Lw .* t)

P_FORCE(x, y, t, B_p, MMS) = - (μ_x(x, y, B_p) .* Pe_x(x, y, t, MMS) .+
    μ(x, y, B_p) .* Pe_xx(x, y, t, MMS) .+
    μ_y(x, y, B_p) .* Pe_y(x, y, t, MMS) .+
    μ(x, y, B_p) .* Pe_yy(x, y, t, MMS))

P_face3(x, y, t, B_p, MMS) = - μ(x, y, B_p) .* Pe_y(x, y, t, MMS)

P_face4(x, y, t, B_p, MMS) = μ(x, y, B_p) .* Pe_y(x, y, t, MMS)

P_face2(x, y, t, MMS) = Pe(x, y, t, MMS) .- t

P_face1(x, y, t, MMS) = Pe(x, y, t, MMS) .- 1/2 * t^2
