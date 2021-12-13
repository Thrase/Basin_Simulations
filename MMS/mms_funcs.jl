include("../physical_params.jl")

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


ϕ(x, y, MMS) = (MMS.H*(MMS.H + x))/((MMS.H + x)^2 + y^2)

ϕ_x(x, y, MMS) = MMS.H * (y^2 - (MMS.H + x)^2)/(((MMS.H + x)^2 + y^2)^2)

ϕ_xx(x, y, MMS) = (2*MMS.H * (H + x)(MMS.H^2 + 2*MMS.H*x + x^2 + 3y^2))/((MMS.H^2 + 2MMS.H*x + x^2 + y^2)^3)

ϕ_y(x, y, MMS) = -(2*MMS.H*y*(MMS.H + x))/(((MMS.H + x)^2 + y^2)^2)

ϕ_yy(x, y, MMS) = (2 * MMS.H * (MMS.H + x)*(3*y^2 - (MMS.H + x)^2))/(((MMS.H+x)^2 + y^2)^3)

K(t, MMS) = 1/π * (atan((t - MMS.t̄)/MMS.t_w) + π/2) + MMS.Vmin/MMS.δ_e * t

K_t(t, MMS) = MMS.t_w/(π * (MMS.t̄^2 - 2*MMS.t̄*t + MMS.t_w^2 + t^2)) + MMS.Vmin/MMS.δ_e
                  
K_tt(t, MMS) = 2*MMS.t_w*(MMS.t̄ - t)/(π * (MMS.t̄^2 - 2MMS.t̄*t + MMS.t_w^2 + t^2)^2)

h_e(x, y, t, MMS) = MMS.δ_e/2 * K(t, MMS)*ϕ(x,y) + MMS.Vp/2*t(1 - ϕ(x,y,MMS)) + MMS.τ∞/24*x

h_et(x, y, t, MMS) = MMS.δ_e/2 * K_t(t, MMS)*ϕ(x,y) + MMS.Vp/2(1 - ϕ(x,y,MMS))

h_ett(x, y, t, MMS) = MMS.δ_e/2 * K_tt(t, MMS)*ϕ(x,y, MMS)

h_ex(x, y, t, MMS) = MMS.δ_e/2 * K(t, MMS)*ϕ_x(x,y, MMS) - MMS.Vp/2*t*ϕ_x(x, y, MMS) + MMS.τ∞/24

h_ext(x, y, t, MMS) = MMS.δ_e/2 * K_t(t, MMS)*ϕ_x(x,y, MMS) - MMS.Vp/2 * ϕ_x(x, y, MMS)

h_exx(x, y, t, MMS) = MMS.δ_e/2 * K(t, MMS)*ϕ_xx(x,y, MMS) - MMS.Vp/2*t*ϕ_xx(x, y, MMS)

h_ey(x , y, t, MMS) = MMS.δ_e/2 * K(t, MMS)*ϕ_y(x, y, MMS) - ϕ_y(x, y, MMS)

h_eyy(x , y, t, MMS) = MMS.δ_e/2 * K(t, MMS)*ϕ_yy(x, y, MMS) - ϕ_yy(x, y, MMS)


h_FORCE(x, y, t, B_p, MMS) = (μ_x(x, y, B_p) .* ue_x(x, y, t, MMS) .+ μ(x, y, B_p) .* ue_xx(x, y, t, MMS) .+
                       μ_y(x, y, B_p) .* ue_y(x, y, t, MMS) .+ μ(x, y, B_p) .* ue_yy(x, y, t, MMS))

function ψe_2(y, t, B_p, RS, MMS)

    τe = μ(0, y, B_p) * h_ex(0, y, t, MMS)
    Ve = 2 * h_et(0, y, t, MMS)

    return RS.a .* log.((2 * RS.V0 ./ Ve) .* sinh.(-τe ./ (RS.a .* RS.σn))) - η(y, B_p) * Ve
end

function ψe_2t(y, t, B_p, RS, MMS)

    τe = - μ(0, y, B_p) * h_ex(0, y, t, MMS)
    Ve = 2 * h_et(0, y, t, MMS)
    Ve_t = 2 * h_ett(0, y, t, MMS)
    τe_t = - μ(0, y, B_p) * h_ext(0, y, t, MMS)
    
    return τf_t ./ RS.σn .* coth.(τe ./ (RS.σn .* RS.a)) - RS.a .* Ve_t ./ Ve - η(y, B_p) * Vett
end

function fault_force(y, t, b, B_p, RS, MMS)

    ψe = ψe_2(y, t, B_p, RS, MMS)
    Ve = 2 * he_t(0, y, t, MMS)
    G = (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ) ./ b) .- abs.(Ve) / RS.V0)
    
    return ψe_2t(y, t, B_p, RS, MMS) - G
end




