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
    #display(τf)
    return RS.a .* log.((2 * RS.V0 ./ Ve) .* sinh.(-τf ./ (RS.a .* RS.σn)))
    
end

function ψe_t(fx, fy, t, B_p, RS, MMS)

    Ve = 2*ue_t(fx, fy, t, MMS)
    Ve_t = 2*ue_tt(fx, fy, t, MMS)
    τf = τe(fx, fy, t, 1, B_p, MMS)
    τf_t = τe_t(fx, fy, t, 1, B_p, MMS)
    
    return τf_t ./ RS.σn .* coth.(τf ./ (RS.σn .* RS.a)) - (RS.a .* Ve_t ./ Ve)

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


function S_rs(fx, fy, b, t, fnum, B_p, RS, MMS)
    
    ψ = ψe(fx, fy, t, B_p, RS, MMS)
    V = 2*ue_t(fx, fy, t, MMS)
    G = (b .* RS.V0 ./ RS.Dc) .* (exp.((RS.f0 .- ψ) ./ b) .- abs.(V) ./ RS.V0)
    ψ_t = ψe_t(fx, fy, t, B_p, RS, MMS)
    return  ψ_t .- G

end
