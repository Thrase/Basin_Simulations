function μ(x, y, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w

    
    if ndims(x) == 2
        return repeat([μ_out], outer=size(x))
    else
        return repeat([μ_out], outer=length(x))
    end
    

    
    #return (μ_out - μ_in)/2 *
        #(tanh.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .+ 1) .+ μ_in
    
end

function ρ(x, y, B_p)

    c = B_p.c
    ρ_in = B_p.ρ_in
    ρ_out = B_p.ρ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w

    
    if ndims(x) == 2
        return repeat([ρ_out], outer=size(x))
    else
        return repeat([ρ_out], outer=length(x))
    end
    

    
    #return (ρ_out - ρ_in)/2 *
    #    (tanh.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .+ 1) .+ ρ_in
    
end

function μ_x(x, y, B_p)
    
    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    
    #=
    if ndims(x) == 2
        return zeros(size(x))
    else
        return zeros(length(x))
    end
    =#

    return ((μ_out - μ_in) .* x .*
        sech.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w
end

function μ_y(x, y, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w

    #=
    if ndims(x) == 2
        return zeros(size(x))
    else
        return zeros(length(x))
    end
    =#

    return ((μ_out - μ_in) .* (c^2 * y) .*
        sech.((x .^ 2 + c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w
end


function η(y, B_p)
    μf = μ(0, y, B_p)
    return μf ./ (2 .* sqrt.(μf ./ ρ.(0, y, B_p)))
end



function fault_params(fc, Dc)

    
    Wf = 24
    Hvw = 12
    Ht = 6
    δNp = findmin(abs.(Wf .- fc))[2]
    gNp = findmin(abs.(16 .- fc))[2]
    VWp = findmin(abs.((Hvw + Ht) .- fc))[2]
    a = .015
    b0 = .02
    bmin = 0.0
    
    
    function b_fun(y)
        if 0 <= y < Hvw
            return b0
        end
        if Hvw <= y < Hvw + Ht
            return b0 + (bmin - b0)*(y-Hvw)/Ht
        end
        if Hvw + Ht <= y < Wf
            return bmin
        end
        return bmin
    end

    
    RS = (σn = 50.0,
          a = a,
          b = b_fun.(fc[1:δNp]),
          Dc = Dc,
          f0 = .6,
          V0 = 1e-6,
          τ_inf = 24.82,
          Vp = 1e-9)


    return δNp, gNp, VWp, RS
    
end
