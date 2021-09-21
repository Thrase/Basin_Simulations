function μ(x, y, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w

    return (μ_out - μ_in)/2 *
        (tanh.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .+ 1) .+ μ_in
end

function ρ(x, y, B_p)

    c = B_p.c
    ρ_in = B_p.ρ_in
    ρ_out = B_p.ρ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w

    return (ρ_out - ρ_in)/2 *
        (tanh.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .+ 1) .+ ρ_in
end

function μ_x(x, y, B_p)
    
    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    
    return ((μ_out - μ_in) .* x .*
        sech.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w
end

function μ_y(x, y, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    
    return ((μ_out - μ_in) .* (c^2 * y) .*
        sech.((x .^ 2 + c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w
end


function η(y, B_p)
    μf = μ(0, y, B_p)
    return μf ./ (2 .* sqrt.(μf ./ ρ.(0, y, B_p)))
end


function b_fun(yf, RS)
    
    b = Array{Float64,1}(undef, length(yf))
    depth = yf[end]

    for i in 1:length(yf)
        #if 0 <= yf[i] < RS.Hvw
            b[i] =  RS.b0
        #end
        #if RS.Hvw <= yf[i] < RS.Hvw + RS.Ht
        #    b[i] = RS.b0 + (RS.bmin - RS.b0)*(yf[i] - RS.Hvw) / RS.Ht
        #end
        #if RS.Hvw + RS.Ht <= yf[i] < depth
        #    b[i] = RS.bmin
        #end
    end
    return b

end
