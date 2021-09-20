# Function for reading in numerical parameters for basin simulations
function read_params(f_name)
    f = open(f_name, "r")
    params = []
    while ! eof(f)
        s = readline(f)
        if s[1] != '#'
            push!(params, split(s, '=')[2])
        end
    end
    close(f)
    T = parse(Float64, params[1])
    N = parse(Int64, params[2])
    Lw = parse(Float64, params[3])
    trans_flag = params[4]
    r̂ = parse(Float64, params[5])
    l = parse(Float64, params[6])
    b_depth = parse(Float64, params[7])
    dynamic_flag = parse(Int64,params[8])
    d_to_s = parse(Float64, params[9])
    dt_scale = parse(Float64, params[10])
    ic_file = params[11]
    ic_t_file = params[12]

    return T, N, Lw, trans_flag, r̂, l, b_depth, dynamic_flag, d_to_s, dt_scale, ic_file, ic_t_file
end
