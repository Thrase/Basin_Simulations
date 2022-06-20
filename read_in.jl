# Function for reading in numerical parameters for basin simulations
function read_params(f_name)
    f = open(f_name, "r")
    params = []
    while ! eof(f)
        s = readline(f)
        if s[1] != '#'
            push!(params, split(s, '=')[2])
            flush(stdout)
        end
    end
    close(f)
    p = parse(Int64, params[1])
    T = parse(Float64, params[2])
    N = parse(Int64, params[3])
    Lw = parse(Float64, params[4])
    r̂ = parse(Float64, params[5])
    l = parse(Float64, params[6])
    b_depth = parse(Float64, params[7])
    dynamic_flag = parse(Int64,params[8])
    d_to_s = parse(Float64, params[9])
    dt_scale = parse(Float64, params[10])
    ic_file = params[11]
    ic_remote =params[12]
    ic_t_file = params[13]
    Dc = parse(Float64, params[14])
    B_on = parse(Int64, params[15])
    dir_out = params[16]
    volume_plots = parse(Int64, params[17])
    cycle_flag = parse(Int64, params[18])
    num_cycles = parse(Int64, params[19])
    intime_plotting = parse(Int64, params[20])
    μ_in = parse(Float64, params[21])
    return p, T, N, Lw, r̂, l, b_depth, dynamic_flag, d_to_s, dt_scale, ic_file, ic_remote, ic_t_file, Dc, B_on, dir_out, volume_plots, cycle_flag, num_cycles, intime_plotting, μ_in
end
