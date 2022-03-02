using Printf
using DelimitedFiles
using Plots

get_line(data, i) = map(x -> parse(Float64, x), split.(data[i]))

function get_cycle_indices(slip_data)
    
    cycle_index = [1]
    count = 0
    count1 = 0
    for (i, line) in enumerate(slip_data)
        if line == "BREAK"
            count += 1
            push!(cycle_index, i+1)
        end
        count1 += 1
    end
    push!(cycle_index, count1)

    return count, cycle_index

end

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
    ic_t_file = params[12]
    Dc = parse(Float64, params[13])
    B_on = parse(Int64, params[14])
    return (p, T, N, Lw, r̂, l, b_depth, dynamic_flag, d_to_s, dt_scale, ic_file, ic_t_file, Dc, B_on)
end



function plot_slip(slip_data, final_index, title; index_offset=0)

    nn = parse(Int64, slip_data[1])
    y = get_line(slip_data, 2)[3:end]

    plt = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Slip(m)", title=title)

    δ = zeros(nn)
    break_count = 0
    if index_offset != 0
        δ_off = get_line(slip_data, index_offset)[3:end]
    else
        δ_off = zeros(nn)
        index_offset = 3
    end
    for index in index_offset:final_index

        @printf "\r%f" (index - index_offset)/(final_index - index_offset)
        if slip_data[index] == "BREAK"
            plt = plot!(δ-δ_off, y, linecolor=:black, linewidth = 1)
            break_count += 1
        else
             if break_count % 2 == 0
                δ .= get_line(slip_data, index)[3:end]
                plt = plot!(δ-δ_off, y, linecolor=:blue, linewidth = .2)
            elseif break_count % 2 == 1 && index % 3 == 1 
                δ .= get_line(slip_data, index)[3:end]
                plt = plot!(δ-δ_off, y, linecolor=:red, linewidth = .2)
            end 
        end
        
    end

    return plt

end


function get_plot_indices(cycle_offset, final_cycle, cycle_index, total_cycles)

    final_index = 0
    index_offset = 0
    if final_cycle < total_cycles
        final_index = cycle_index[2*final_cycle+1]
    else
        final_index = cycle_index[end]
    end
    
    if cycle_offset == 0
        index_offset = 3
    elseif cycle_offset == length(cycle_index)
        index_offset = cycle_index[2*cycle_offset+1]
    elseif cycle_offset == -1
        index_offset = 0
    else
        index_offset = cycle_index[2*cycle_offset+1] + 100
    end

    return final_index, index_offset

end
