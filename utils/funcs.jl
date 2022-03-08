using Printf
using DelimitedFiles
using Plots

get_line(data, i) = map(x -> parse(Float64, x), split.(data[i]))

function get_y_nn(slip_data)

    nn = parse(Int64, split.(Iterators.take(slip_data, 1))[1][1])
    y = map(x -> parse(Float64, x), split.(slip_data[2]))[3:end]
    slip_data = @view slip_data[3:end, :]

    return y, nn, slip_data

end


function get_break_indices(slip_data)
    
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



function plot_slip(slip_data, y, nn, title)

    plt = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Slip(m)", title=title)

    δ = zeros(nn)
    break_count = 0
    δ_off = get_line(slip_data, 1)[3:end]

    final_index = size(slip_data)[1]
    for index in 1:final_index

        @printf "\r%f" index/final_index
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


function get_plot_indices(cycle_offset, final_cycle, break_indices)

    index_offset = break_indices[2*cycle_offset - 1]
    if 2*final_cycle + 1 <= length(break_indices)
        final_index = break_indices[2*final_cycle + 1] - 2
    else
        final_index = break_indices[2*final_cycle] - 2
    end

    return final_index, index_offset

end
