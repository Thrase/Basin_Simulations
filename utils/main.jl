# Program for plotting Basin output, and getting new initial conditions.
using Printf
using Plots

let 
    
    v = false
    while v == false
        @printf "\nOutput directory: "
        folder_name = chomp(readline())
        try
            cd(string("../../erickson/output_files/", folder_name))
            v = true
        catch
            @printf "Directory not found\n"
        end
    end
    @printf "Found directory...\n\n"

    @printf "\t(1) Slip Plot\n"
    @printf "\t(2) Slip Rate Animation\n"
    @printf "\t(3) Traction Animation\n"
    @printf "\t(4) Station Data\n"
    @printf "\t(5) Single Cycle\n\n"

    v = false
    flag = [0]
    while v == false
        @printf "Option: "
        try
            flag[1] = parse(Int64, chomp(readline()))
            if 0 < flag[1] < 6
                v = true
            end
        catch
            @printf "Not and Option\n"
        end

    end

    if flag[1] == 1
        
        slip_file = open("slip.dat", "r")
        slip_data = eachline(slip_file)

        nn = parse(Int64, split.(Iterators.take(slip_data, 1))[1][1])
        y = map(x -> parse(Float64, x), split.(Iterators.take(slip_data, 2))[1])[3:end]
        
        cycle_index = []
        count = 0

        
        for (i, line) in enumerate(slip_data)

            if line == "BREAK"
                
                count += 1
                if count%2 == 0
                    push!(cycle_index, i)
                end
            end

        end
        
        @printf "\n%d cycles in this data\n" length(cycle_index)
        @printf "Offset contours by # of cycles: "
        cycle_offset = parse(Int64, chomp(readline()))
        if cycle_offset == 0
            index_offset = 1
        else
            index_offset = cycle_index[cycle_offset] + 1
        end

        plt = plot(lengend=false, yflip = true, ylabel="Depth(Km)", xlabel="Slip(m)")

        cycle_count = 0
        for (index , line) in enumerate(Iterators.drop(slip_data, index_offset))
            @show index
            if line == "BREAK"
                plt = plot!(δ, y, linecolor=:black, linewidth = 1)
                cycle_count += 1
            else

                d = map(x->parse(Float64, x), split.(line))
                t = d[1]
                vmax = d[2]
                δ = d[3:end]

                if cycle_count % 2 == 0
                    plt = plot!(δ, y, linecolor=:blue, linewidth = .2)
                else
                    plt = plot!(δ, y, linecolor=:red, linewidth = .2)
                end 
            end

        end
        plot(plt)
        gui()

    elseif flag[1] == 2

    elseif flag[1] == 3

    elseif flag[1] == 4

    elseif flag[1] == 5

    end

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
