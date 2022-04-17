using Printf
using DelimitedFiles
using Plots

const year_seconds = 31556952

### open filenames and retrun list of files, and list of arrays with strings of data
function open_files(filenames...; begin_index=2, final_index=nothing)
    files = []
    plot_data = []
    for filename in filenames
        file = open(filename, "r")
        push!(files, file)
        temp_data = collect(eachline(file))
        temp_data = temp_data[2:end, :]
        if final_index == nothing            
            final_index = size(temp_data)[1]
        end
        data = @view temp_data[begin_index:final_index, :]
        push!(plot_data, data)
    end
    return files, plot_data
end


### parse line of 2D array of strings data into 1D array of Float64s
get_line(data, i) = map(x -> parse(Float64, x), split.(data[i]))

### parse single string into 1D array of FLoat64s
parse_line(line) = map(x -> parse(Float64,x), split.(line))

### take off the first two lines of the slip_file and return them and the new array
function get_y_nn(slip_data)

    nn = parse(Int64, split.(Iterators.take(slip_data, 1))[1][1])
    y = map(x -> parse(Float64, x), split.(slip_data[2]))[3:end]
    slip_data = @view slip_data[3:end, :]

    return y, nn, slip_data

end

### get the number of breaks in a file, and the starting indices of each file
function get_break_indices(slip_data)
    
    cycle_index = [1]
    count = 0

    for (i, line) in enumerate(slip_data)
        if line == "BREAK"
            count += 1
            push!(cycle_index, i+1)
        end
    end
    

    return count, cycle_index

end

### make slip contour plot with given slip_data
function plot_slip(slip_data, y, nn, title)

    plt = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Slip(m)", title=title)

    δ = zeros(nn)
    line = zeros(nn+2)
    break_count = 0
    off_line = get_line(slip_data, 1)
    t = off_line[1]
    δ_off = off_line[3:end]
    @printf "begining time: %f\n" t/year_seconds
    final_index = size(slip_data)[1]
    for index in 1:final_index

        @printf "\r%f" index/final_index
        if slip_data[index] == "BREAK"
            plt = plot!(δ-δ_off, y, linecolor=:black, linewidth = 1)
            break_count += 1
        else
             if break_count % 2 == 0
                 line .= get_line(slip_data, index)
                 δ .= line[3:end]
                 t = line[1]
                 plt = plot!(δ-δ_off, y, linecolor=:blue, linewidth = .2)
             elseif break_count % 2 == 1 && index % 3 == 1 
                 δ .= get_line(slip_data, index)[3:end]
                 t = line[1]
                 plt = plot!(δ-δ_off, y, linecolor=:red, linewidth = .2)
            end 
        end
        
    end
    
    @printf "\nfinal time: %f\n" t/year_seconds

    return plt
end

### Create fault animations, with data gotten from open_files. Assumes all data is of the same length.
function fault_animations(plot_data, y)

    an_length = size(plot_data[1])[1]
    
    for i in 1:an_length
        plots = []
        for data in plot_data
            line = get_line(data, i)[3:end]
            plt = plot(line, y, yflip=true, legend=false)
            push!(plots, plt)
        end
        plot(plots..., layout = (length(plots), 1))
        gui()
    end
end

### convert begin_cycle, and final_cycle to there corrisponding indices in slip_data
function get_plot_indices(begin_cycle, final_cycle, break_indices)

    index_offset = break_indices[2*begin_cycle - 1]
    if 2*final_cycle + 1 <= length(break_indices)
        final_index = break_indices[2*final_cycle + 1] - 2
    elseif 2*final_cycle + 1 > length(break_indices)
        final_index = break_indices[2*final_cycle] - 2
    end

    return final_index, index_offset

end


### write out new initial conditions
function write_ψδ(ψ, δ, t)

    ψδ = vcat(ψ, δ)
    
    @printf "Filename: "
    name = chomp(readline())
    filename = string("/home/tharvey2/Basin_Simulations/input_files/initial_cons/", name)
    filename_t = string(filename, "_t")
    
    fout = open(filename, "w")
    writedlm(fout , ψδ)
    close(fout)
    
    fout = open(filename_t, "w")
    writedlm(fout, [t])
    close(fout)

    @printf "inital conditions write to %s\n" filename

end


### get coordinates, and data from volume files
function parse_volume_file(file_name, cycle_number, break_indices)
       
    file = open(file_name, "r")
    iter = eachline(file)
    
    # get volume coordinates
    x, iter = firstrest(iter)
    x = parse_line(x)
    y, iter = firstrest(iter)
    y = parse_line(y)

    iter = Iterators.drop(iter, 1)

    return file, x, y, iter
    
end
