include("funcs.jl")
using Printf

let

    cd("../../erickson/output_files/")
    done = false
    dir_names = []
    titles = []
    dir_count = 1
    while !done
        @printf "Directory %d (-1 is done): " dir_count
        dir = chomp(readline())
     
        if dir == "-1"
            done = true
        else
            @printf "Title: "
            push!(titles, chomp(readline()))
            push!(dir_names, dir)
            dir_count += 1
        end
    end

    nb_arr= []
    bi_arr = []
    for dir_name in dir_names
        slip_file = open(string(dir_name, "/slip.dat"))
        slip_data = eachline(slip_file)
        num_breaks, break_indices = get_break_indices(slip_data)
        push!(nb_arr, num_breaks)
        push!(bi_arr, break_indices)
        close(slip_file)
    end

    @printf "The minimum number of cycles over the simulations is %d\n" floor(minimum(nb_arr)/2)

    @printf "Offset contours by # of cycles (or -1 is no offset): "
    cycle_offset = parse(Int64, chomp(readline()))
    @printf "Last cycle: "
    final_cycle = parse(Int64, chomp(readline()))
    
    plts = []
    count = 1
    for (i, dir_name) in enumerate(dir_names)
        
        break_indices = bi_arr[i]
        total_cycles = floor(nb_arr[i])/2
        final_index, index_offset = get_plot_indices(cycle_offset,
                                                     final_cycle,
                                                     break_indices)

                                                     
        
        slip_file = open(string(dir_name, "/slip.dat"))
        slip_data = collect(eachline(slip_file))
        
        y, nn, slip_data = get_y_nn(slip_data)
        
        slip_data = @view slip_data[index_offset:final_index,:]

        push!(plts, plot_slip(slip_data, y, nn, titles[i]))

        close(slip_file)
        
        @printf "\rGot plot %d" count
        count += 1
    end

    plot(plts..., layout=(length(plts),1))
    gui()

end
