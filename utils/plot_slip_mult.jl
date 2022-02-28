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

    total_cycles = []
    cycle_indices = []
    for dir_name in dir_names
        slip_file = open(string(dir_name, "/slip.dat"))
        slip_data = eachline(slip_file)
        cycle_count, cycle_index = get_cycle_indices(slip_data)
        push!(total_cycles, cycle_count)
        push!(cycle_indices, cycle_index)
        close(slip_file)
    end

    @printf "The minimum number of cycles over the simulations is %d\n" floor(minimum(total_cycles)/2)

    @printf "Offset contours by # of cycles (or -1 is no offset): "
    cycle_offset = parse(Int64, chomp(readline()))
    @printf "Last cycle: "
    final_cycle = parse(Int64, chomp(readline()))
    
    plts = []
    count = 1
    for (i, dir_name) in enumerate(dir_names)
        
        cycle_index = cycle_indices[i]
        total_cycle = floor(total_cycles[i])/2
        final_index, index_offset = get_plot_indices(cycle_offset,
                                                     final_cycle,
                                                     cycle_index,
                                                     total_cycle)

        slip_file = open(string( dir_name, "/slip.dat"))
        slip_data = collect(eachline(slip_file))
              
        
        push!(plts, plot_slip(slip_data, 
                              final_index,
                              titles[i],
                              index_offset=index_offset))
        
        close(slip_file)
        
        @printf "\rGot plot %d" count
        count += 1
    end

    plot(plts..., layout=(length(plts),1))
    gui()

end
