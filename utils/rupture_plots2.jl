using Plots
include("funcs.jl")


function rupture_plot(dir_name, begin_cycle, end_cycle)
    
    cd(string("../../erickson/output_files/", dir_name))
    @printf "found directory...\n"
    slip_file = collect(eachline(open("slip.dat":, "r")))
    y = get_line(slip_file, 2)[3:end]
    # slip data without first two lines
    slip_data = @view slip_file[3:end, :]
    
    get_cycle_indices(slip_data)
    # get indices in file where breaks are
    break_number, break_indices = get_break_indices(slip_data)
    # convert first cycle number to break number
    begin_break = 2*begin_cycle - 1
    # convert end cycle to break number 
    # 2 * end_cycle + 1 is the interseismic phase of end_cycle + 1 
    end_break = 2*end_cycle + 1
 
    # index of begin_cycle interseismic
    b_inter_index = break_indices[begin_break]
    # index of begin_cycle coseismic 
    b_co_index = break_indices[begin_break + 1]
    # get last index of end_cycle coseismic
    end_index = break_indices[end_break] - 2 
    
    # truncate slip_data around those indices
    slip_data = @view slip_data[b_inter_index:end_index, :]

end
