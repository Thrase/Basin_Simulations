# Program for plotting Basin output, and getting new initial conditions.
include("funcs.jl")

let 
    

    done = false
    cd(string("../../erickson/output_files/"))
    while !done 
        v = false
        while v == false
            @printf "\nOutput directory: "
            folder_name = chomp(readline())
            try
                cd(folder_name)
                v = true
            catch
                @printf "Directory not found\n"
            end
        end
        @printf "Found directory...\n\n"

        done1 = false
        while !done1
            
            
            @printf "\t(1) Slip Plot\n"
            @printf "\t(2) Slip Rate Animation\n"
            @printf "\t(3) Traction Animation\n"
            @printf "\t(4) Station Data\n"
            @printf "\t(5) Single Cycle\n"
            @printf "\t(6) Done\n\n"

            v = false
            flag = [0]
            while v == false
                @printf "Option: "
                try
                    flag[1] = parse(Int64, chomp(readline()))
                    if 0 < flag[1] < 7
                        v = true
                    end
                catch
                    @printf "Not and Option\n"
                end
                
            end
            
            if flag[1] == 1
                
                slip_file = open("slip.dat", "r")
                slip_data = collect(eachline(slip_file))
                
                cycle_index = get_cycle_indices(slip_data)[2]
                total_cycles = floor(length(cycle_index)/2)

                @printf "\n%d cycles in this data\n" total_cycles
                @printf "Offset contours by # of cycles (or -1 is no offset): "
                cycle_offset = parse(Int64, chomp(readline()))

                @printf "Last cycle: "
                final_cycle = parse(Int64, chomp(readline()))
                
                final_index, index_offset = get_plot_indices(cycle_offset,
                                                             final_cycle,
                                                             cycle_index)
                
                plot_slip(slip_data, final_index, index_offset=index_offset)
                
                close(slip_file)

            elseif flag[1] == 2
                
            elseif flag[1] == 3
                
            elseif flag[1] == 4
                
            elseif flag[1] == 5
                
                slip_file = open("slip.dat", "r")
                slip_data = collect(eachline(slip_file))
                
                nn = parse(Int64, split.(Iterators.take(slip_data, 1))[1][1])
                y = map(x -> parse(Float64, x), split.(slip_data[2]))[3:end]
                
                break_number, cycle_indices = get_cycle_indices(slip_data)
                
                @printf "\n%d cycles in this data\n" length(cycle_indices)
                @printf "Which Cycle: "
                
                cycle_number = parse(Int64, chomp(readline()))
                @show cycle_indices
                cycle_index = cycle_indices[cycle_number]
                done2 = false

                while !done2
                    @printf "\n\t(1) Write-out cycle initial conditions\n"
                    @printf "\t(2) Plot volume variables\n"
                    @printf "\t(3) Write vtk files\n"
                    @printf "\t(4) Back\n"
                    
                    @printf "\nOption: "
                    option = parse(Int64, chomp(readline()))
                    
                    if option == 1
                        
                        state_file = open("state.dat", "r")
                        state_data = collect(eachline(state_file))
                        state_index = cycle_index - break_number - 1
                        
                        ψ = get_line(state_data, state_index)[3:end]
                        line = get_line(slip_data, cycle_index)
                        δ = line[3:end]
                        t = line[1]

                        close(state_file)
                        close(slip_file)
                        
                        ψδ = vcat(ψ, δ)
                        
                        @printf "Filename: "
                        name = chomp(readline())
                        filename = string("/home/tharvey2/Basin_Simulations/input_files/", name)
                        filename_t = string(filename, "_t")

                        fout = open(filename, "w")
                        writedlm(fout , ψδ)
                        close(fout)
                        
                        fout = open(filename_t, "w")
                        writedlm(fout, [t])
                        close(fout)

                        @printf "inital conditions write to %s\n" filename

                    elseif option == 2

                        t_begin = parse(Float64, slip_data[cycle_index][1])
                        t_end = parse(Float64, slip_data[cycle_index - 2][1])

                        @printf "(1) displacements\n"
                        @printf "(2) velocity\n"

                        option = chomp(readline())

                        if option == 1
                            u_file = open("us.dat", "r")
                            u_data = collect(eachline(u_file))
                        elseif option == 2

                        end
                        

                    elseif option == 4
                        done2 = true

                    end
                end
                
            elseif flag[1] == 6
                done1 = true
                cd("..")
            end
        end
    end
    cd("/home/tharvey2/Basin_Simulations/utils")
end

 

