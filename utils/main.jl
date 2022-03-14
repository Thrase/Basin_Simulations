# Program for plotting Basin output, getting new initial conditions, etc...
include("funcs.jl")
using DelimitedFiles
using InteractiveViz
using IterTools
let 
    
    year_seconds = 31556952
    ### getting directory for simulation data
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
            
            ### options for plotting etc..
            @printf "\t(1) Slip Plot\n"
            @printf "\t(2) Slip Rate Animation\n"
            @printf "\t(3) Traction Animation\n"
            @printf "\t(4) Station Data\n"
            @printf "\t(5) Single Cycle\n"
            @printf "\t(6) Back\n"
            @printf "\t(7) Done\n\n"

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
            

            ### basic slip plot
            if flag[1] == 1
                
                slip_file = open("slip.dat", "r")
                slip_data = collect(eachline(slip_file))
                y, nn, slip_data = get_y_nn(slip_data)
                num_breaks, break_indices = get_break_indices(slip_data)
                
                total_cycles = floor(length(break_indices)/2)
                @printf "\n%d cycles in this data\n" total_cycles
                @printf "Offset contours by # of cycles: "
                cycle_offset = parse(Int64, chomp(readline()))
                @printf "Last cycle: "
                final_cycle = parse(Int64, chomp(readline()))
                
                final_index, index_offset = get_plot_indices(cycle_offset,
                                                             final_cycle,
                                                             break_indices)
                                                             
                slip_data = @view slip_data[index_offset:final_index, :]
                plt = plot_slip(slip_data, y, nn, "")
                display(plt)
                close(slip_file)
                
            elseif flag[1] == 2
                
            elseif flag[1] == 3
                
            ### station plots
            elseif flag[1] == 4
                
                @printf "Station depth: "
                depth = chomp(readline())
                station_data = @view readdlm(string("station_", depth))[2:end, :]
                t = @view station_data[ : , 1]
                V = @view station_data[ : , 3]
                iplot(t, V)

            ### single cycle options
            elseif flag[1] == 5
                
                slip_file = open("slip.dat", "r")
                slip_data = collect(eachline(slip_file))
                
                y_depth, nn, slip_data = get_y_nn(slip_data)
                
                break_number, break_indices = get_break_indices(slip_data)
                
                @printf "\n%d cycles in this data\n" floor(length(break_indices)/2)
                @printf "Which Cycle: "
                
                cycle_number = parse(Int64, chomp(readline()))
                                
                final_index, begin_index = get_plot_indices(cycle_number, cycle_number, break_indices)

                # get the single cycle slip data
                slip_cycle_data = @view slip_data[begin_index : final_index, :]

                done2 = false

                while !done2

                    ### get single cycle options
                    @printf "\n\t(1) Write-out cycle initial conditions\n"
                    @printf "\t(2) Plot volume variables\n"
                    @printf "\t(3) Write vtk files\n"
                    @printf "\t(4) Back\n"
                    
                    @printf "\nOption: "
                    option = parse(Int64, chomp(readline()))
                    
                    ### new initial conditions 
                    if option == 1
                        
                        # get a wider view to check initial conditions against
                        temp_plot = @view slip_data[break_indices[1] : final_index, :]
                        δ_off = get_line(temp_plot, 1)[3:end]
                        plt1 = plot_slip(temp_plot, y_depth, nn, "")

                        # get state initial condition
                        state_file = open("state.dat", "r")
                        state_data = collect(eachline(state_file))[2:end, :]
                        #state_index = begin_index - 2*(cycle_number - 1)
                        line = get_line(state_data, begin_index)
                        ψ = line[3:end]
                        t1 = line[1]

                        # get slip initial condition
                        line = get_line(slip_cycle_data, 1)
                        δ = line[3:end]
                        t2 = line[1]
                        
                        # plot initial condition to check that it is the right one
                        plt1 = plot!(δ-δ_off, y_depth, linecolor=:green, linewidth=3)
                        plt2 = plot(δ, y_depth, yflip = true, legend = false)
                        plt3 = plot(ψ, y_depth, yflip = true, legend = false)
                        display(plot(plt1,plt2,plt3, layout=(3,1)))
                        
                        # check that ψ and δ are coming from the same time
                        @assert t1 == t2
                        
                        close(state_file)
                        close(slip_file)

                        @printf "\nTime is: %f \n" t1/year_seconds
                        @printf "\nIs this the right initial condition (y/n): "
                        anw = chomp(readline())

                        ### if initial conditions look good write them out.
                        if anw == "y"
                            write_ψδ(ψ, δ, t1)
                        else
                            done2 = false
                        end
                        
                    ### make the 4 pannel volume plot
                    elseif option == 2
                        
                        @printf "Getting volume data...\n"
                        flush(stdout)
                        u_file, x, y, u_iter = parse_volume_file("us.dat", cycle_number)
                        v_file, _, _, v_iter = parse_volume_file("vs.dat", cycle_number)
                        

                        @printf "Getting stress data...\n"
                        flush(stdout)
                        # get inter and coseismic stress data
                        stress_file = open("stress.dat", "r")

                        # this indexing only works for new "break" writting
                        stress_data = @view collect(eachline(stress_file))[2:end, :]
                        inter_stress_data = @view stress_data[begin_index : break_indices[2*cycle_number] - 2, :]
                        co_stress_data = @view stress_data[break_indices[2*cycle_number] : final_index, :]
                        @printf "Getting slip data...\n"
                        flush(stdout)
                        # get inter and coeseismic slip data
                        inter_slip_data = @view slip_data[begin_index : break_indices[2*cycle_number] - 2, :]
                        co_slip_data = @view slip_data[break_indices[2*cycle_number] : final_index, :]
                        δ_off = get_line(inter_slip_data, 1)[3:end]

                        # again checking to make sure data is aligned
                        t1 = get_line(co_stress_data, size(co_stress_data)[1])[1]
                        t2 = get_line(co_slip_data, size(co_slip_data)[1])[1]
                        @assert t1 == t2

                        slip_plot = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Slip(m)")
                        stress_plot = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Stress(MPa)")
                        u_plot = heatmap(c=cgrad([:blue, :white,:red]), yflip=true)
                        v_plot = heatmap(clim = (0.0, .75), c=cgrad([:blue, :white,:red]), yflip=true)

                        @printf "Beginning plotting...\n"
                        for i in 1:size(inter_stress_data)[1]
                            
                            @printf "\r %f" i/size(inter_stress_data)[1]
                            δ = get_line(inter_slip_data, i)[3:end]
                            τ = get_line(inter_stress_data, i)[3:end]

                            slip_plot = plot!(slip_plot, 
                                              δ .- δ_off, 
                                              y_depth, 
                                              linecolor=:blue, 
                                              linewidth=.1)

                            stress_plot = plot(τ, 
                                               y_depth,
                                               legend=false,
                                               linecolor=:black,
                                               flip = true,
                                               ylabel="Depth(Km)",
                                               xlabel="Stress(MPa)")
                         
                            plot(slip_plot, v_plot, u_plot, stress_plot, layout=(2,2))
                            gui()

                        end

                        # loop over iterators
                        for (i, (u_line, v_line)) in enumerate(zip(u_iter, v_iter))
                            @printf "\r %f" i/size(co_stress_data)[1]
                            u = reshape(parse_line(u_line), length(x), length(y))'
                            v = reshape(parse_line(v_line), length(x), length(y))'

                            δ = get_line(co_slip_data, i)[3:end]
                            τ = get_line(co_stress_data, i)[3:end]

                            slip_plot = plot!(slip_plot, 
                                              δ .- δ_off, 
                                              y_depth, 
                                              linecolor=:red, 
                                              linewidth=.1)

                            stress_plot = plot(τ, 
                                               y_depth,
                                               legend=false,
                                               linecolor=:black,
                                               flip = true,
                                               ylabel="Depth(Km)",
                                               xlabel="Stress(MPa)")


                            v_plot = heatmap(x, y, v, clim = (0.0, .75),
                                    c=cgrad([:blue, :white,:red]), yflip=true)
                            u_plot = heatmap(x, y, u,
                                    c=cgrad([:blue, :white,:red]), yflip=true)

                            plot(slip_plot, v_plot, u_plot, stress_plot, layout=(2,2))
                            gui()
                            
                        end

                        close(u_file)
                        close(v_file)
                        close(slip_file)
                        close(stress_file)
                        

                    elseif option == 4
                        done2 = true
                    end
                end
            
            ### go back to look at different data
            elseif flag[1] == 6
                done1 = true
                cd("..")

            else
                exit()
            end
        end
    end
    cd("/home/tharvey2/Basin_Simulations/utils")
end

 

