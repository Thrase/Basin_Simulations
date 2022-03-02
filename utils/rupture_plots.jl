using Images
using Plots
include("funcs.jl")


let

    cd("../../erickson/output_files/d4_p4_spinup")
    @printf "found directory...\n"
    slip_file = collect(eachline(open("slip.dat", "r")))
    stress_data = collect(eachline(open("stress.dat","r")))

    nn = Int64(get_line(slip_file,1)[1])
    y = get_line(slip_file, 2)[3:end]
    slip_data = @view slip_file[3:end , :]
    
    img1 = load(string("V_png/velocities.", lpad(0,4,"0"), ".png"))
    plt1 = plot(img1, yaxis=false, xaxis=false, ticks=false)
    img2 = load(string("U_png/displacements.", lpad(0,4,"0"), ".png"))
    plt2 = plot(img2,  yaxis=false, xaxis=false, ticks=false)
    plt3 = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Slip(m)")
    plt4 = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Stress(MPa)")

    @printf "intialized plotting...\n"
    l = @layout [
        a{.7w} b{.3w}
        c{.7w} d{.3w}
    ]

    offset = zeros(nn)
    inter_flag = 1
    break_count = 0
    inter_count = 0
    cycle_count = 1
    final_index = size(slip_data)[1]
    anim = @animate for i = 1:final_index
        @printf "\r%f" i/final_index
        stress_index = i - break_count
        
        if inter_flag == 1
            if slip_data[i] == "BREAK"
                inter_flag = 0
                break_count += 1 
            else
                
                slip = get_line(slip_data, i)[3:end]
                stress = get_line(stress_data, stress_index)[3:end]
                plt3 = plot!(plt3, slip .- offset, y, linecolor=:blue, linewidth=.1, title=string("Cycle # ", cycle_count))
                plt4 = plot(stress, y,
                            legend=false,
                            linecolor=:black,
                            flip = true,
                            ylabel="Depth(Km)",
                            xlabel="Stress(MPa)")
                plot(plt1, plt3, plt2, plt4, layout=l)
                #gui()
                inter_count += 1
            end
        else
            
            if slip_data[i] == "BREAK"
                inter_flag = 1
                break_count += 1
                cycle_count += 1
                if break_count == 12
                    offset .= get_line(slip_data, i+1)[3:end]
                    plt3 = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Slip(m)")
                end
            else
                
                slip = get_line(slip_data, i)
                V_max = slip[2]

                if V_max > log(.1)

                    particle_index = i - break_count - inter_count
                    img1 = load(string("V_png/velocities.", lpad(particle_index,4,"0"), ".png"))
                    plt1 = plot(img1, yaxis=false, xaxis=false, ticks=false)
                    img2 = load(string("U_png/displacements.", lpad(particle_index,4,"0"), ".png"))
                    plt2 = plot(img2, yaxis=false, xaxis=false, ticks=false)
                    slip = slip[3:end]
                    stress = get_line(stress_data, stress_index)[3:end]
                    plt3 = plot!(plt3, slip .- offset, y, linecolor=:red, linewidth=.1)
                    plt4 = plot(stress, y,
                                legend=false,
                                linecolor=:black,
                                flip = true,
                                ylabel="Depth(Km)",
                                xlabel="Stress(MPa)")
                    
                    plot(plt1, plt3, plt2, plt4, layout=l)
                    #gui()
                end
                
            end
        end
    end
    
    gif(anim, "ruptures.gif", fps=16)
    cd("/home/tharvey2/Basin_Simulations/utils")
end
