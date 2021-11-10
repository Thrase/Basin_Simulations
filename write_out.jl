using DelimitedFiles
using WriteVTK

function make_ss(fault_coord, δNp, input_file)

    num = 1
    dir_name = string("../../erickson/output_files/basin_data", num)
    while isdir(dir_name)
        num += 1
        dir_name = string("../../erickson/output_files/basin_data", num)
    end
    
    mkdir(dir_name)
    cp(input_file, string(dir_name, "/input_file.dat"))

    slip_file = string(dir_name,  "/slip.dat")
    to_write = copy(fault_coord)
    pushfirst!(to_write, 0.0, 0.0)
    io = open(slip_file, "w")
    writedlm(io, [δNp])
    writedlm(io, to_write', " ")
    close(io)

    slip_rate_file = string(dir_name,  "/slip_rate.dat")
    to_write = copy(fault_coord)
    pushfirst!(to_write, 0.0, 0.0)
    io = open(slip_rate_file, "w")
    writedlm(io, to_write', " ")
    close(io)

    stress_file = string(dir_name,  "/stress.dat")
    io = open(stress_file, "w")
    writedlm(io, to_write', " ")
    close(io)

    state_file = string(dir_name, "/state.dat")
    io = open(state_file, "w")
    writedlm(io, to_write', " ")
    close(io)
    

    return dir_name, slip_file, slip_rate_file, stress_file, state_file

end

function write_out_ss(δ, V, τ, ψ, t, slip_file, stress_file, slip_rate_file, state_file)
    
    max_V = log(10, maximum(V))
    to_write = copy(δ)
    pushfirst!(to_write, t, max_V)
    io = open(slip_file, "a")
    writedlm(io, to_write')
    close(io)

    to_write = copy(V)
    pushfirst!(to_write, t, max_V)
    io = open(slip_rate_file, "a")
    writedlm(io, to_write')
    close(io)
    
    to_write = copy(τ)
    pushfirst!(to_write, t, max_V)
    io = open(stress_file, "a")
    writedlm(io, to_write')
    close(io)
    
    to_write = copy(ψ)
    pushfirst!(to_write, t, max_V)
    io = open(state_file, "a")
    writedlm(io, to_write')
    close(io)
        
end


function make_stations(dir_name)

    stat_depth = collect(0.0:2.0:22.0)
    file_names = [string(dir_name, "/station_", stat_depth[i]) for i in 1:length(stat_depth)]
        
    header = "t slip slip_rate shear_stress state\n"

    for file in file_names
        io = open(file, "w")
        write(io, header)
        close(io)
    end
    
    return file_names

end


function write_out(δ, V, τ, θ, t, fault_coord, Lw, file_names, η=nothing)
    
    stat_depth = collect(0.0:2.0:22.0)
    @assert stat_depth[end] < Lw
    for i in 1:length(stat_depth)

        file_name = file_names[i]
        depth = stat_depth[i]
        d_ind = 0
        d_val = Lw
        for j in 1:length(fault_coord)
            if abs(depth-fault_coord[j]) < d_val
                d_ind = j
                d_val = abs(depth-fault_coord[j])
            end
        end
        
        @assert d_ind != 0

        x1 = fault_coord[d_ind]
        δ1 = δ[d_ind]
        V1 = V[d_ind]
        τ1 = τ[d_ind]
        θ1 = θ[d_ind]
        if η != nothing
            η1 = η[d_ind]
        end
        
        if fault_coord[d_ind] <= depth
            x2 = fault_coord[d_ind + 1]
            δ2 = δ[d_ind + 1]
            V2 = V[d_ind + 1]
            τ2 = τ[d_ind + 1]
            θ2 = θ[d_ind + 1]
            if η != nothing
                η2 = η[d_ind + 1]
            end
        end
        
        if fault_coord[d_ind] > depth
            x2 = fault_coord[d_ind - 1]
            δ2 = δ[d_ind - 1]
            V2 = V[d_ind - 1]
            τ2 = τ[d_ind - 1]
            θ2 = θ[d_ind - 1]
            if η != nothing
                η2 = η[d_ind - 1]
            end
        end

        δw = l_interp(depth, x1, x2, δ1, δ2)
        Vw = l_interp(depth, x1, x2, V1, V2)
        if η != nothing
            ηw = l_interp(depth, x1, x2, η1, η2)
            τw = l_interp(depth, x1, x2, τ1, τ2) - (Vw * ηw)
        else
            τw = l_interp(depth, x1, x2, τ1, τ2)
        end
        
        θw = l_interp(depth, x1, x2, θ1, θ2)
        
        if θw < 0
            @show θw, θ1, θ2, t, depth, d_val, fault_coord[d_ind], fault_coord[d_ind-1], fault_coord[d_ind+1]
        end

        dat = [t, δw, log(10,abs(Vw)), τw, log(10, θw)]
        io = open(file_name, "a")
        writedlm(io, dat', " ")
        close(io)
    end 
end


function write_out_vtk(u,v,x,y,step)
    
    vtkfile = vtk_grid(string("vtkfiles/basin_wave_", step, ".vtr"), x, y)
    vtkfile["u"] = u
    vtkfile["v"] = v
    outfiles = vtk_save(vtkfile)

end


function make_uv_files(dir_name, x, y)

    u_file = string(dir_name, "/us.dat")
    v_file = string(dir_name, "/vs.dat")
    i1 = open(v_file, "w")
    i2 = open(u_file, "w")
    writedlm(i1, x', " ")
    writedlm(i1, y', " ")
    writedlm(i2, x', " ")
    writedlm(i2, y', " ")
    close(i1)
    close(i2)

    return u_file,v_file
end

function write_out_uv(u, v, n, δNp, u_file, v_file)

    u = reshape(u, (n,n))
    v = reshape(v, (n,n))
    u = @view u[1:2:δNp, 1:2:δNp]
    v = @view v[1:2:δNp, 1:2:δNp]
    u = reshape(u, (length(u)))
    v = reshape(v, (length(v)))

    i1 = open(u_file, "a")
    i2 = open(v_file, "a")
    writedlm(i1, u', " ")
    writedlm(i2, v', " ")

    close(i1)
    close(i2)

end


function l_interp(x, x1, x2, y1, y2)

    return (y2 - y1)/(x2 - x1) * x - (y2 - y1)/(x2 - x1) * x1 + y1
    
end

