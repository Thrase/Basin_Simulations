using DelimitedFiles
using NCDatasets
using Interpolations

const fvars_name = ("δ", "V", "τ̂", "ψ")
const svars_name = ("δ", "V", "τ̂", "ψ")

"""
    init_fault_data(filename::String, var::String, nn::Integer)

creates a NetCDF file called `filename` in which fault time series of variable `var` is stored at along a total of `nn` nodes.

"""
function init_fault_data(filename::String, nn::Integer, depth::Array{Float64, 1})

    ds = NCDataset(filename, "c")
    
    defDim(ds, "time index", Inf)
    defDim(ds, "depth index", nn)

    defVar(ds, "time", Float64, ("time index",))
    defVar(ds, "depth", Float64, ("depth index",))
    defVar(ds, "maximum V", Float64, ("time index",))
    defVar(ds, "maximum v", Float64, ("time index",))

    for var in fvars_name
        defVar(ds, var, Float64, ("depth index", "time index"))
    end
    
    ds["depth"][:] .= depth

    close(ds)

end


"""
    init_station_data(filename::String, lendepths::Integer)

 creates a NetCDF file called `filename` in which station time series data is stored in `lendepths` total stations.

"""
function init_station_data(filename::String, stations::AbstractVector)

    ds = NCDataset(filename, "c")

    defDim(ds, "station index", length(stations))
    defDim(ds, "time index", Inf)
    
    defVar(ds, "time", Float64, ("time index",))
    defVar(ds, "maximum V", Float64, ("time index",))
    defVar(ds, "maxR", Float64, ("time index",))
    defVar(ds, "stations", Float64, ("station index",))
    defVar(ds, "δ", Float64, ("time index", "station index"))
    defVar(ds, "V", Float64, ("time index", "station index"))
    defVar(ds, "τ̂", Float64, ("time index", "station index"))
    defVar(ds, "ψ", Float64, ("time index", "station index"))
    

    ds["stations"][:] .= stations

    close(ds)

end


function init_volume_data(filename::String, x::Array{Float64, 1}, y::Array{Float64, 1})
    
    ds = NCDataset(filename, "c")

    defDim(ds, "time index", Inf)
    defDim(ds, "x index", length(x))
    defDim(ds, "y index", length(y))

    defVar(ds, "time", Float64, ("time index",))
    defVar(ds, "x", Float64, ("x index",))
    defVar(ds, "y", Float64, ("y index",))
    defVar(ds, "maximum V", Float64, ("time index",))
    defVar(ds, "u", Float64, ("x index", "y index", "time index"))
    defVar(ds, "v", Float64, ("x index", "y index", "time index"))
    defVar(ds, "σ", Float64, ("x index", "y index", "time index"))

    ds["x"][:] .= x
    ds["y"][:] .= y
    
    close(ds)

end


"""
    new_dir(new_dir::String, stations::AbstractVector, nn::Integer)

creates a new directory called `new_dir` to store data from `stations`, volume, and fault variables, for solutions with `nn` nodes per dimension.

"""
function new_dir(new_dir::String, input_file::String, stations::Array{Float64, 1}, depth::Array{Float64,1}, x::Array{Float64,1}, y::Array{Float64,1})

    if !isdir(new_dir)
        mkdir(new_dir)
    else
        error("new directory already exists.")
    end

    nn = length(depth)

    fault_name = string(new_dir, "fault.nc")
    stations_name = string(new_dir, "stations.nc")
    stations_r_name = string(new_dir, "remote_stations.nc")
    volume_name = string(new_dir, "volume.nc")
    
    init_fault_data(fault_name, nn, depth)
    init_station_data(stations_name, stations)
    init_station_data(stations_r_name, stations)

    init_volume_data(volume_name, x, y)

    cp(input_file, string(new_dir, "input_file.dat"))
    write_depth_grid(string(new_dir, "depth_grid.dat"), depth)

    return fault_name, stations_name, stations_r_name, volume_name

end

"""
    write_out_fault_data(filenames::Tuple, vars::Tuple, t::Float64)

writes out `vars` fault varibles at time `t` to NetCDF `filenames`.

"""
function write_out_fault_data(filename::String, vars::Tuple, maxv::Float64, t::Float64)

    file = NCDataset(filename, "a")
    max_V = maximum(vars[2])
    
    
    t_ind = size(file["time"])[1] + 1
    file["time"][t_ind] = t
    file["maximum V"][t_ind] = max_V
    file["maximum v"][t_ind] = maxv
    for i in 1:length(vars)
        file[fvars_name[i]][:, t_ind] .= vars[i]
    end
    
    close(file)

end

"""
    write_out_stations(station_file::String, stations::AbstractVector, depth:: Array{Float64,1}, vars::Tuple)

writes out interpolated station data `vars` at `stations` using grid spacing `depth` at time `t` to netCDF file `station_file`.

"""
function write_out_stations(station_file::String, stations::Array{Float64,1}, depth::Array{Float64,1}, vars::Tuple, maxR::Float64, t::Float64)

    file = NCDataset(station_file, "a")
    t_ind = size(file["time"])[1] + 1
    file["time"][t_ind] = t
    file["maximum V"][t_ind] = maximum(vars[2])
    file["maxR"][t_ind] = maxR
    #file["maximum v"][t_ind] = maximum(maxv)
    for (i, var) in enumerate(vars)
        interp = interpolate((depth,), var, Gridded(Linear()))
        var_stations = interp(stations)
        file[svars_name[i]][t_ind, :] .= var_stations
    end
    
    close(file)

end

function write_out_volume(volume_file::String, volume_vars::Tuple, V::Array{Float64,1}, nn::Integer, t::Float64)

    # this is probably pretty inefficient.....
    u = reshape(volume_vars[1], (nn,nn))
    v = reshape(volume_vars[2], (nn,nn))

    file = NCDataset(volume_file, "a")
    t_ind = size(file["time"])[1] + 1
    file["time"][t_ind] = t
    file["maximum V"][t_ind] = maximum(V)
    file["u"][:, :, t_ind] .= u[1:2:end, 1:2:end]
    file["v"][:, :, t_ind] .= v[1:2:end, 1:2:end]

end

function write_depth_grid(filename, fc)
    
    io = open(filename, "w")

    for i in 2:length(fc)
        write(io, string(fc[i], ", ", fc[i] - fc[i-1], "\n"))
    end
    
    close(io)


end
