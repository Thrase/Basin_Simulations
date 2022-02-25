using WriteVTK
using Printf

let

    dir = string("../../erickson/output_files/", ARGS[1])
    u_file = open(string(dir, "/us.dat"))

    @printf "Opened u files...\n"

    us = eachline(u_file)

    @printf "Read u and v data\n..."

    x = map(x->parse(Float64, x), split.(Iterators.take(us, 1))[1])
    y = map(x->parse(Float64, x), split.(Iterators.take(us, 2))[1])
    
    count = 1
    @printf "Converting to vtk...\n"
    for line in us
        @printf "\r%d" count
        vtk = vtk_grid(string(dir, "/vtk_filesU/cycles_", count, ".vtr"), x, y)
        u = map(x-> parse(Float64, x), split.(line))
        vtk["u"] = u
        saved_files = vtk_save(vtk)
        count += 1
    end
    
    close(u_file)

    @printf ("Done.\n")
    
end
