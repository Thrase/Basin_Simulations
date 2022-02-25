using WriteVTK
using Printf

let

    dir = string("../../erickson/output_files/", ARGS[1])
    v_file = open(string(dir, "/vs.dat"))

    @printf "Opened u and v files...\n"

    vs = eachline(v_file)

    @printf "Read v data\n..."

    x = map(x->parse(Float64, x), split.(Iterators.take(vs, 1))[1])
    y = map(x->parse(Float64, x), split.(Iterators.take(vs, 2))[1])
    
    count = 1
    @printf "Converting to vtk...\n"
    for line in vs
        @printf "\r%d" count
        vtk = vtk_grid(string(dir, "/vtk_filesV/cycles_", count, ".vtr"), x, y)
        v = map(x-> parse(Float64, x), split.(line))
        vtk["v"] = v
        saved_files = vtk_save(vtk)
        count += 1
    end

    close(v_file)

    @printf ("Done.\n")
    
end
