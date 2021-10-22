Using CUDA

CUDA.allowscalar(false)

a = [1 2 3 4 5 6]
db = CUDA.ones(1000)
b = ones(1000)
dc = similar(db)
c = similar(b)

for i in 1:length(a)
    @time dc .= a[i] .* db
    @time c .= a[i] .* b
end

nothing
