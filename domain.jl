
⊗(A,B) = kron(A, B)

function transforms_e(Lw, r̂, l)
    

    A = (Lw - Lw*r̂ - Lw)/(2*tanh((r̂-1)/l) + tanh(-2/l)*(r̂ - 1))
    b = (A*tanh(-2/l) + Lw)/2
    c = Lw - b
    xt = (r,s) -> (A .* tanh.((r .- 1) ./ l) .+ b .* r .+ c,
                   ((A .* sech.((r .- 1) ./ l).^2) ./ l) .+ b,
                   zeros(size(s)))
    yt = (r,s) -> (A .* tanh.((s .- 1) ./ l) .+ b.*s .+ c,
                   zeros(size(r)),
                   ((A .* sech.((s .- 1) ./ l).^2) ./ l) .+ b)

    r = -1:.01:1
    s = -1:.01:1
    plot(r, xt(r,s))
    gui()
        
    return xt, yt
    
end

function transforms_ne(Lw, el_x, el_y)
    
    xt = (r,s) -> (el_x .* tan.(atan((Lw)/el_x).* (0.5*r .+ 0.5)),
                   el_x .* sec.(atan((Lw)/el_x).* (0.5*r .+ 0.5)).^2 * atan((Lw)/el_x) * 0.5 ,
                   zeros(size(s)))
    
    yt = (r,s) -> (el_y .* tan.(atan((Lw)/el_y).* (0.5*s .+ 0.5)) ,
                   zeros(size(r)),
                   el_y .* sec.(atan((Lw)/el_y).*(0.5*s .+ 0.5)) .^2 * atan((Lw)/el_y) * 0.5 )

    return xt, yt
end

function transforms_n(Lw)
    
    xt = (r,s) -> (Lw/2 .* (r .+ 1),
                   fill(Lw/2, size(r)),
                   zeros(size(s)))
    
    yt = (r,s) -> (Lw/2 .* (s .+ 1),
                   zeros(size(r)),
                   fill(Lw/2, size(s)))
    return xt, yt
end

function create_metrics(Nr, Ns, B_p, μ,
                        xf=(r,s)->(r, ones(size(r)), zeros(size(r))),
                        yf=(r,s)->(s, zeros(size(s)), ones(size(s))))


    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp * Nsp

    r = range(-1, stop=1, length=Nrp)
    s = range(-1, stop=1, length=Nsp)
        
    # Create the mesh
    r = ones(1, Nsp) ⊗ r
    s = s' ⊗ ones(Nrp)
    
    (x, xr, xs) = xf(r, s)
    (y, yr, ys) = yf(r, s)

    J = xr .* ys - xs .* yr
    
    @assert minimum(J) > 0
    
    JI = 1 ./ J
    
    rx =  ys ./ J
    sx = -yr ./ J
    ry = -xs ./ J
    sy =  xr ./ J

    #display(x)
    #display(y)
    #quit()
    μx = μy = μ(x, y, B_p)
    

    # variable coefficient matrix components
    crr = J .* (rx .* μx .* rx + ry .* μy .* ry)
    crs = J .* (sx .* μx .* rx + sy .* μy .* ry)
    css = J .* (sx .* μx .* sx + sy .* μy .* sy)
    
    # surface matrices
    (xf1, yf1) = (view(x, 1, :), view(y, 1, :))
    nx1 = -ys[1, :]
    ny1 =  xs[1, :]
    sJ1 = hypot.(nx1, ny1)
    nx1 = nx1 ./ sJ1
    ny1 = ny1 ./ sJ1

    (xf2, yf2) = (view(x, Nrp, :), view(y, Nrp, :))
    nx2 =  ys[end, :]
    ny2 = -xs[end, :]
    sJ2 = hypot.(nx2, ny2)
    nx2 = nx2 ./ sJ2
    ny2 = ny2 ./ sJ2

    (xf3, yf3) = (view(x, :, 1), view(y, :, 1))
    nx3 =  yr[:, 1]
    ny3 = -xr[:, 1]
    sJ3 = hypot.(nx3, ny3)
    nx3 = nx3 ./ sJ3
    ny3 = ny3 ./ sJ3

    (xf4, yf4) = (view(x, :, Nsp), view(y, :, Nsp))
    nx4 = -yr[:, end]
    ny4 =  xr[:, end]
    sJ4 = hypot.(nx4, ny4)
    nx4 = nx4 ./ sJ4
    ny4 = ny4 ./ sJ4

    (coord = (x,y),
     facecoord = ((xf1, xf2, xf3, xf4), (yf1, yf2, yf3, yf4)),
     crr = crr, css = css, crs = crs,
     J=J,
     JI = JI,
     sJ = (sJ1, sJ2, sJ3, sJ4),
     nx = (nx1, nx2, nx3, nx4),
     ny = (ny1, ny2, ny3, ny4),
     rx = rx, ry = ry, sx = sx, sy = sy)
end


function transfinite_blend(α1, α2, α3, α4, α1s, α2s, α3r, α4r, r, s)
  # +---4---+
  # |       |
  # 1       2
  # |       |
  # +---3---+
  @assert [α1(-1) α2(-1) α1( 1) α2( 1)] ≈ [α3(-1) α3( 1) α4(-1) α4( 1)]


  x = (1 .+ r) .* α2(s)/2 + (1 .- r) .* α1(s)/2 +
      (1 .+ s) .* α4(r)/2 + (1 .- s) .* α3(r)/2 -
     ((1 .+ r) .* (1 .+ s) .* α2( 1) +
      (1 .- r) .* (1 .+ s) .* α1( 1) +
      (1 .+ r) .* (1 .- s) .* α2(-1) +
      (1 .- r) .* (1 .- s) .* α1(-1)) / 4

  xr =  α2(s)/2 - α1(s)/2 +
        (1 .+ s) .* α4r(r)/2 + (1 .- s) .* α3r(r)/2 -
      (+(1 .+ s) .* α2( 1) +
       -(1 .+ s) .* α1( 1) +
       +(1 .- s) .* α2(-1) +
       -(1 .- s) .* α1(-1)) / 4


  xs = (1 .+ r) .* α2s(s)/2 + (1 .- r) .* α1s(s)/2 +
       α4(r)/2 - α3(r)/2 -
      (+(1 .+ r) .* α2( 1) +
       +(1 .- r) .* α1( 1) +
       -(1 .+ r) .* α2(-1) +
       -(1 .- r) .* α1(-1)) / 4

  return (x, xr, xs)
end


function transfinite(x1, x2, x3, x4,
                     y1, y2, y3, y4)

    
        ex = [(α) -> x1 * (1 .- α) / 2 + x3 * (1 .+ α) / 2,
          (α) -> x2 * (1 .- α) / 2 + x4 * (1 .+ α) / 2,
          (α) -> x1 * (1 .- α) / 2 + x2 * (1 .+ α) / 2,
          (α) -> x3 * (1 .- α) / 2 + x4 * (1 .+ α) / 2]
    exα = [(α) -> -x1 / 2 + x3 / 2,
           (α) -> -x2 / 2 + x4 / 2,
           (α) -> -x1 / 2 + x2 / 2,
           (α) -> -x3 / 2 + x4 / 2]
    ey = [(α) -> y1 * (1 .- α) / 2 + y3 * (1 .+ α) / 2,
          (α) -> y2 * (1 .- α) / 2 + y4 * (1 .+ α) / 2,
          (α) -> y1 * (1 .- α) / 2 + y2 * (1 .+ α) / 2,
          (α) -> y3 * (1 .- α) / 2 + y4 * (1 .+ α) / 2]
    eyα = [(α) -> -y1 / 2 + y3 / 2,
           (α) -> -y2 / 2 + y4 / 2,
           (α) -> -y1 / 2 + y2 / 2,
           (α) -> -y3 / 2 + y4 / 2]


    xt(x,y) = transfinite_blend(ex[1], ex[2], ex[3], ex[4],
                                exα[1], exα[2], exα[3], exα[4],
                                x, y)
    yt(x,y) = transfinite_blend(ey[1], ey[2], ey[3], ey[4],
                                eyα[1], eyα[2], eyα[3], eyα[4],
                                x, y)


    return xt, yt

end
