module Transmission3D
    using HMatrices # https://github.com/WaveProp/HMatrices.jl?tab=readme-ov-file https://waveprop.github.io/HMatrices.jl/dev/
    using StaticArrays
    using LinearAlgebra
    using IterativeSolvers
    # XXX DEBUG
    using Plots
    using Plots.PlotMeasures

    # Define static vector with fixed size globally
    PointdD = SVector{3,Float64}
    
    # Since this is a block-wise matrix, follow the steps in https://waveprop.github.io/HMatrices.jl/dev/
    struct GreensMatrix <: AbstractMatrix{ComplexF64}
        X::Vector{PointdD}
        Y::Vector{PointdD}
        k0::Float64
        alpha::ComplexF64
        radius::Float64
        regularize::Bool
        G0_center_value::ComplexF64
        solve::Bool
    end
    
    function M(x, y, row, col, k0, alpha, radius, regularize, G0_center_value, solve)::ComplexF64
        if solve
            if x == y && row == col
                # Diagonal has an identity: this is not dangerous since it's only STRICTLY at the same point
                extra = 1
            else
                extra = 0
            end
            extra - k0*k0*alpha*G(x,y,row,col,k0,radius,regularize,G0_center_value)
        else 
            k0*k0*alpha*G(x,y,row,col,k0,radius,regularize,G0_center_value)
        end
    end
    
    function G(x, y, row, col, k0, radius, regularize, G0_center_value)::ComplexF64
        Rvec = y - x
        d = norm(Rvec)
        Rvec /= d
        RxR = Rvec[row]*Rvec[col]
        R = k0 * d
        
        if regularize
            # Make the value equal to the center value in the whole disk scatterer
            threshold = radius
        else
            # Just strictly the center
            threshold = 0.0
        end
        
        if d <= threshold
            if row == col
                G0_center_value
            else
                0.0
            end
        else
            # propagator
            if row == col
                I = 1
            else
                I = 0
            end
            
            ( I-RxR - (I-3*RxR) * (1.0/(1.0im*R)+(R)^(-2))) * exp(1.0im*R)/(4*pi*d)
            
        end
    end
    
    function block_id(i)::Int64
        # Return index of point to use from index in G matrix
        floor(Int64, (i-1)/3) + 1
    end
    
    Base.getindex(K::GreensMatrix,i::Int,j::Int) = M(K.X[block_id(i)], K.Y[block_id(j)], 1+(i-1)%3, 1+(j-1)%3, K.k0, K.alpha, K.radius, K.regularize, K.G0_center_value, K.solve)
    Base.size(K::GreensMatrix) = 3*length(K.X), 3*length(K.Y)
    
    
    # Since this is a block-wise matrix, follow the steps in https://waveprop.github.io/HMatrices.jl/dev/
    struct GreensMatrixScalar <: AbstractMatrix{ComplexF64}
        X::Vector{PointdD}
        Y::Vector{PointdD}
        k0::Float64
        alpha::ComplexF64
        radius::Float64
        regularize::Bool
        G0_center_value::ComplexF64
        solve::Bool
    end
    
    function M_scalar(x, y, k0, alpha, radius, regularize, G0_center_value, solve)::ComplexF64
        if solve
            if x == y
                # Diagonal has an identity: this is not dangerous since it's only STRICTLY at the same point
                extra = 1
            else
                extra = 0
            end
            extra - k0*k0*alpha*G_scalar(x,y,k0,radius,regularize,G0_center_value)
        else 
            k0*k0*alpha*G_scalar(x,y,k0,radius,regularize,G0_center_value)
        end
    end
    
    function G_scalar(x, y, k0, radius, regularize, G0_center_value)::ComplexF64
        Rvec = y - x
        d = norm(Rvec)
        R = k0 * d
        
        if regularize
            # Make the value equal to the center value in the whole disk scatterer
            threshold = radius
        else
            # Just strictly the center
            threshold = 0.0
        end
        
        if d <= threshold
            
            G0_center_value
            
        else
            
            exp(1.0im*R)/(4*pi*d)
            
        end
    end
    
    Base.getindex(K::GreensMatrixScalar,i::Int,j::Int) = M_scalar(K.X[i], K.Y[j], K.k0, K.alpha, K.radius, K.regularize, K.G0_center_value, K.solve)
    Base.size(K::GreensMatrixScalar) = length(K.X), length(K.Y)
    
    function solve(python_points::AbstractArray, points_Einc::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape = size(python_points)
        dim = shape[2]
        
        if dim != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n = shape[1]
        PointdD = SVector{dim,Float64}
        points = [PointdD(python_points[k,:]) for k in 1:n]
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * ((2.0/3.0)*exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensMatrix(points, points, k0, alpha, radius, regularize, G0_center_value, true)
        
        # Need pointsclt with right size!
        pointsproxy = [points[1+floor(Int64,k/3)] for k in 0:dim*n-1]
        pointsclt = ClusterTree(pointsproxy)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol, rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt, pointsclt; adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (solve): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, margin = 0px) #left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot.svg")
        end
        
        # Maybe the loop can be bypassed, for now doing it brute-force
        field_shape = size(points_Einc)
        n_angles = field_shape[2]
        
        # Initialize output structure
        points_Etot = similar(points_Einc)
        
        # Initialize static structure for rhs
        PointND = SVector{dim * n, ComplexF64}
        
        if use_lu
            # Use an LU decomposition to make solving faster in the angle loop
            # XXX MAY NEED EXPLICIT PIVOT OPTION?
            lu_decomp = lu!(H; atol = atol, rtol = rtol)
            for angle_index in 1:n_angles
                inc_field_vec = PointND(points_Einc[:,angle_index])
                approx_fields = lu_decomp \ inc_field_vec
                points_Etot[:,angle_index] = approx_fields[:]
            end
        else
            # Just use GMRES at every angle.
            # This is in principle slower, as solves are independent!
            # Other idea: use LU solution as preconditioner?
            for angle_index in 1:n_angles
                inc_field_vec = PointND(points_Einc[:,angle_index])
                precondit = similar(inc_field_vec) # Probably not best https://iterativesolvers.julialinearalgebra.org/stable/preconditioning/#Preconditioning
                approx_fields = gmres!(precondit, H, inc_field_vec) # Maybe not best? https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/gmres/#GMRES
                points_Etot[:,angle_index] = approx_fields[:]
            end
        end
        
        return points_Etot
        
    end

    function propagate(python_points_scat::AbstractArray, python_points_meas::AbstractArray, points_Escat::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
            
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape_scat = size(python_points_scat)
        dim = shape_scat[2]
        
        shape_meas = size(python_points_meas)
        dim_meas = shape_meas[2]
        
        if dim != 3 || dim_meas != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n_scat = shape_scat[1]
        n_meas = shape_meas[1]
        points_scat = [PointdD(python_points_scat[k,:]) for k in 1:n_scat]
        points_meas = [PointdD(python_points_meas[k,:]) for k in 1:n_meas]
        
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * ((2.0/3.0)*exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensMatrix(points_meas,points_scat,k0,alpha,radius,regularize,G0_center_value, false)
        
        # Need pointsclt with right size!
        pointsproxy_scat = [points_scat[1+floor(Int64,k/3)] for k in 0:dim*n_scat-1]
        pointsclt_scat = ClusterTree(pointsproxy_scat)
        pointsproxy_meas = [points_meas[1+floor(Int64,k/3)] for k in 0:dim*n_meas-1]
        pointsclt_meas = ClusterTree(pointsproxy_meas)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol, rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt_meas, pointsclt_scat; adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (propagate): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_calc.svg")
        end
        
        # Maybe the loop can be bypassed, for now doing it brute-force
        field_shape = size(points_Escat)
        n_angles = field_shape[2]
        
        # Initialize output structure
        points_Emeas = similar(points_Escat, dim * n_meas, n_angles)
        
        # Compute H times fields for every angle
        mul!(points_Emeas, H, points_Escat, 1, 0; threads=threads)
        
        return points_Emeas
        
    end
    
    function mean_dos(python_points_scat::AbstractArray, python_points_meas::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, discard_absorption = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape_scat = size(python_points_scat)
        dim = shape_scat[2]
        
        shape_meas = size(python_points_meas)
        dim_meas = shape_meas[2]
        
        if dim != 3 || dim_meas != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n_scat = shape_scat[1]
        n_meas = shape_meas[1]
        points_scat = [PointdD(python_points_scat[k,:]) for k in 1:n_scat]
        points_meas = [PointdD(python_points_meas[k,:]) for k in 1:n_meas]
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * ((2.0/3.0)*exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        if discard_absorption
            alpha_ = real(alpha)
        else
            alpha_ = alpha
        end
        
        # K is an abstract representation of the kernel. Here, need it for both M and G
        K = GreensMatrix(points_scat, points_scat, k0, alpha_, radius, regularize, G0_center_value, true)
        K_prop = GreensMatrix(points_meas, points_scat, k0, alpha_, radius, regularize, G0_center_value, false)
        K_prop_T = GreensMatrix(points_scat, points_meas, k0, alpha_, radius, regularize, G0_center_value, false)
        
        # Need pointsclt with right size!
        # Need pointsclt with right size!
        pointsproxy_scat = [points_scat[1+floor(Int64,k/3)] for k in 0:dim*n_scat-1]
        pointsclt_scat = ClusterTree(pointsproxy_scat)
        pointsproxy_meas = [points_meas[1+floor(Int64,k/3)] for k in 0:dim*n_meas-1]
        pointsclt_meas = ClusterTree(pointsproxy_meas)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol,rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt_scat, pointsclt_scat; adm,comp, threads=threads, distributed=false)
        H_prop = assemble_hmatrix(K_prop, pointsclt_meas, pointsclt_scat; adm,comp, threads=threads, distributed=false)
        H_prop_T = assemble_hmatrix(K_prop_T, pointsclt_scat, pointsclt_meas; adm,comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (propagate): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_calc.svg")
        end
        
        # Throughout the calculation, GT.G is needed 
        GTG = assemble_hmatrix(K, pointsclt_scat, pointsclt_scat; adm,comp, threads=threads, distributed=false) # Initialization... similar does not work here
        HMatrices.hmul!(GTG, H_prop_T, H_prop, 1, 0, comp) # Compute GTG <- GT.G * 1 - 0 * GT
        
        # Need to perform inverse on M: do it using the LU decomposition
        # Other solutions like below instantiate W in memory
        # W = lu_decomp\Matrix(1.0I,n_scat,n_scat)
        # W = ldiv!(lu_decomp,Matrix((1.0+0im)*I,n_scat,n_scat))
        lu_decomp = lu!(H; atol = atol, rtol = rtol)
        
        sum = 0.0 + 0im
        
        # Instead of trying to store the (large) inverse, treat columns separately!
        for k in 1:dim*n_scat
            
            # Compute a COLUMN of the inverse matrix
            unit_vector = zeros(dim*n_scat)
            unit_vector[k] = 1.0
            inv_column = lu_decomp \ unit_vector
            
            # Deduce a LINE of the matrix GT.G.W
            matrix_line = similar(inv_column)
            mul!(matrix_line,GTG,inv_column,1,0; threads=threads)
            
            # The element that matters for the trace is the k-th element
            sum += matrix_line[k]
        end
        
        sum /= n_meas
        # The G has a k0**2 alpha prefactor in this code
        # rho should be 2 pi k0 alpha * trace
        # it is currently k0^4 * alpha^2
        sum *= 2.0 * pi / (k0^3 * alpha_)
        
        # rho is the imaginary part of the Green function of the medium
        mean_dos = imag(sum)
        
        return mean_dos
        
    end
    
    function ldos(python_points_scat::AbstractArray, python_points_meas::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, discard_absorption = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape_scat = size(python_points_scat)
        dim = shape_scat[2]
        
        shape_meas = size(python_points_meas)
        dim_meas = shape_meas[2]
        
        if dim != 3 || dim_meas != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n_scat = shape_scat[1]
        n_meas = shape_meas[1]
        points_scat = [PointdD(python_points_scat[k,:]) for k in 1:n_scat]
        points_meas = [PointdD(python_points_meas[k,:]) for k in 1:n_meas]
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * ((2.0/3.0)*exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        if discard_absorption
            alpha_ = real(alpha)
        else
            alpha_ = alpha
        end
        
        # K is an abstract representation of the kernel. Here, need it for both M and G
        K = GreensMatrix(points_scat, points_scat, k0, alpha_, radius, regularize, G0_center_value, true)
        K_prop = GreensMatrix(points_meas, points_scat, k0, alpha_, radius, regularize, G0_center_value, false)
        
        # Need pointsclt with right size!
        # Need pointsclt with right size!
        pointsproxy_scat = [points_scat[1+floor(Int64,k/3)] for k in 0:dim*n_scat-1]
        pointsclt_scat = ClusterTree(pointsproxy_scat)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol,rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt_scat, pointsclt_scat; adm,comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (propagate): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_calc.svg")
        end
        
        # Need to perform inverse on M: do it using the LU decomposition
        # Other solutions like below instantiate W in memory
        # W = lu_decomp\Matrix(1.0I,n_scat,n_scat)
        # W = ldiv!(lu_decomp,Matrix((1.0+0im)*I,n_scat,n_scat))
        lu_decomp = lu!(H; atol = atol, rtol = rtol)
        
        ldos_list = zeros(ComplexF64,n_meas)
        col = zeros(ComplexF64, dim*n_scat)
        
        # Instead of trying to store the (large) inverse, treat columns separately!
        # XXX This is just because this library does not have a good hierarchical inverse!
        for k in 1:dim*n_scat
            
            # Compute a COLUMN of the inverse matrix
            unit_vector = zeros(dim*n_scat)
            unit_vector[k] = 1.0
            inv_column = lu_decomp \ unit_vector
        
            # Only the trace is needed for the LDOS: accumulate the sum that defines trace elements
            for row in 1:dim*n_meas
                for j in 1:dim*n_scat
                    col[j] = getindex(K_prop,row,j)
                end
                output_row = 1 + floor(Int64,(row - 1)/dim)
                ldos_list[output_row] += col[k] * sum(vx*vy for (vx,vy) in zip(col, inv_column)) # NOT dot(col,inv_column) # dot conjugates!
            end
        end
        
        # The G has a k0**2 alpha prefactor in this code
        # rho should be 2 pi k0 alpha * trace
        # it is currently k0^4 * alpha^2
        ldos_list .*= 2.0 * pi / (k0^3 * alpha_)
        
        # rho is the imaginary part of the Green function of the medium
        ldos_list = imag.(ldos_list)
        
        return ldos_list
        
    end
    
    function spectrum(python_points::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape = size(python_points)
        dim = shape[2]
        
        if dim != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n = shape[1]
        PointdD = SVector{dim,Float64}
        points = [PointdD(python_points[k,:]) for k in 1:n]
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * ((2.0/3.0)*exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensMatrix(points, points, k0, alpha, radius, regularize, G0_center_value, true)
        
        # Need pointsclt with right size!
        pointsproxy = [points[1+floor(Int64,k/3)] for k in 0:dim*n-1]
        pointsclt = ClusterTree(pointsproxy)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol, rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt, pointsclt; adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (solve): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, margin = 0px) #left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot.svg")
        end
        
        lambdas = eigvals(H) # XXX Not implemented
        
        return lambdas
        
    end

    function solve_scalar(python_points::AbstractArray, points_Einc::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape = size(python_points)
        dim = shape[2]
        
        if dim != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n = shape[1]
        PointdD = SVector{dim,Float64}
        points = [PointdD(python_points[k,:]) for k in 1:n]
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * (exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensMatrixScalar(points, points, k0, alpha, radius, regularize, G0_center_value, true)
        
        # Need pointsclt with right size!
        pointsclt = ClusterTree(points)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol, rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt, pointsclt; adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (solve): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, margin = 0px) #left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot.svg")
        end
        
        # Maybe the loop can be bypassed, for now doing it brute-force
        field_shape = size(points_Einc)
        n_angles = field_shape[2]
        
        # Initialize output structure
        points_Etot = similar(points_Einc)
        
        # Initialize static structure for rhs
        PointND = SVector{n, ComplexF64}
        
        if use_lu
            # Use an LU decomposition to make solving faster in the angle loop
            # XXX MAY NEED EXPLICIT PIVOT OPTION?
            lu_decomp = lu!(H; atol = atol, rtol = rtol)
            for angle_index in 1:n_angles
                inc_field_vec = PointND(points_Einc[:,angle_index])
                approx_fields = lu_decomp \ inc_field_vec
                points_Etot[:,angle_index] = approx_fields[:]
            end
        else
            # Just use GMRES at every angle.
            # This is in principle slower, as solves are independent!
            # Other idea: use LU solution as preconditioner?
            for angle_index in 1:n_angles
                inc_field_vec = PointND(points_Einc[:,angle_index])
                precondit = similar(inc_field_vec) # Probably not best https://iterativesolvers.julialinearalgebra.org/stable/preconditioning/#Preconditioning
                approx_fields = gmres!(precondit, H, inc_field_vec) # Maybe not best? https://iterativesolvers.julialinearalgebra.org/stable/linear_systems/gmres/#GMRES
                points_Etot[:,angle_index] = approx_fields[:]
            end
        end
        
        return points_Etot
        
    end

    function propagate_scalar(python_points_scat::AbstractArray, python_points_meas::AbstractArray, points_Escat::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
            
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape_scat = size(python_points_scat)
        dim = shape_scat[2]
        
        shape_meas = size(python_points_meas)
        dim_meas = shape_meas[2]
        
        if dim != 3 || dim_meas != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n_scat = shape_scat[1]
        n_meas = shape_meas[1]
        points_scat = [PointdD(python_points_scat[k,:]) for k in 1:n_scat]
        points_meas = [PointdD(python_points_meas[k,:]) for k in 1:n_meas]
        
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * (exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensMatrixScalar(points_meas,points_scat,k0,alpha,radius,regularize,G0_center_value, false)
        
        # Need pointsclt with right size!
        pointsclt_scat = ClusterTree(points_scat)
        pointsclt_meas = ClusterTree(points_meas)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol, rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt_meas, pointsclt_scat; adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (propagate): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_calc.svg")
        end
        
        # Maybe the loop can be bypassed, for now doing it brute-force
        field_shape = size(points_Escat)
        n_angles = field_shape[2]
        
        # Initialize output structure
        points_Emeas = similar(points_Escat, n_meas, n_angles)
        
        # Compute H times fields for every angle
        mul!(points_Emeas, H, points_Escat, 1, 0; threads=threads)
        
        return points_Emeas
        
    end
    
    function mean_dos_scalar(python_points_scat::AbstractArray, python_points_meas::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, discard_absorption = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape_scat = size(python_points_scat)
        dim = shape_scat[2]
        
        shape_meas = size(python_points_meas)
        dim_meas = shape_meas[2]
        
        if dim != 3 || dim_meas != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n_scat = shape_scat[1]
        n_meas = shape_meas[1]
        points_scat = [PointdD(python_points_scat[k,:]) for k in 1:n_scat]
        points_meas = [PointdD(python_points_meas[k,:]) for k in 1:n_meas]
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * (exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        if discard_absorption
            alpha_ = real(alpha)
        else
            alpha_ = alpha
        end
        
        # K is an abstract representation of the kernel. Here, need it for both M and G
        K = GreensMatrixScalar(points_scat, points_scat, k0, alpha_, radius, regularize, G0_center_value, true)
        K_prop = GreensMatrixScalar(points_meas, points_scat, k0, alpha_, radius, regularize, G0_center_value, false)
        K_prop_T = GreensMatrixScalar(points_scat, points_meas, k0, alpha_, radius, regularize, G0_center_value, false)
        
        # Need pointsclt with right size!
        pointsclt_scat = ClusterTree(points_scat)
        pointsclt_meas = ClusterTree(points_meas)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol,rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt_scat, pointsclt_scat; adm,comp, threads=threads, distributed=false)
        H_prop = assemble_hmatrix(K_prop, pointsclt_meas, pointsclt_scat; adm,comp, threads=threads, distributed=false)
        H_prop_T = assemble_hmatrix(K_prop_T, pointsclt_scat, pointsclt_meas; adm,comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (propagate): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_calc.svg")
        end
        
        # Throughout the calculation, GT.G is needed 
        GTG = assemble_hmatrix(K, pointsclt_scat, pointsclt_scat; adm,comp, threads=threads, distributed=false) # Initialization... similar does not work here
        HMatrices.hmul!(GTG, H_prop_T, H_prop, 1, 0, comp) # Compute GTG <- GT.G * 1 - 0 * GT
        
        # Need to perform inverse on M: do it using the LU decomposition
        # Other solutions like below instantiate W in memory
        # W = lu_decomp\Matrix(1.0I,n_scat,n_scat)
        # W = ldiv!(lu_decomp,Matrix((1.0+0im)*I,n_scat,n_scat))
        lu_decomp = lu!(H; atol = atol, rtol = rtol)
        
        sum = 0.0 + 0im
        
        # Instead of trying to store the (large) inverse, treat columns separately!
        for k in 1:n_scat
            
            # Compute a COLUMN of the inverse matrix
            unit_vector = zeros(n_scat)
            unit_vector[k] = 1.0
            inv_column = lu_decomp \ unit_vector
            
            # Deduce a LINE of the matrix GT.G.W
            matrix_line = similar(inv_column)
            mul!(matrix_line,GTG,inv_column,1,0; threads=threads)
            
            # The element that matters for the trace is the k-th element
            sum += matrix_line[k]
        end
        
        sum /= n_meas
        # The G has a k0**2 alpha prefactor in this code
        # rho should be 2 pi k0 alpha * trace
        # it is currently k0^4 * alpha^2
        # for scalar waves: extra factor of 2!
        sum *= 4.0 * pi / (k0^3 * alpha_)
        
        # rho is the imaginary part of the Green function of the medium
        mean_dos = imag(sum)
        
        return mean_dos
        
    end
    
    function ldos_scalar(python_points_scat::AbstractArray, python_points_meas::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, discard_absorption = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape_scat = size(python_points_scat)
        dim = shape_scat[2]
        
        shape_meas = size(python_points_meas)
        dim_meas = shape_meas[2]
        
        if dim != 3 || dim_meas != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n_scat = shape_scat[1]
        n_meas = shape_meas[1]
        points_scat = [PointdD(python_points_scat[k,:]) for k in 1:n_scat]
        points_meas = [PointdD(python_points_meas[k,:]) for k in 1:n_meas]
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * (exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        if discard_absorption
            alpha_ = real(alpha)
        else
            alpha_ = alpha
        end
        
        # K is an abstract representation of the kernel. Here, need it for both M and G
        K = GreensMatrixScalar(points_scat, points_scat, k0, alpha_, radius, regularize, G0_center_value, true)
        K_prop = GreensMatrixScalar(points_meas, points_scat, k0, alpha_, radius, regularize, G0_center_value, false)
        
        # Need pointsclt with right size!
        pointsclt_scat = ClusterTree(points_scat)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol,rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt_scat, pointsclt_scat; adm,comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (propagate): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_calc.svg")
        end
        
        # Need to perform inverse on M: do it using the LU decomposition
        # Other solutions like below instantiate W in memory
        # W = lu_decomp\Matrix(1.0I,n_scat,n_scat)
        # W = ldiv!(lu_decomp,Matrix((1.0+0im)*I,n_scat,n_scat))
        lu_decomp = lu!(H; atol = atol, rtol = rtol)
        
        ldos_list = zeros(ComplexF64,n_meas)
        col = zeros(ComplexF64, n_scat)
        
        # Instead of trying to store the (large) inverse, treat columns separately!
        # XXX This is just because this library does not have a good hierarchical inverse!
        for k in 1:n_scat
            
            # Compute a COLUMN of the inverse matrix
            unit_vector = zeros(n_scat)
            unit_vector[k] = 1.0
            inv_column = lu_decomp \ unit_vector
        
            # Only the trace is needed for the LDOS: accumulate the sum that defines trace elements
            for row in 1:n_meas
                for j in 1:n_scat
                    col[j] = getindex(K_prop,row,j)
                end
                output_row = row
                ldos_list[output_row] += col[k] * sum(vx*vy for (vx,vy) in zip(col, inv_column)) # NOT dot(col,inv_column) # dot conjugates!
            end
        end
        
        # The G has a k0**2 alpha prefactor in this code
        # rho should be 2 pi k0 alpha * trace
        # it is currently k0^4 * alpha^2
        # for scalar waves: extra factor of 2!
        ldos_list .*= 4.0 * pi / (k0^3 * alpha_)
        
        # rho is the imaginary part of the Green function of the medium
        ldos_list = imag.(ldos_list)
        
        return ldos_list
        
    end
    
    function spectrum_scalar(python_points::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape = size(python_points)
        dim = shape[2]
        
        if dim != 3
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n = shape[1]
        PointdD = SVector{dim,Float64}
        points = [PointdD(python_points[k,:]) for k in 1:n]
        
        if self_interaction
            # G0 integrated over a finite ball
            volume = 4.0 * pi * radius^3 / 3.0
            G0_center_value = (1.0 / (k0*k0*volume)) * (exp(1.0im*k0*radius)*(1.0- 1.0im*k0*radius) - 1.0) 
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensMatrixScalar(points, points, k0, alpha, radius, regularize, G0_center_value, true)
        
        # Need pointsclt with right size!
        pointsclt = ClusterTree(points)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol, rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt, pointsclt; adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (solve): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, margin = 0px) #left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot.svg")
        end
        
        lambdas = eigvals(H) # XXX Not implemented
        
        return lambdas
        
    end
    
end