module Transmission2D
    using SpecialFunctions
    using HMatrices # https://github.com/WaveProp/HMatrices.jl?tab=readme-ov-file https://waveprop.github.io/HMatrices.jl/dev/
    using StaticArrays
    using LinearAlgebra
    using IterativeSolvers
    # XXX DEBUG
    using Plots
    using Plots.PlotMeasures

    # Define static vector with fixed size globally
    PointdD = SVector{2,Float64}
    
    # Even though it's not strictly necessary for TM, follow the steps in https://waveprop.github.io/HMatrices.jl/dev/
    # This is more consistent with the TE definition and makes it easier to modify
    struct GreensTMMatrix <: AbstractMatrix{ComplexF64}
        X::Vector{PointdD}
        Y::Vector{PointdD}
        k0::Float64
        alpha::ComplexF64
        radius::Float64
        regularize::Bool
        G0_center_value::ComplexF64
        solve::Bool
    end
    
    function M_TM(x, y, k0, alpha, radius, regularize, G0_center_value, solve)::ComplexF64
        if solve
            if x == y 
                # Diagonal has an identity: this is not dangerous since it's only STRICTLY at the same point
                extra = 1
            else
                extra = 0
            end
            extra - k0*k0*alpha*G_TM(x, y, k0, radius, regularize, G0_center_value)
        else
            k0*k0*alpha*G_TM(x, y, k0, radius, regularize, G0_center_value)
        end
    end
    
    function G_TM(x, y, k0, radius, regularize, G0_center_value)::ComplexF64
        d = norm(x-y)
        
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
            # TM propagator
            0.25im * hankelh1(0, k0 * d)
        end
    end
    
    Base.getindex(K::GreensTMMatrix,i::Int,j::Int) = M_TM(K.X[i], K.Y[j], K.k0, K.alpha, K.radius, K.regularize, K.G0_center_value, K.solve)
    Base.size(K::GreensTMMatrix) = length(K.X), length(K.Y)
    
    # Since this is a block-wise matrix, follow the steps in https://waveprop.github.io/HMatrices.jl/dev/
    struct GreensTEMatrix <: AbstractMatrix{ComplexF64}
        X::Vector{PointdD}
        Y::Vector{PointdD}
        k0::Float64
        alpha::ComplexF64
        radius::Float64
        regularize::Bool
        G0_center_value::ComplexF64
        solve::Bool
    end
    
    function M_TE(x, y, row, col, k0, alpha, radius, regularize, G0_center_value, solve)::ComplexF64
        if solve
            if x == y && row == col
                # Diagonal has an identity: this is not dangerous since it's only STRICTLY at the same point
                extra = 1
            else
                extra = 0
            end
            extra - k0*k0*alpha*G_TE(x,y,row,col,k0,radius,regularize,G0_center_value)
        else 
            k0*k0*alpha*G_TE(x,y,row,col,k0,radius,regularize,G0_center_value)
        end
    end
    
    function G_TE(x, y, row, col, k0, radius, regularize, G0_center_value)::ComplexF64
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
            # TE propagator
            if row == col
                I = 1
            else
                I = 0
            end
            0.25im*((I-RxR)*hankelh1(0,R)-(I-2*RxR)*(hankelh1(1,R)/R))
            
        end
    end
    
    function block_id(i)::Int64
        # Return index of point to use from index in G_TE matrix
        floor(Int64, (i-1)/2) + 1
    end
    
    Base.getindex(K::GreensTEMatrix,i::Int,j::Int) = M_TE(K.X[block_id(i)], K.Y[block_id(j)], 1+(i-1)%2, 1+(j-1)%2, K.k0, K.alpha, K.radius, K.regularize, K.G0_center_value, K.solve)
    Base.size(K::GreensTEMatrix) = 2*length(K.X), 2*length(K.Y)
    
    function solve_TM(python_points::AbstractArray, points_Einc::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape = size(python_points)
        dim = shape[2]
        
        if dim != 2
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n = shape[1]
        points = [PointdD(python_points[k,:]) for k in 1:n]
        
        if self_interaction
            # G0 integrated over a finite disk
            volume = pi*radius*radius
            G0_center_value = (-1.0 / (k0*k0*volume)) + 0.5im * hankelh1(1,k0*radius)/(k0*radius)
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensTMMatrix(points, points, k0, alpha, radius, regularize, G0_center_value, true)
        
        # Need pointsclt with right size!
        pointsclt = ClusterTree(points)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt, pointsclt;adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (solve TM): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_TM.svg")
        end
        
        # Maybe the loop can be bypassed, for now doing it brute-force
        field_shape = size(points_Einc)
        n_angles = field_shape[2]
        
        # Initialize output structure
        points_Etot = similar(points_Einc)
        
        # Initialize static structure for rhs
        PointND = SVector{n,ComplexF64}
        
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
    
    
    function solve_TE(python_points::AbstractArray, points_Einc::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape = size(python_points)
        dim = shape[2]
        
        if dim != 2
            println("Wrong dimensionality for points!")
            exit()
        end
        
        # Needed conversion for HMatrices!
        # There is probably a better way to do this memory-wise
        n = shape[1]
        PointdD = SVector{dim,Float64}
        points = [PointdD(python_points[k,:]) for k in 1:n]
        
        if self_interaction
            # G0 integrated over a finite disk
            volume = pi*radius*radius
            G0_center_value = (-1.0 / (k0*k0*volume)) + 0.25im * hankelh1(1,k0*radius)/(k0*radius)
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensTEMatrix(points, points, k0, alpha, radius, regularize, G0_center_value, true)
        
        # Need pointsclt with right size!
        pointsproxy = [points[1+floor(Int64,k/2)] for k in 0:dim*n-1]
        pointsclt = ClusterTree(pointsproxy)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol, rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt, pointsclt; adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (solve TE): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, margin = 0px) #left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_TE.svg")
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
    
    function calc_TM(python_points_scat::AbstractArray, python_points_meas::AbstractArray, points_Escat::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
        
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape_scat = size(python_points_scat)
        dim = shape_scat[2]
        
        shape_meas = size(python_points_meas)
        dim_meas = shape_meas[2]
        
        if dim != 2 || dim_meas != 2
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
            # G0 integrated over a finite disk
            volume = pi*radius*radius
            G0_center_value = (-1.0 / (k0*k0*volume)) + 0.5im * hankelh1(1,k0*radius)/(k0*radius)
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensTMMatrix(points_meas, points_scat, k0, alpha, radius, regularize, G0_center_value, false)
        
        # Need pointsclt with right size!
        pointsclt_scat = ClusterTree(points_scat)
        pointsclt_meas = ClusterTree(points_meas)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol,rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt_meas, pointsclt_scat; adm,comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (calc TM): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_TM_calc.svg")
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

    function calc_TE(python_points_scat::AbstractArray, python_points_meas::AbstractArray, points_Escat::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 0, rtol = 1e-2, debug=false, threads=true)
            
        if debug
            println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
            println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        end
        
        shape_scat = size(python_points_scat)
        dim = shape_scat[2]
        
        shape_meas = size(python_points_meas)
        dim_meas = shape_meas[2]
        
        if dim != 2 || dim_meas != 2
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
            # G0 integrated over a finite disk
            volume = pi*radius*radius
            G0_center_value = (-1.0 / (k0*k0*volume)) + 0.25im * hankelh1(1,k0*radius)/(k0*radius)
        else
            # G0 discarded at center if volume is neglected completely
            G0_center_value = 0.0
        end
        
        # K is an abstract representation of the kernel
        K = GreensTEMatrix(points_meas,points_scat,k0,alpha,radius,regularize,G0_center_value, false)
        
        # Need pointsclt with right size!
        pointsproxy_scat = [points_scat[1+floor(Int64,k/2)] for k in 0:dim*n_scat-1]
        pointsclt_scat = ClusterTree(pointsproxy_scat)
        pointsproxy_meas = [points_meas[1+floor(Int64,k/2)] for k in 0:dim*n_meas-1]
        pointsclt_meas = ClusterTree(pointsproxy_meas)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol, rtol=rtol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K, pointsclt_meas, pointsclt_scat; adm, comp, threads=threads, distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression (calc TE): $(HMatrices.compression_ratio(H))")
        
        if debug
            plot(H, axis=nothing, legend=false, border=:none, left_margin = 0px, right_margin = 0px, bottom_margin = 0px, top_margin = 0px)
            savefig("testplot_TM_calc.svg")
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

end