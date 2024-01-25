module Transmission2D
    using SpecialFunctions
    using HMatrices # https://github.com/WaveProp/HMatrices.jl?tab=readme-ov-file https://waveprop.github.io/HMatrices.jl/dev/
    using StaticArrays
    using LinearAlgebra
    using IterativeSolvers
    # XXX DEBUG
    using Plots

    PointdD = SVector{2,Float64}
    # Since this is a block-wise matrix, follow the steps in https://waveprop.github.io/HMatrices.jl/dev/
    struct GreensTEMatrix <: AbstractMatrix{ComplexF64}
        X::Vector{PointdD}
        Y::Vector{PointdD}
        k0::Float64
        alpha::ComplexF64
        radius::Float64
        regularize::Bool
        G0_center_value::ComplexF64
    end
    
    function M_TE(x,y,row,col, k0, alpha,radius,regularize,G0_center_value)::ComplexF64
        if x == y && row == col
            # Diagonal has an identity: this is not dangerous since it's only STRICTLY at the same point
            extra = 1
        else
            extra = 0
        end
        extra - k0*k0*alpha*G_TE(x,y,row,col,k0,radius,regularize,G0_center_value)
    end
    
    function G_TE(x,y,row,col,k0,radius,regularize,G0_center_value)::ComplexF64
        d = norm(x-y)
        RxR = x[row]*y[col]
        RdotR = dot(x,y)
        RxR /= RdotR
        RxR *= k0
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
    
    Base.getindex(K::GreensTEMatrix,i::Int,j::Int) = M_TE(K.X[1+i%2], K.Y[1+j%2], 1+i%2, 1+j%2, K.k0, K.alpha, K.radius, K.regularize, K.G0_center_value)
    Base.size(K::GreensTEMatrix) = 2*length(K.X), 2*length(K.Y)
    
    function solve_TM(python_points::AbstractArray, points_Einc::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 1e-6)
        
        println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
        println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        
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
        
        function M(x,y)::ComplexF64
            if x == y 
                # Diagonal has an identity: this is not dangerous since it's only STRICTLY at the same point
                extra = 1
            else
                extra = 0
            end
            extra - k0*k0*alpha*G(x,y)
        end
        
        function G(x,y)::ComplexF64
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
        
        # use_threads() = true # XXX Is that truly how it works? Useful?
        
        # K is an abstract representation of the kernel
        K = KernelMatrix(M,points,points)
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K;atol=atol)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression: $(HMatrices.compression_ratio(H))")
        
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
            lu_decomp = lu!(H; atol = atol)
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
    
    
    function solve_TE(python_points::AbstractArray, points_Einc::AbstractArray, k0, alpha, radius, self_interaction; regularize = false, use_lu = true, atol = 1e-6)
        
        println("Number of threads used by julia (UNSAFE if >1 through python!): $(Threads.nthreads())")
        println("Number of threads used by BLAS: $(BLAS.get_num_threads())")
        
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
        K = GreensTEMatrix(points,points,k0,alpha,radius,regularize,G0_center_value)
        print(K)
        # Need pointsclt with right size!
        pointsproxy = [points[1+floor(Int64,k/2)] for k in 0:dim*n-1]
        pointsclt = ClusterTree(pointsproxy)
        adm = StrongAdmissibilityStd()
        comp = PartialACA(;atol=atol)
        
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K,pointsclt,pointsclt;adm,comp,threads=false,distributed=false)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression: $(HMatrices.compression_ratio(H))")
        
        # Maybe the loop can be bypassed, for now doing it brute-force
        field_shape = size(points_Einc)
        n_angles = field_shape[2]
        
        # Initialize output structure
        points_Etot = similar(points_Einc)
        
        # Initialize static structure for rhs
        PointND = SVector{dim*n,ComplexF64}
        
        if use_lu
            # Use an LU decomposition to make solving faster in the angle loop
            # XXX MAY NEED EXPLICIT PIVOT OPTION?
            lu_decomp = lu!(H; atol = atol)
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
    
end
