module Transmission2D
    using SpecialFunctions
    using HMatrices
    using StaticArrays
    using LinearAlgebra
    using IterativeSolvers

    function solve_TM(python_points::AbstractArray, points_Einc, k0, alpha, radius, self_interaction, regularize)
        
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
        
        # K is an abstract representation of the kernel
        K = KernelMatrix(M,points,points)
        # H is a hierarchical compression of the matrix, atol and rtol can be tuned in principle
        H = assemble_hmatrix(K;atol=1e-6)
        
        # Print this for consistency checks for now
        println("Compression ratio of hierarchical compression: $(HMatrices.compression_ratio(H))")
        
        # Maybe the loop can be bypassed, for now doing it brute-force
        field_shape = size(points_Einc)
        n_angles = field_shape[2]
        
        # Initialize output structure
        points_Etot = similar(points_Einc)
        
        # Initialize static structure for rhs
        PointND = SVector{n,ComplexF64}
        
        # TODO play with lu within hmatrices, it would in principle be faster!
        for angle_index in 1:n_angles
            inc_field_vec = PointND(points_Einc[:,angle_index])
            precondit = similar(inc_field_vec)
            approx_fields = gmres!(precondit, H, inc_field_vec)
            points_Etot[:,angle_index] = approx_fields[:]
        end
        
        # TODO Structure for lu:
        # MAY NEED PIVOT!
        # F = lu(H; atol = 1e-6)
        # points_Etot = F\points_Einc
        
        return points_Etot
        
    end
end
