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
        n = shape[1]
        PointdD = SVector{dim,Float64}
        points = [PointdD(python_points[k,:]) for k in 1:n]
        
        if self_interaction
            volume = pi*radius*radius
            G0_center_value = (-1.0 / (k0*k0*volume)) + 0.5im * hankelh1(1,k0*radius)/k0*radius
        else
            G0_center_value = 0.0
        end
        
        function M(x,y)::ComplexF64
            if x == y 
                extra = 1
            else
                extra = 0
            end
            extra - k0*k0*alpha*G(x,y)
        end
        
        function G(x,y)::ComplexF64
            d = norm(x-y)
            
            if regularize
                threshold = radius
            else
                threshold = 0.0
            end
            
            if d <= threshold
                G0_center_value
            else
                0.25im * hankelh1(0, k0 * d)
            end
        end
        
        
        K = KernelMatrix(M,points,points)
        H = assemble_hmatrix(K;atol=1e-6)
        
        println(HMatrices.compression_ratio(H))
        
        field_shape = size(points_Einc)
        n_angles = field_shape[2]
        
        points_Etot = similar(points_Einc)
        
        PointND = SVector{n,ComplexF64}
        
        for angle_index in 1:n_angles
            inc_field_vec = PointND(points_Einc[:,angle_index])
            precondit = similar(inc_field_vec)
            approx_fields = gmres!(precondit, H, inc_field_vec)
            points_Etot[:,angle_index] = approx_fields[:]
        end
        
        # points_Etot = similar(points_Einc)
        # approx_fields = gmres!(points_Etot, H, points_Einc)
        # F = lu(H; atol = 1e-6)
        # points_Etot = F\points_Einc
        
        return points_Etot
        
    end
end
