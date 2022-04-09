using CayleyMengerDeterminant
import InverseFunctions
import Static: StaticInt
import StaticArrays: Dynamic
using Test

@testset "CayleyMengerDeterminant.jl" begin
    InverseFunctions.test_inverse.(
        binomial2,
        [1, 2, 3, 4, 5, 6, Dynamic(), StaticInt(2), StaticInt(3)],
        compare = isequal,
    )
    InverseFunctions.test_inverse.(
        inverse_binomial2,
        [0, 1, 3, 6, 10, 15, Dynamic(), StaticInt(1), StaticInt(3)],
        compare = isequal,
    )

    @testset "index_triangular_nodiag" begin
        N = 100
        v = zeros(Int, binomial(N, 2))
        for ixA = 1:N
            for ixB = 1:(ixA-1)
                v[CayleyMengerDeterminant.index_triangular_nodiag(ixA, ixB)] += 1
            end
        end
        for k in v
            @test k == 1
        end
    end

    @testset "CayleyMengerDistanceMatrix" begin
        m0 = CayleyMengerDistanceMatrix(())
        @test m0.simplex_dimensions == 0
        @test size(m0.square_distances) == (0,)


        m1 = CayleyMengerDistanceMatrix((0,), (1,))
        @test m1.simplex_dimensions == 1
        @test size(m1.square_distances) == (1,)
        @test m1.square_distances == [1]
        @test m1 == CayleyMengerDistanceMatrix((1,), (2,))
        @test m1 == CayleyMengerDistanceMatrix((1,), (0,))


        m3 = CayleyMengerDistanceMatrix((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        @test m3.simplex_dimensions == 3
        @test size(m3.square_distances) == (6,)
        @test m3.square_distances == [1, 1, 2, 1, 2, 2]
        @test m3 == CayleyMengerDistanceMatrix((1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1))
        @test m3 == CayleyMengerDistanceMatrix([1 1 1; 1 1 0; 0 1 1; 1 0 1])
    end

    @testset "simplex_volume" begin
        @test simplex_volume(()) == false
        @test simplex_volume((), distance_type = Int) == 0

        @test simplex_volume((0,), (1,)) == 1
        @test simplex_volume((1,), (2,)) == 1
        @test simplex_volume((1,), (0,)) == 1
        @test simplex_volume((2,), (0,)) == 2
        @test simplex_volume((-1,), (1,)) == 2

        @test simplex_volume((0, 0), (1, 0), (0, 1)) == 1 / 2
        @test simplex_volume((0, 0), (0, 2), (-2, 0)) == 2
        @test simplex_volume((0, 0), (0, 2), (-2, 0), distance_type = Int) == 2

        @test simplex_volume((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)) == 1 / 6
        @test simplex_volume((1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)) == 1 / 6
        @test simplex_volume((1, 1, 1), (1, 1, 2), (2, 1, 1), (1, 2, 1)) == 1 / 6
        @test simplex_volume((0, 0, 0), (0, 1, 2), (2, 1, 0), (0, 2, 0)) == 4 / 3

        @test simplex_volume([0 0 0; 1 0 0; 0 1 0; 0 0 1]) == 1 / 6
    end
end
