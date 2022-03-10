using CayleyMengerDeterminant
using Test

@testset "CayleyMengerDeterminant.jl" begin
    @testset "index_triangular_nodiag" begin
        N = 100
        v = zeros(Int, binomial(N, 2))
        for ixA = 1:N
            for ixB = 1:(ixA-1)
                v[CayleyMengerDeterminant.index_triangular_nodiag(ixA, ixB)] += 1
            end
        end
        @test all(v .== 1)
    end

    @testset "CayleyMengerDistanceMatrix" begin
        m0 = CayleyMengerDistanceMatrix(())
        @test m0.simplex_dimensions == 0
        @test size(m0.square_distances) == (0,)


        m1 = CayleyMengerDistanceMatrix((0,), (1,))
        @test m1.simplex_dimensions == 1
        @test size(m1.square_distances) == (1,)
        @test m1.square_distances[1] == 1
        @test m1 == CayleyMengerDistanceMatrix((1,), (2,))
    end
end
