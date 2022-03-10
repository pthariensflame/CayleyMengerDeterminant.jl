module CayleyMengerDeterminant
using LinearAlgebra

index_triangular_nodiag(ixA, ixB) = binomial(ixA - 1, 2) + ixB

struct CayleyMengerDistanceMatrix{T,V<:AbstractVector{T}} <: AbstractMatrix{T}
    simplexDimensions::Int
    squareDistances::V
end
CayleyMengerDistanceMatrix(onlyPoint::Tuple{}) =
    CayleyMengerDistanceMatrix(1, (Union{})[])
function CayleyMengerDistanceMatrix(
    firstPoint::NTuple{N,T},
    otherPoints::Vararg{NTuple{N,T},N},
) where {N,T}
    points = (firstPoint, otherPoints...)

    sqrDists = Array{T}(undef, binomial(N+1, 2))
    for pointIxA in 1:(N+1)
        for pointIxB in 1:(pointIxA-1)
            pos = index_triangular_nodiag(pointIxA, pointIxB)
            sqrDist = sum((points[pointIxA] .- points[pointIxB]).^2)
            sqrDists[pos] = sqrDist
        end
    end

    CayleyMengerDistanceMatrix(N+1, sqrDists)
end

Base.:(==)(A::CayleyMengerDistanceMatrix, B::CayleyMengerDistanceMatrix) =
    (A.simplexDimensions == B.simplexDimensions) && all(A.squareDistances .== B.squareDistances)

Base.isequal(A::CayleyMengerDistanceMatrix, B::CayleyMengerDistanceMatrix) =
    (A.simplexDimensions == B.simplexDimensions) && all(Base.isequal.(A.squareDistances, B.squareDistances))

function Base.hash(A::CayleyMengerDistanceMatrix, h::UInt)
    nh = h
    nh = Base.hash(A.simplexDimensions, nh)
    nh = Base.hash(A.squareDistances, nh)
    nh
end

Base.widen(A::CayleyMengerDistanceMatrix) = CayleyMengerDistanceMatrix(A.simplexDimensions, widen.(A.squareDistances))

export CayleyMengerDistanceMatrix

end
