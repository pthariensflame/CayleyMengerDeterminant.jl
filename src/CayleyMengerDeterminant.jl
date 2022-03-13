module CayleyMengerDeterminant
using LinearAlgebra

### indexing utility

index_triangular_nodiag(ixA, ixB) = binomial(ixA - 1, 2) + ixB

### main matrix type

struct CayleyMengerDistanceMatrix{T<:Real} <: AbstractMatrix{T}
    simplex_dimensions::Int
    square_distances::Vector{T}
end

CayleyMengerDistanceMatrix(onlyPoint::Tuple{}) = CayleyMengerDistanceMatrix(0, (Union{})[])

function CayleyMengerDistanceMatrix(
    first_point::NTuple{N,T},
    other_points::Vararg{NTuple{N,T},N},
) where {N,T<:Real}
    points = (first_point, other_points...)

    sqr_dists = Array{T}(undef, binomial(N + 1, 2))
    for pointIxA = 1:(N+1)
        for pointIxB = 1:(pointIxA-1)
            pos = index_triangular_nodiag(pointIxA, pointIxB)
            sqr_dists[pos] = sum((points[pointIxA] .- points[pointIxB]) .^ 2)
        end
    end

    CayleyMengerDistanceMatrix(N, sqr_dists)
end

export CayleyMengerDistanceMatrix

### equality and hashing

Base.:(==)(A::CayleyMengerDistanceMatrix, B::CayleyMengerDistanceMatrix) =
    (A.simplex_dimensions == B.simplex_dimensions) &&
    all(A.square_distances .== B.square_distances)

Base.isequal(A::CayleyMengerDistanceMatrix, B::CayleyMengerDistanceMatrix) =
    (A.simplex_dimensions == B.simplex_dimensions) &&
    all(isequal.(A.square_distances, B.square_distances))

Base.hash(A::CayleyMengerDistanceMatrix, h::UInt) =
    hash(A.square_distances, hash(A.simplex_dimensions, h))

# conversion and promotion

Base.widen(A::CayleyMengerDistanceMatrix) =
    CayleyMengerDistanceMatrix(A.simplex_dimensions, widen.(A.square_distances))

Base.convert(T::Type{<:Real}, A::CayleyMengerDistanceMatrix) =
    CayleyMengerDistanceMatrix(A.simplex_dimensions, convert.(T, A.square_distances))

Base.promote_rule(
    ::Type{<:CayleyMengerDistanceMatrix{T}},
    ::Type{<:CayleyMengerDistanceMatrix{U}},
) where {T<:Real,U<:Real} = CayleyMengerDistanceMatrix{promote_type(T, U)}

### AbstractArray interface

Base.IndexStyle(::Type{<:CayleyMengerDistanceMatrix}) = IndexCartesian()

Base.size(A::CayleyMengerDistanceMatrix) =
    (A.simplex_dimensions + 2, A.simplex_dimensions + 2)

Base.length(A::CayleyMengerDistanceMatrix) = (A.simplex_dimensions + 2)^2

function Base.getindex(A::CayleyMengerDistanceMatrix{T}, i::Int, j::Int) where {T<:Real}
    n = A.simplex_dimensions + 2
    i == j && return zero(T)
    (i == n || j == n) && return one(T)
    return A.square_distances[index_triangular_nodiag(max(i, j), min(i, j))]
end

### LinearAlgebra interface

LinearAlgebra.issymmetric(A::CayleyMengerDistanceMatrix) = true

LinearAlgebra.ishermitian(A::CayleyMengerDistanceMatrix) = true

# the operation

function simplex_volume(
    points::NTuple{N,T}...;
    distance_type::Union{Nothing,Type{<:Real}} = nothing,
) where {N,T<:Real}
    distances = CayleyMengerDistanceMatrix(points...)
    distances.simplex_dimensions == 0 && return false
    distances_converted = isnothing(distance_type) ? distances : convert(distances, A)
    sqrt(abs(det(distances_converted)) / (2^N)) / factorial(N)
end

export simplex_volume

end
