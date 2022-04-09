module CayleyMengerDeterminant
using LinearAlgebra
import ArrayInterface
using Static
import InverseFunctions

### utility functions

@inline binomial2(x::Int)::Int = binomial(x, 2)

@inline inverse_binomial2(x::Int)::Int = (convert(Int, sqrt(8x+1)) + 1) ÷ 2

InverseFunctions.inverse(::typeof(binomial2)) = inverse_binomial2

InverseFunctions.inverse(::typeof(inverse_binomial2)) = binomial2

@inline index_triangular_nodiag(ixA::Int, ixB::Int)::Int = binomial2(ixA - 1) + ixB

export binomial2, inverse_binomial2

### main matrix type

struct CayleyMengerDistanceMatrix{T<:Real,Sz<:Union{Int,StaticInt}} <: AbstractMatrix{T}
    simplex_dimensions::Sz
    square_distances::Vector{T}
end

CayleyMengerDistanceMatrix(onlyPoint::Tuple{}) = CayleyMengerDistanceMatrix{Union{},StaticInt{0}}(StaticInt(0), (Union{})[])

function CayleyMengerDistanceMatrix(
    first_point::NTuple{N,T},
    other_points::Vararg{NTuple{N,T},N},
) where {N,T<:Real}
    points = (first_point, other_points...)

    sqr_dists = Array{T}(undef, binomial2(N + 1))
    for pointIxA = 1:(N+1)
        for pointIxB = 1:(pointIxA-1)
            pos = index_triangular_nodiag(pointIxA, pointIxB)
            sqr_dists[pos] = sum((points[pointIxA] .- points[pointIxB]) .^ 2)
        end
    end

    CayleyMengerDistanceMatrix{T,StaticInt{N}}(StaticInt(N), sqr_dists)
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
    (i == n || j == n) && return oneunit(T)
    return A.square_distances[index_triangular_nodiag(max(i, j), min(i, j))]
end

### LinearAlgebra interface

LinearAlgebra.issymmetric(::CayleyMengerDistanceMatrix) = true

LinearAlgebra.ishermitian(::CayleyMengerDistanceMatrix) = true

### ArrayInterface integration

ArrayInterface.can_change_size(::Type{<:CayleyMengerDistanceMatrix}) = false

ArrayInterface.can_setindex(::Type{<:CayleyMengerDistanceMatrix}) = false

ArrayInterface.has_sparsestruct(::CayleyMengerDistanceMatrix) = true

ArrayInterface.ismutable(::CayleyMengerDistanceMatrix) = false

ArrayInterface.isstructured(::CayleyMengerDistanceMatrix) = true

ArrayInterface.known_last(::CayleyMengerDistanceMatrix{T}) where {T <: Real} = oneunit(T)

ArrayInterface.parent(A::CayleyMengerDistanceMatrix{T}) where {T <: Real} = convert.(T, A.square_distances)

function ArrayInterface.findstructralnz(A::CayleyMengerDistanceMatrix{T}) where {T <: Real}
    n = A.simplex_dimensions + 1
    v = n*n - n
    I = 1:n
    J = Vector{Int}(undef, n)
    V = Vector{T}(undef, v)
    k = 0
    for i in 1:n
        for j in 1:(i-1)
            k += 1
            J[k] = j
            V[k] = A.square_distances[index_triangular_nodiag(i, j)]
        end
    end
    I, J, V 
end

ArrayInterface.getindex(A::CayleyMengerDistanceMatrix, i::Int, j::Int) = A[i,j]

ArrayInterface.size(A::CayleyMengerDistanceMatrix{T,StaticInt{N}}) where {T<:Real,N} =
    (StaticInt(N+2), StaticInt(N+2))

ArrayInterface.size(A::CayleyMengerDistanceMatrix{T,Int}) where {T<:Real} =
    (A.simplex_dimensions + 2, A.simplex_dimensions + 2)

# the operation

function simplex_volume(
    points::NTuple{N,T}...;
    distance_type::Union{Nothing,Type{<:Real}} = nothing,
) where {N,T<:Real}
    Ty = isnothing(distance_type) ? T : distance_type
    distances = CayleyMengerDistanceMatrix(points...)
    distances_converted = convert(Ty, distances)
    sqrt(abs(det(distances_converted)) / (2^N)) / factorial(N)
end

simplex_volume(::Tuple{};distance_type::Union{Nothing,Type{<:Real}} = nothing) =
    isnothing(distance_type) ? false : zero(distance_type)

export simplex_volume

end
