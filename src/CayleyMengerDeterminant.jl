module CayleyMengerDeterminant
using LinearAlgebra
import ArrayInterface
using Static: StaticInt
using StaticArrays
import StaticArrays: Dynamic
import InverseFunctions

### utility functions

"""
    binomial2(x::Union{Int,StaticInt,Dynamic}) -> Union{Int,StaticInt,Dynamic}

Compute the second binomial coefficient of `x`.

This function is a convenience shorthand for `binomial(x, 2)`.

```julia-repl
julia> binomial2(4)
6
```
"""
function binomial2 end

@inline binomial2(x::Int)::Int = binomial(x, 2)

@inline binomial2(::StaticInt{X}) where {X} = StaticInt{binomial2(x)}

@inline binomial2(::Dynamic) = Dynamic()

"""
    inverse_binomial2(x::Union{Int,StaticInt,Dynamic) -> Union{Int,StaticInt,Dynamic}

Compute the integer value that `x` is the second binomial coefficient of.

This function is the inverse of `binomial2`.

```julia-repl
julia> inverse_binomial2(6)
4
```
"""
function inverse_binomial2 end

@inline inverse_binomial2(x::Int)::Int = (convert(Int, sqrt(8x + 1)) + 1) ÷ 2

@inline inverse_binomial2(::StaticInt{X}) where {X} = StaticInt{inverse_binomial2(x)}

@inline inverse_binomial2(::Dynamic) = Dynamic()

InverseFunctions.inverse(::typeof(binomial2)) = inverse_binomial2

InverseFunctions.inverse(::typeof(inverse_binomial2)) = binomial2

"""
    CayleyMengerDeterminant.index_triangular_nodiag(ixA::Int, ixB::Int) -> Int

For row `ixA` and column `ixB`, compute the linear index into the flat vector storage for a symmetric or
lower-triangular matrix with a zero diagonal.


`ixA` and `ixB` are both 1-based and contiguous, and `ixA` is assumed to be greater than or equal to `ixB`.

```julia-repl
julia> CayleyMengerDeterminant.index_triangular_nodiag(4,2)
5
```
"""
@inline index_triangular_nodiag(ixA::Int, ixB::Int)::Int = binomial2(ixA - 1) + ixB

export binomial2, inverse_binomial2

### main matrix type

"""
    CayleyMengerDistanceMatrix{T,Sz}(simplex_dimensions::Int, square_distances::Vector{T}) where {T<:Real,Sz::Union{Int,Dynamic}}
    CayleyMengerDistanceMatrix(points::Vararg{NTuple{N,T},N+1})::CayleyMengerDistanceMatrix{T,N} where {T<:Real,N::Int}
    CayleyMengerDistanceMatrix(P::AbstractMatrix{T})::CayleyMengerDistanceMatrix{T,Dynamic} where {T<:Real}

The zero-diagonal symmetric matrix of square distances among the points of an `N`-simplex, backed by efficient linear storage.

When providing `points`, it must be N+1 tuples of N values each, which are the coordinates of the points of the `N`-simplex.
When providing `P`, it must be a tall near-square matrix with N+1 rows and N columns, where the rows are the coordinates of the
points of the `N`-simplex.
When providing `simplex_dimensions` and `square_distances` directly, `simplex_dimensions` must be the integer number of dimension
`N` of the `N-simplex, and `square_distances` must be the precomputed backing linear storage of the square distances among the
points of the `N`-simplex and so must have length `binomial2(simplex_dimensions + 1)`.
"""
struct CayleyMengerDistanceMatrix{T<:Real,Sz} <: AbstractMatrix{T}
    "The number of dimensions `N` of the `N`-simplex whose points are being calculated with."
    simplex_dimensions::Int

    "The squared distances of points in the simplex, in natural iteration order as initially given, stored flat and triangular for efficiency."
    square_distances::Vector{T}
end

CayleyMengerDistanceMatrix(onlyPoint::Tuple{}) =
    CayleyMengerDistanceMatrix{Union{},0}(0, (Union{})[])

function CayleyMengerDistanceMatrix(
    first_point::NTuple{N,T},
    other_points::Vararg{NTuple{N,T},N},
) where {N,T<:Real}
    points = (first_point, other_points...)

    num_distances = binomial2(N + 1)
    sqr_dists = Vector{T}(undef, num_distances)
    for pointIxA = 1:(N+1)
        for pointIxB = 1:(pointIxA-1)
            pos = index_triangular_nodiag(pointIxA, pointIxB)
            sqr_dists[pos] = sum((points[pointIxA] .- points[pointIxB]) .^ 2)
        end
    end

    CayleyMengerDistanceMatrix{T,N}(N, sqr_dists)
end

function CayleyMengerDistanceMatrix(P::AbstractMatrix{T}) where {T<:Real}
    ps, n = size(P)
    @assert (ps == n + 1) "P should be a tall near-square matrix (N+1 by N)"

    sqr_dists = Vector{T}(undef, binomial2(n + 1))
    for pointIxA = 1:(n+1)
        for pointIxB = 1:(pointIxA-1)
            pos = index_triangular_nodiag(pointIxA, pointIxB)
            sqr_dists[pos] = sum((P[pointIxA, :] .- P[pointIxB, :]) .^ 2)
        end
    end

    CayleyMengerDistanceMatrix{T,Dynamic}(n, sqr_dists)
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

### StaticArrays integration

@inline function StaticArrays.Size(A::CayleyMengerDistanceMatrix{T,Sz}) where {T<:Real,Sz}
    N = Sz isa Dynamic ? Dynamic() : Sz + 2
    N, N
end

### ArrayInterface integration

ArrayInterface.can_change_size(::Type{<:CayleyMengerDistanceMatrix}) = false

ArrayInterface.can_setindex(::Type{<:CayleyMengerDistanceMatrix}) = false

ArrayInterface.has_sparsestruct(::CayleyMengerDistanceMatrix) = true

ArrayInterface.ismutable(::CayleyMengerDistanceMatrix) = false

ArrayInterface.isstructured(::CayleyMengerDistanceMatrix) = true

ArrayInterface.known_first(::CayleyMengerDistanceMatrix{T}) where {T<:Real} = zero(T)

ArrayInterface.known_last(::CayleyMengerDistanceMatrix{T}) where {T<:Real} = zero(T)

ArrayInterface.parent(A::CayleyMengerDistanceMatrix{T}) where {T<:Real} =
    convert.(T, A.square_distances)

function ArrayInterface.findstructralnz(A::CayleyMengerDistanceMatrix{T}) where {T<:Real}
    n = A.simplex_dimensions + 1
    v = n * n - n
    I = 1:n
    J = Vector{Int}(undef, n)
    k = 0
    for i = 1:n
        for j = 1:(i-1)
            k += 1
            J[k] = j
        end
    end
    I, J
end

ArrayInterface.getindex(A::CayleyMengerDistanceMatrix, i::Int, j::Int) = A[i, j]

ArrayInterface.size(A::CayleyMengerDistanceMatrix{T,N}) where {T<:Real,N} =
    (StaticInt(N + 2), StaticInt(N + 2))

ArrayInterface.size(A::CayleyMengerDistanceMatrix{T,Int}) where {T<:Real} =
    (A.simplex_dimensions + 2, A.simplex_dimensions + 2)

# the operation

"""
    simplex_volume(distances::CayleyMengerDistanceMatrix{T,Sz}; distance_type::Union{Nothing,Type{<:Real}} = nothing) where {N,T<:Real}
    simplex_volume(points::Vararg{NTuple{N,T},N+1}; distance_type::Union{Nothing,Type{<:Real}} = nothing) where {N,T<:Real}
    simplex_volume(P::AbstractMatrix{T}; distance_type::Union{Nothing,Type{<:Real}} = nothing) where {T<:Real}

Calculates the (interior) measure of an `N`-simplex given the coordinates of its points, using the Cayley-Menger
determinant involving the squared distances between the points.

For example, if `N == 2`, then this function calculates the area of a triangle given the 2-dimensional coordinates
of its 3 points; likewise, if `N == 3`, then then this function calculates the volume of a tetrahedron given the
3-dimensional coordinates of its 4 points.

When providing `points`, it must be N+1 tuples of N values each, which are the coordinates of the points of the `N`-simplex.
When providing `P`, it must be a tall near-square matrix with N+1 rows and N columns, where the rows are the coordinates of the
points of the `N`-simplex.

If `distance_type` is provided and is not `nothing`, then the internal calculations on the squared distances will be done using
`distance_type` as the type; otherwise, the type used for the internal calculations will be automatically derived from `T`.
"""
function simplex_volume end

function simplex_volume(
    distances::CayleyMengerDistanceMatrix{T};
    distance_type::Union{Nothing,Type{<:Real}} = nothing,
) where {T<:Real}
    Ty = isnothing(distance_type) ? T : distance_type
    n::Int = distances.simplex_dimensions
    n == 0 && return simplex_volume((); distance_type)
    distances_converted = convert(Ty, distances)
    sqrt(abs(det(distances_converted)) / (2^n)) / factorial(n)
end

function simplex_volume(
    points::NTuple{N,T}...;
    distance_type::Union{Nothing,Type{<:Real}} = nothing,
) where {N,T<:Real}
    distances = CayleyMengerDistanceMatrix(points...)
    simplex_volume(distances; distance_type)
end

simplex_volume(::Tuple{}; distance_type::Union{Nothing,Type{<:Real}} = nothing) =
    isnothing(distance_type) ? false : zero(distance_type)

function simplex_volume(
    P::AbstractMatrix{T};
    distance_type::Union{Nothing,Type{<:Real}} = nothing,
) where {T<:Real}
    distances = CayleyMengerDistanceMatrix(P)
    simplex_volume(distances; distance_type)
end

export simplex_volume

end
