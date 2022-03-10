module CayleyMengerDeterminant
using LinearAlgebra

struct CayleyMengerDistanceMatrix{T, V <: AbstractVector{T}}
    distances::V
end

end
