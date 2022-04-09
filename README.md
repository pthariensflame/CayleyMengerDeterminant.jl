# CayleyMengerDeterminant

Calculate the Cayley-Menger squared-distance matrix of an `N`-simplex from its points, and use the matrix’s determinant to compute the (interior) measure of the simplex—for example, the area of a triangle, or the volume of a tetrahedron.

This package exports two small utility functions, `binomial2` and `inverse_binomial2`, as well as the full-featured immutable structured matrix type `CayleyMengerDistanceMatrix` and its constructors, and the function `simplex_volume` for calculating the measure directly from the points.
