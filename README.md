# GSVD
Generalized singular value decomposition

The following is referenced from Wikipedia

The weighted version of the generalized singular value decomposition (GSVD)
is a constrained matrix decomposition with constraints imposed on the left
and right singular vectors of the singular value decomposition.This form of
the GSVD is an extension of the SVD as such. 

Given the SVD of an m×n real or complex matrix M

M = U * S * V
where U.T * W * U = V.T * Y * V = I.

I is the Identity Matrix and where U and V are orthonormal given their constraints.
Additionally, W and Y are positive definite matrices (often diagonal matrices of weights). 
