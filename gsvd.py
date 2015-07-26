#!/usr/bin/env python

import numpy as np
import scipy.linalg

def gsvd(a, m, w):
	"""
	:param a: Matrix to GSVD
	:param m: 1st Constraint, (u.T * m * u) = I
	:param w: 2nd Constraint, (v.T * w * v) = I
	:return: (u ,s, v)
	"""

	(aHeight, aWidth) = a.shape
	(mHeight, mWidth) = m.shape
	(wHeight, mWidth) = w.shape

	assert(aHeight == mHeight)
	assert(aWidth == mWidth)

	mSqrt = scipy.linalg.sqrtm(m)
	wSqrt = scipy.linalg.sqrtm(w)


	mSqrtInv = np.linalg.inv(mSqrt)
	wSqrtInv = np.linalg.inv(wSqrt)

	_a = np.dot(np.dot(mSqrt, a), wSqrt)

	(_u, _s, _v) = np.linalg.svd(_a)

	u = np.dot(mSqrtInv, _u)
	v = np.dot(wSqrtInv, _v.T).T
	s = _s

	return (u, s, v)
