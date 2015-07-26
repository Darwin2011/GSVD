#!/usr/bin/env python

import gsvd
import random
import numpy as np
import unittest

class GSVDTest(unittest.TestCase):

	def test_gsvd(self):
		for i in range(100):
			height = random.randint(10, 20)
			width = random.randint(10, 20)
			a = (np.random.rand(height, width) - 0.5) * 255
			#m = np.diag(255 * (np.random.rand(height) - 0.5))
			#w = np.diag(255 * (np.random.rand(width) - 0.5))
			m = (np.random.rand(height, height) - 0.5) * 255
			m_symmetric = m + m.T
			w = (np.random.rand(width, width) - 0.5) * 255
			w_symmetric = w + w.T
			(u, s, v) = gsvd.gsvd(a, m_symmetric, w_symmetric)
			S = np.zeros((height, width))
			for j in range(min(height, width)):
				S[j][j] = s[j]
			gsvdResult = np.dot(np.dot(u, S), v)
			np.testing.assert_array_almost_equal(a, gsvdResult)

if __name__ == "__main__":
	unittest.main()
