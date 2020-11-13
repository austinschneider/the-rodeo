# -*- coding: utf-8 -*-
import numpy as np
from context import rodeo
import unittest


class TrySomething(unittest.TestCase):
    """Basic test cases."""

    def test_this(self):
        N = int(1e5)

        data = np.array([np.random.normal(loc=0.0, scale=1.0, size=N)])

        mod_kernels = [rodeo.gaussian_kernel()]

        point = np.array([0])

        mod = rodeo.model(mod_kernels)

        beta = 0.9
        r = rodeo.the_rodeo(mod, data, weights=np.ones(N) / N)

        h = np.array([1.0])

        f = mod.evaluate(point, data, h, 0)
        print(np.sum(np.prod(f, axis=0)) / N)

        bandwidth, f = r.local_rodeo(point, beta)

        print("bandwidth =", bandwidth)
        print("f =", f)


if __name__ == "__main__":
    unittest.main()
