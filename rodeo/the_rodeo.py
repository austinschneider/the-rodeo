import numpy as np

def accumulate(inp):
    """ Performs Kahan summation """
    summ = c = 0
    for num in inp:
        y = num - c
        t = summ + y
        c = (t - summ) - y
        summ = t
    return summ

class the_rodeo:
    """
        http://www.cs.cmu.edu/~hanliu/papers/jsm07.pdf
    """
    def __init__(self, stat_model, data, weights=None):
        self._stat_model = stat_model
        self._data = data
        self._weights = weights if weights is not None else np.ones(data.shape[1])

    def get_TS(self, point, h):
        n = self._data.shape[1]
        d = self._stat_model._n_bandwidths

        f = np.ones((d, n))
        for bi in range(d):
            f[bi,:] = self._stat_model.evaluate(point, self._data, h, bi)

        Z = 0

        for bi in range(d):
            # Evaluate the derivative w.r.t the current bandwidth dimension
            Zd = self._stat_model.evaluate_bandwidth_derivative(point, self._data, h, bi).flatten()

            # Other components are constant w.r.t the current bandwidth dimension
            mask = np.ones(d).astype(bool)
            mask[bi] = False
            Znd = np.prod(f[mask,:], axis=0).flatten()
            Zs = Znd * Zd

            # Compute the variance of the test statistic Z
            Z += abs(accumulate((Zs[vi]*self._weights[vi] for vi in range(n))))
        return Z

    def local_rodeo(self, point, beta, h0=None):
        assert(beta > 0)
        #assert(beta < 1)
        n = self._data.shape[1]
        # d corresponds to te number of dimensions $d$ in the paper
        d = self._stat_model._n_bandwidths

        # Precomputed information about the weights
        # n_eff replaces n which is defined in the paper in Eq(2)
        v1 = np.sum(self._weights)
        v2 = np.sum(self._weights*self._weights)
        n_eff = v1**2 / v2
        w_eff = v2 / v1

        need_distances = (h0 is None) or (np.any([x is None for x in h0]))

        hh0 = None
        if need_distances:
            avg_diffs = []
            max_range = []
            # Compute mean distance to point
            for j in range(len(self._data)):
                sorted_indices = np.argsort(self._data[j])
                sorted_weights = self._weights[sorted_indices]
                sorted_vals = self._data[j][sorted_indices]
                val_sum = accumulate((abs(sorted_vals[vi] - point[j])*sorted_weights[vi] for vi in range(n)))

                vv1 = accumulate((sorted_weights[vi] for vi in range(n)))
                vv2 = accumulate(((sorted_weights[vi])**2 for vi in range(n)))

                avg_diff = val_sum / vv1
                s_sum = accumulate((sorted_weights[vi]*(abs(sorted_vals[vi]-point[j])-avg_diff)**2 for vi in range(n)))

                s = np.sqrt(s_sum/(vv1-(vv2/vv1)))
                avg_diffs.append(avg_diff + s)
                max_range.append(np.amax(self._data[j]) - np.amin(self._data[j]))


            c0 = np.ones(d)
            c0_alt = np.ones(d)

            i = 0
            # Initialize the starting scale to be the maximum scale across dimensions
            for j, dd in enumerate(self._stat_model._data_dimensions):
                c0[j] = np.amax(avg_diffs[i:i+dd])
                c0_alt[j] = np.amax(max_range[i:i+dd])
                i += dd
            #c0 = np.amax(c0) * np.ones(d)

            # Make sure our starting scale is large
            if beta < 1:
                c0 = np.amin([c0*100, c0_alt*100], axis=0)
            else:
                c0 = np.amin([c0/10, c0_alt/10], axis=0)

            cn = c0 * np.e * np.log(n_eff) / (n_eff)
            cn = np.ones(d) * cn
            hh0 = c0 / np.log(np.log(n_eff))
        if h0 is None:
            h0 = hh0
        elif np.any([x is None for x in h0]):
            h0 = np.array([h if h is not None else hh for h,hh in zip(h0,hh0)])
        else:
            pass

        # c0 corresponds to $c_0$ defined between Eq(10) and Eq(11) in the paper
        # within Figure(1)
        c0 = h0 * np.log(np.log(n_eff))
        # cn corresponds to $c_n$ defined in Figure(1) of the paper
        cn = c0 * np.e * np.log(n_eff) / (n_eff)
        h = np.ones(d)*h0


        # The set of bandwidths that we are currently working with
        # A corresponds to $\mathcal{A} in Figure(1)$
        A = list(range(d))

        # Store the KDE components
        # f corresponds to the components of $\hat{f_H}(x)$ before the sum and product is taken
        f = np.ones((d, n))
        for bi in A:
            f[bi,:] = self._stat_model.evaluate(point, self._data, h, bi)

        # Keep going while we have dimensions to work with
        while len(A) > 0:
            print(h)
            for bi in A:
                # Evaluate the derivative w.r.t the current bandwidth dimension
                # Zd corresponds to the different derivatives $Z_j$, but split into their components before the product
                Zd = self._stat_model.evaluate_bandwidth_derivative(point, self._data, h, bi).flatten()

                # Other components are constant w.r.t the current bandwidth dimension
                mask = np.ones(d).astype(bool)
                mask[bi] = False
                # Znd corresponds to the different components of $\hat{f_H}(x)$ before the sum. The product over the dimensions has been taken
                Znd = np.prod(f[mask,:], axis=0).flatten()
                Zs = Znd * Zd

                # Compute the variance of the test statistic Z
                Z = accumulate((Zs[vi]*self._weights[vi] for vi in range(n)))
                mu_star = Z / v1
                s = np.sqrt(accumulate((self._weights[vi]*((Zs[vi]-mu_star)**2.0) for vi in range(n))) * v2/(v1**3-(v2*v1)))

                # Compute the threshold that we compare to
                lam = s*np.sqrt(2.0*np.log(n_eff*cn[bi]))

                if np.abs(Z) > lam:
                    # Next iteration / update the KDE components
                    h[bi] = beta * h[bi]
                    f[bi,:] = self._stat_model.evaluate(point, self._data, h, bi)
                else:
                    # Done with this bandwidth dimension
                    del A[A.index(bi)]

        # Finally evaluate the whole thing
        for bi in A:
            f[bi,:] = self._stat_model.evaluate(point, self._data, h, bi)

        f = np.sum(np.prod(f, axis=0)*self._weights)
        print('Number of events contributing:', np.sum(np.prod(f, axis=0)))

        return h, f

