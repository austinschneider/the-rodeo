import numpy as np
import scipy.special
from abc import ABCMeta

from functools import reduce


class abstractstatic(staticmethod):
    __slots__ = ()

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


class kernel(metaclass=ABCMeta):
    """
    Abstract kernel class (probably not used properly because I don't know how to write OO python code)
    The methods below are assumed to exist for any kernel class
    """

    @abstractstatic
    def data_dimensions(self):
        pass

    @abstractstatic
    def bandwidth_dimensions(self):
        pass

    @abstractstatic
    def evaluate(self, x, mu, bandwidth):
        pass

    @abstractstatic
    def evaluate_bandwidth_derivative(self, x, mu, bandwidth):
        pass


class gaussian_kernel(object):
    """
    A one dimensional Gaussian kernal
    """

    def data_dimensions(self):
        return 1

    def bandwidth_dimensions(self):
        return 1

    def evaluate(self, x, mu, bandwidth):
        x = np.asarray(x)
        sigma = bandwidth
        xmu = x - mu
        exponent = -(xmu ** 2.0) / (2.0 * sigma ** 2.0)
        return np.exp(np.asarray(exponent).astype(float)) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
        # return np.exp(np.asarray(exponent).astype(float))

    def evaluate_bandwidth_derivative(self, x, mu, bandwidth):
        x = np.asarray(x)
        sigma = bandwidth
        sigma2 = sigma ** 2.0
        sigma3 = sigma2 * sigma
        sigma4 = sigma2 ** 2.0
        xmu = x - mu
        xmu2 = xmu ** 2.0
        return (
            np.exp(np.array(-(xmu2) / (2.0 * sigma2)).astype(float))
            * (xmu2 - sigma2)
            / (sigma4 * np.sqrt(2.0 * np.pi))
        )
        # return np.exp(np.array(-(xmu2)/(2.0*sigma2)).astype(float)) * (xmu2 - sigma2) / sigma3


kernel.register(gaussian_kernel)


class epanechnikov_kernel(object):
    """
    A one dimensional Epanechnikov kernal
    """

    def __init__(self, bounds):
        self.bounds = bounds

    def data_dimensions(self):
        return 1

    def bandwidth_dimensions(self):
        return 1

    def norm(self, mu, sigma):
        lower_bound = self.bounds[0]
        upper_bound = self.bounds[1]
        a, b = lower_bound, upper_bound
        if a is None:
            b0 = False
        else:
            b0 = lower_bound > mu - sigma
        if b is None:
            b1 = False
        else:
            b1 = upper_bound < mu + sigma
        mask_a = reduce(np.logical_and, [b0, ~b1])
        mask_b = reduce(np.logical_and, [~b0, b1])
        mask_both = reduce(np.logical_and, [b0, b1])
        mask_none = reduce(np.logical_and, [~b0, ~b1])
        norm = np.zeros(np.shape(mu))
        norm[mask_none] = sigma
        if np.any(mask_both):
            norm[mask_both] = (
                (a - b)
                * (
                    a ** 2
                    + b ** 2
                    + a * b
                    - 3 * (a + b) * mu[mask_both]
                    + 3 * mu[mask_both] ** 2
                    - 3 * sigma ** 2
                )
                / (4 * sigma ** 2)
            )
        if np.any(mask_a):
            norm[mask_a] = (
                (mu[mask_a] - a + sigma) ** 2
                * (a - mu[mask_a] + 2 * sigma)
                / (4 * sigma ** 2)
            )
        if np.any(mask_b):
            norm[mask_b] = (
                (2 * sigma + mu[mask_b] - b)
                * (b - mu[mask_b] + sigma) ** 2
                / (4 * sigma ** 2)
            )
        return norm

    def evaluate(self, x, mu, bandwidth):
        # x = np.asarray(x, dtype=float)
        # mu = np.asarray(mu, dtype=float)
        sigma = bandwidth
        xmu = x - mu
        z = xmu / sigma
        result = np.zeros(z.shape)
        mask = abs(z) <= 1
        norm = self.norm(mu[mask], sigma)
        result[mask] = 3.0 / 4.0 * (1.0 - z[mask] ** 2.0) / norm
        return result

    def norm_d_both(self, x, mu, sigma):
        lower_bound = self.bounds[0]
        upper_bound = self.bounds[1]
        a, b = lower_bound, upper_bound
        return (
            -(a - b)
            * (a ** 2 + a * b + b ** 2 - 3.0 * (a + b) * mu + 3.0 * mu ** 2)
            / (2.0 * sigma ** 2)
        )

    def norm_d_a(self, x, mu, sigma):
        lower_bound = self.bounds[0]
        upper_bound = self.bounds[1]
        a, b = lower_bound, upper_bound
        return (1.0 - ((a - mu) / sigma) ** 3) / 2.0

    def norm_d_b(self, x, mu, sigma):
        lower_bound = self.bounds[0]
        upper_bound = self.bounds[1]
        a, b = lower_bound, upper_bound
        return (((b - mu) / sigma) ** 3 + 1.0) / 2.0

    def norm_d_none(self, x, mu, sigma):
        return 1.0

    def evaluate_bandwidth_derivative(self, x, mu, bandwidth):
        x = np.array(x)
        mu = np.array(mu)
        sigma = bandwidth
        xmu = x - mu
        z = xmu / sigma
        result = np.zeros(z.shape)
        mask = abs(z) <= 1

        if mu.shape == z.shape:
            reduced_mu = mu[mask]
        else:
            reduced_mu = mu

        norm = self.norm(reduced_mu, sigma)

        reduced_shape = result[mask].shape

        # main = np.zeros(reduced_shape)
        main = 3.0 / 4.0 * (1.0 - z[mask] ** 2.0)

        # main_d = np.zeros(reduced_shape)
        main_d = -3.0 / 2.0 * xmu[mask] / sigma ** 2
        if self.bounds is None:
            norm_d = self.norm_d_none(x, reduced_mu, sigma)
            result[mask] = (main_d - norm_d * main / norm) / norm
        else:
            lower_bound = self.bounds[0]
            upper_bound = self.bounds[1]
            a, b = lower_bound, upper_bound
            if a is None:
                b0 = False
            else:
                b0 = lower_bound > reduced_mu - sigma
            if b is None:
                b1 = False
            else:
                b1 = upper_bound < reduced_mu + sigma
            mask_a = reduce(np.logical_and, [b0, ~b1])
            mask_b = reduce(np.logical_and, [~b0, b1])
            mask_both = reduce(np.logical_and, [b0, b1])
            mask_none = reduce(np.logical_and, [~b0, ~b1])
            norm_d = np.zeros(reduced_shape)
            if np.any(mask_none):
                norm_d[mask_none] = 1.0
            if np.any(mask_both):
                norm_d[mask_both] = self.norm_d_both(x, reduced_mu[mask_both], sigma)
            if np.any(mask_a):
                norm_d[mask_a] = self.norm_d_a(x, reduced_mu[mask_a], sigma)
            if np.any(mask_b):
                norm_d[mask_b] = self.norm_d_b(x, reduced_mu[mask_b], sigma)
            result[mask] = (main_d - norm_d * main / norm) / norm

        return result


kernel.register(epanechnikov_kernel)


class vonMises_kernel(object):
    """
    A vonMises kernel to describe pdfs on a circle
    """

    def data_dimensions(self):
        return 1

    def bandwidth_dimensions(self):
        return 1

    def evaluate(self, x, mu, bandwidth):
        x = np.asarray(x)
        k = 1.0 / bandwidth ** 2.0
        i0 = scipy.special.i0(k)
        cos = np.cos(x - mu)
        return np.exp(k * cos) / (2.0 * np.pi * i0)
        # return np.exp(k*cos) / 2.0

    def evaluate_bandwidth_derivative(self, x, mu, bandwidth):
        x = np.asarray(x)
        k = 1.0 / bandwidth ** 2.0
        i0 = scipy.special.i0(k)
        i1 = scipy.special.i1(k)
        cos = np.cos(x - mu)
        return (
            np.exp(k * cos) * k ** (3.0 / 2.0) * (i1 - i0 * cos) / (np.pi * i0 ** 2.0)
        )
        # return np.exp(k*cos) * k**(3.0/2.0) * (i1 - i0*cos) / i0


kernel.register(vonMises_kernel)


class vonMisesFisher_kernel(object):
    """
    A vonMisesFisher kernel to describe pdfs on a sphere
    """

    def data_dimensions(self):
        return 3

    def bandwidth_dimensions(self):
        return 1

    def evaluate(self, x, mu, bandwidth):
        x = np.asarray(x)
        mu = np.asarray(mu)
        shape = x.shape
        split_dim = None
        if 3 in shape:
            split_dim = shape.index(3)
        else:
            raise ValueError(
                "x must contain a dimension of length 2 to represent the two coordinates!"
            )

        x0_slice = [slice(None) for i in range(len(shape))]
        x0_slice[split_dim] = 0
        x1_slice = [slice(None) for i in range(len(shape))]
        x1_slice[split_dim] = 1
        x2_slice = [slice(None) for i in range(len(shape))]
        x2_slice[split_dim] = 2

        x0 = x[tuple(x0_slice)]
        x1 = x[tuple(x1_slice)]
        x2 = x[tuple(x2_slice)]

        shape = mu.shape
        split_dim = None
        if 3 in shape:
            split_dim = shape.index(3)
        else:
            raise ValueError(
                "x must contain a dimension of length 3 to represent the two coordinates!"
            )

        mu0_slice = [slice(None) for i in range(len(shape))]
        mu0_slice[split_dim] = 0
        mu1_slice = [slice(None) for i in range(len(shape))]
        mu1_slice[split_dim] = 1
        mu2_slice = [slice(None) for i in range(len(shape))]
        mu2_slice[split_dim] = 2

        mu0 = mu[tuple(mu0_slice)]
        mu1 = mu[tuple(mu1_slice)]
        mu2 = mu[tuple(mu2_slice)]

        k = 1.0 / bandwidth ** 2.0

        cos = x0 * mu0 + x1 * mu1 + x2 * mu2

        return np.exp(k * cos) * k / (np.sinh(k) * (4.0 * np.pi))
        # return np.exp(k*cos)

    def evaluate_bandwidth_derivative(self, x, mu, bandwidth):
        x = np.asarray(x)
        shape = x.shape
        split_dim = None
        if 3 in shape:
            split_dim = shape.index(3)
        else:
            raise ValueError(
                "x must contain a dimension of length 3 to represent the two coordinates!"
            )

        x0_slice = [slice(None) for i in range(len(shape))]
        x0_slice[split_dim] = 0
        x1_slice = [slice(None) for i in range(len(shape))]
        x1_slice[split_dim] = 1
        x2_slice = [slice(None) for i in range(len(shape))]
        x2_slice[split_dim] = 2

        x0 = x[tuple(x0_slice)]
        x1 = x[tuple(x1_slice)]
        x2 = x[tuple(x2_slice)]

        shape = mu.shape
        split_dim = None
        if 3 in shape:
            split_dim = shape.index(3)
        else:
            raise ValueError(
                "x must contain a dimension of length 3 to represent the two coordinates!"
            )

        mu0_slice = [slice(None) for i in range(len(shape))]
        mu0_slice[split_dim] = 0
        mu1_slice = [slice(None) for i in range(len(shape))]
        mu1_slice[split_dim] = 1
        mu2_slice = [slice(None) for i in range(len(shape))]
        mu2_slice[split_dim] = 2

        mu0 = mu[tuple(mu0_slice)]
        mu1 = mu[tuple(mu1_slice)]
        mu2 = mu[tuple(mu2_slice)]

        k = 1.0 / bandwidth ** 2.0

        cos = x0 * mu0 + x1 * mu1 + x2 * mu2

        return (
            -np.exp(k * cos)
            * k ** (3.0 / 2.0)
            * (1.0 + k * (cos - 1.0 / np.tanh(k)))
            / (np.sinh(k) * (2.0 * np.pi))
        )
        # return -np.exp(k*cos) * k**0.5 * (1.0 + k*(cos - 1.0/np.tanh(k)))


kernel.register(vonMisesFisher_kernel)
