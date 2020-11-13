import numpy as np
from .kernels import kernel


class model(object):
    """
    Kernel model class
    Describes the non-parameteric kernel model to use for density estimation
    Assumptions:
        each kernel has one bandwidth
        each kernel has one or more dimensions
        dimensions are not shared between kernels
        bandwidths are not shared between kernels
        there are as many or fewer bandwidths than there are dimensions
    """

    def __init__(self, model_kernels):
        """
        Initialize information about the model
        """
        if not np.all([isinstance(k, kernel) for k in model_kernels]):
            raise ValueError("model_kernels must all be instances of kernels.kernel")
        self._kernels = model_kernels
        self._data_dimensions = [k.data_dimensions() for k in self._kernels]
        self._bandwidth_dimensions = [k.bandwidth_dimensions() for k in self._kernels]
        self._kernel_data_indices = []
        self._kernel_bandwidth_indices = []
        self._kernel_index_by_bandwidth_index = []
        i = 0
        j = 0
        self._n_dimensions = 0
        self._n_bandwidths = 0
        for ki, (dd, bd) in enumerate(
            zip(self._data_dimensions, self._bandwidth_dimensions)
        ):
            self._kernel_data_indices.append(tuple(np.arange(i, i + dd)))
            self._kernel_bandwidth_indices.append(tuple(np.arange(j, j + bd)))
            self._kernel_index_by_bandwidth_index.extend([ki] * bd)
            self._n_dimensions += dd
            self._n_bandwidths += bd
            i += dd
            j += bd

    def evaluate(self, point, data, bandwidth, index):
        """
        Evaluate the KDE components at a particular point
        index is kernel index to evaluate
        """
        j = self._kernel_index_by_bandwidth_index[index]
        # f = self._kernels[j].evaluate(data[self._kernel_data_indices[j],:], point[(self._kernel_data_indices[j],)], bandwidth[index])
        f = self._kernels[j].evaluate(
            point[(self._kernel_data_indices[j],)],
            data[self._kernel_data_indices[j], :],
            bandwidth[index],
        )
        return f

    def evaluate_bandwidth_derivative(self, point, data, bandwidth, derivative):
        """
        Evaluate the derivative of the KDE components with respect to a particular bandwith
        derivative is the bandwidth index
        """
        j = self._kernel_index_by_bandwidth_index[derivative]
        # Zj = self._kernels[j].evaluate_bandwidth_derivative(data[self._kernel_data_indices[j],:], point[(self._kernel_data_indices[j],)], bandwidth[derivative])
        Zj = self._kernels[j].evaluate_bandwidth_derivative(
            point[(self._kernel_data_indices[j],)],
            data[self._kernel_data_indices[j], :],
            bandwidth[derivative],
        )
        return Zj
