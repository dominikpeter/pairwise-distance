import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.metrics import pairwise_distances

from scipy.sparse import csr_matrix, lil_matrix


def batcher(x, batch_size=50):
    check_array(x,  accept_sparse=True)
    len_x = x.shape[0]
    for i in range(0, len_x, batch_size):
        yield x[i:np.min([i + batch_size, len_x]), ]


def check_XY(X, Y=None):
    check_array(X)
    if Y is None:
        Y = X
    else:
        check_array(Y)
    return X, Y


def softmax(x, axis=0):
    """
    softmax function takes an un-normalized vector,
    and normalizes it into a probability distribution
    """
    exp_sum = np.sum(np.exp(x), axis=axis)
    if axis == 1:
        softmax = np.exp(x) / exp_sum[:, np.newaxis]
    else:
        softmax = np.exp(x) / exp_sum
    return softmax


class PairwiseDistance(BaseEstimator):
    def __init__(self,
                 metric="euclidean",
                 n_jobs=None,
                 batch_size=None,
                 return_self=False,
                 **kwds):

        self.metric = metric
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.return_self = return_self

    def _fit_batch(self, X, Y=None):
        X, Y = check_XY(X, Y)
        distances = lil_matrix((X.shape[0], Y.shape[0]), dtype=np.float64)
        for i, batch in enumerate(batcher(X, batch_size=self.batch_size)):
            ifrom, ito = self._batch_index(i, X.shape[0])
            distances[ifrom:ito, :] = pairwise_distances(
                batch, Y, self.metric, self.n_jobs)
        if not self.return_self:
            distances.setdiag(np.inf)
        return distances.tocsr()

    def _batch_index(self, i, n_max):
        ifrom = i * self.batch_size
        ito = np.min([ifrom + self.batch_size, n_max])
        return ifrom, ito

    def _fit(self, X, Y=None):
        X, Y = check_XY(X, Y)
        distances = pairwise_distances(X, Y, self.metric, self.n_jobs)
        if not self.return_self:
            np.fill_diagonal(distances, np.Inf)
        return distances

    def fit(self, X, Y=None):
        if self.batch_size:
            self.distances_ = self._fit(X, Y)
        else:
            self.distances_ = self._fit_batch(X, Y)
        return self

    def _knearest(self, distances, k, return_proba):
        kn_ind = np.argsort(distances)[:, :k]
        # https://stackoverflow.com/questions/33140674/argsort-for-a-multidimensional-ndarray
        kn_dist = distances[np.arange(distances.shape[0])[
            :, np.newaxis], kn_ind]
        return kn_ind, kn_dist

    def _knearest_batch(self, distances, k, return_proba):
        kn_ind = lil_matrix((distances.shape[0], k), dtype=np.int64)
        kn_dist = lil_matrix((distances.shape[0], k), dtype=np.float64)
        for i, batch in enumerate(batcher(distances, self.batch_size)):
            ifrom, ito = self._batch_index(i, distances.shape[0])
            kn_ind[ifrom:ito, :], kn_dist[ifrom:ito, :] = self._knearest(
                batch, k, return_proba)
        return kn_ind.toarray(), kn_dist.toarray()

    def knearest(self, k=5, return_proba=False, return_distance=False):
        check_is_fitted(self, ["distances_"])
        if self.batch_size:
            kn_ind, kn_dist = self._knearest_batch(
                self.distances_, k, return_proba)
        else:
            kn_ind, kn_dist = self._knearest(self.distances_, k, return_proba)
        if return_distance:
            return kn_ind, kn_dist
        return kn_ind
