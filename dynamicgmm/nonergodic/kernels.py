"""
Object-Oriented Kernel Implementation
"""
import math
from typing import Optional, Tuple, Dict
import numpy as np
from scipy.special import gamma, kv
from scipy.linalg import cho_solve, cho_factor


class KernelFunction():
    """
    """
    REQUIRES_PARAMETERS: {}

    def __init__(self, t_1: np.ndarray, t_2: np.ndarray, parameters: Dict,
                 delta: float = 1.0E-9):
        """
        """
        self.t_1 = t_1
        self.t_2 = t_2
        self.n1, self.ndim1 = self.t_1.shape
        self.n2, self.ndim2 = self.t_2.shape
        self.delta = delta
        self.r_tol = np.min([0.01 / np.max([np.abs(self.t_1).max(), np.abs(self.t_2).max()]),
                             1.0E-11])
        self.P = {}
        for param, default in self.REQUIRES_PARAMETERS.items():
            self.P[param] = parameters.get(param, default)

        self._covariance = None

    @property
    def covariance(self):
        """Build the covariance matrix (cache if already built)
        """
        if self._covariance is not None:
            return self._covariance
        # Build the covariance matrix
        self._covariance = np.zeros([self.n1, self.n2])
        for i in range(self.n1):
            dist = np.linalg.norm(self.t_1[i] - self.t_2, axis=1)
            self._covariance[i, :] = self.kernel(dist)
        if self.n1 == self.n2:
            self._covariance += (self.delta * np.eye(self.n1))
        return self._covariance

    def kernel(self, dist: np.ndarray) -> np.ndarray:
        """Returns the kernel function for the given distances
        """
        raise NotImplementedError


class Group(KernelFunction):
    """
    """
    REQUIRES_PARAMETERS = {"omega": 0.0, }

    def kernel(self, dist: np.ndarray) -> np.ndarray:
        """
        """
        return (self.P["omega"] ** 2.0) * (dist < self.r_tol)


class Exponential(KernelFunction):
    """
    """
    REQUIRES_PARAMETERS = {"L": 0.0, "omega": 0.0, "pi": 0.0}

    def kernel(self, dist: np.ndarray) -> np.ndarray:
        """
        """
        return (self.P["pi"] ** 2.) + (self.P["omega"] ** 2.) * np.exp(-dist / self.P["L"])


class SquaredExponential(KernelFunction):
    """
    """
    REQUIRES_PARAMETERS = {"L": 0.0, "omega": 0.0, "pi": 0.0}

    def kernel(self, dist: np.ndarray) -> np.ndarray:
        """
        """
        return (self.P["pi"] ** 2.) + (self.P["omega"] ** 2.) *\
            np.exp(-(dist ** 2.) / (self.P["L"] ** 2.))


class Matern(KernelFunction):
    """
    """
    REQUIRES_PARAMETERS = {"L": 0.0, "omega": 0.0, "pi": 0.0, "nu": 1.5}

    def kernel(self, dist: np.ndarray) -> np.ndarray:
        """
        """
        if self.P["nu"] == 0.5:
            K = np.exp(-dist / self.P["L"])
        elif self.P["nu"] == 1.5:
            K = (dist / self.P["L"]) * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.P["nu"] == 2.5:
            K = (dist / self.P["L"]) * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.P["nu"] == np.inf:
            K = np.exp(-((dist / self.P["L"]) ** 2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dist / self.P["L"]
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.P["nu"]) * K
            K.fill((2 ** (1.0 - self.P["nu"])) / gamma(self.P["nu"]))
            K *= tmp**self.P["nu"]
            K *= kv(self.P["nu"], tmp)
        return (self.P["pi"] ** 2.0) + (self.P["omega"] ** 2.) * K


class NegativeExponentialSpatiallyIndependent(KernelFunction):
    """
    """
    REQUIRES_PARAMETERS = {"L": 0.0, "omega1": 0.0, "omega2": 0.0, "pi": 0.0}

    def kernel(self, dist: np.ndarray) -> np.ndarray:
        """
        """
        # Exponential kernel
        kern_exp = (self.P["pi"] ** 2.0) +\
            (self.P["omega1"] ** 2.0) * np.exp(-dist / self.P["L"])
        kern_grp = (self.P["omega2"] ** 2.0) * (dist < self.r_tol)
        return kern_exp + kern_grp


KERNELS = {
    "Group": Group,
    "Exponential": Exponential,
    "SquaredExponential": SquaredExponential,
    "Matern": Matern,
    "NegativeExponentialSpatiallyIndependent": NegativeExponentialSpatiallyIndependent
}


def predict_nonergodic_coefficients_fast(
        g_prdct: np.ndarray,
        g_train: np.ndarray,
        c_train_mu: np.ndarray,
        kernel: str,
        parameters: Dict,
        c_train_sig: Optional[np.ndarray] = None,
        hyp_mean_c: float = 0.0,
        hyp_omega: float = 0.0,
        delta: float = 1.0E-9,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    assert kernel in list(KERNELS)
    c_train_mu = c_train_mu - hyp_mean_c
    if c_train_sig is None:
        c_train_sig = np.zeros(len(c_train_mu))
    c_train_cov = np.diag(c_train_sig ** 2.0) if c_train_sig.ndim == 1 else c_train_sig
    # Generate the covariance matrices
    # i) within the training data
    kupper = KERNELS[kernel](g_train, g_train, parameters, delta).covariance
    # ii) between training data and new locations
    klower = KERNELS[kernel](g_prdct, g_train, parameters, 0.0).covariance
    # iii) within new locations
    kstar = KERNELS[kernel](g_prdct, g_prdct, parameters, 0.0).covariance

    #kupper_inv = np.linalg.inv(kupper)
    #klowupp_inv = klower.dot(kupper_inv)
    #klowupp_inv = solve(kupper, klower.T, assume_a="pos").T
    klowupp_inv = cho_solve(cho_factor(kupper), klower.T).T
    # Posterior mean and variance at new locations
    c_prdct_mu = klowupp_inv.dot(c_train_mu)
    c_prdct_cov = kstar - klowupp_inv.dot(klower.transpose()) +\
        klowupp_inv.dot(c_train_cov.dot(klowupp_inv.transpose()))
    # Posterior standard deviation at new locations
    c_prdct_sig = np.sqrt(np.diag(c_prdct_cov))
    # Add mean from training coefficients
    c_prdct_mu += hyp_mean_c
    return c_prdct_mu, c_prdct_sig, c_prdct_cov


def predict_nonergodic_coefficients(
        g_prdct: np.ndarray,
        g_train: np.ndarray,
        c_train_mu: np.ndarray,
        kernel: str,
        parameters: Dict,
        c_train_sig: Optional[np.ndarray] = None,
        hyp_mean_c: float = 0.0,
        hyp_omega: float = 0.0,
        delta: float = 1.0E-9,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    assert kernel in list(KERNELS)
    c_train_mu = c_train_mu - hyp_mean_c
    if c_train_sig is None:
        c_train_sig = np.zeros(len(c_train_mu))
    c_train_cov = np.diag(c_train_sig ** 2.0) if c_train_sig.ndim == 1 else c_train_sig
    # Generate the covariance matrices
    # i) within the training data
    kupper = KERNELS[kernel](g_train, g_train, parameters, delta).covariance
    # ii) between training data and new locations
    klower = KERNELS[kernel](g_prdct, g_train, parameters, 0.0).covariance
    # iii) within new locations
    kstar = KERNELS[kernel](g_prdct, g_prdct, parameters, 0.0).covariance

    kupper_inv = np.linalg.inv(kupper)
    klowupp_inv = klower.dot(kupper_inv)

    # Posterior mean and variance at new locations
    c_prdct_mu = klowupp_inv.dot(c_train_mu)
    c_prdct_cov = kstar - klowupp_inv.dot(klower.transpose()) +\
        klowupp_inv.dot(c_train_cov.dot(klowupp_inv.transpose()))
    # Posterior standard deviation at new locations
    c_prdct_sig = np.sqrt(np.diag(c_prdct_cov))
    # Add mean from training coefficients
    c_prdct_mu += hyp_mean_c
    return c_prdct_mu, c_prdct_sig, c_prdct_cov
