from random import uniform

import numpy as np
import scipy.stats as ss
from scipy.special import logsumexp


class GMM:

    """
    Gaussian Mixture Model by Expectation-Maximization algorithm.

    This model was designed to model the univariate data, but the mode of
    variances can be controlled with fine granularity.
    """

    def __init__(
        self, n_components, sigma_type="full", sigmas=None, tol=1e-3,
        max_iter=100, init_params=None, gamma_type="equal"
    ):
        """
        meaningful properties:
            mus_: the estimated means of gaussian components.
            sigmas_: the estimated standard deviations of gaussian components.
            gammas_: the estimated probabilities of categorical distribution.
            labels_: the clusters to which each sample belongs.
        Parameters:
            n_components: the number of gaussian components.
            sigma_type: the type of mode of variance. "full" means different
                variances for every component; "equal" means the same variance
                for every component; "custom" means variances of the componets
                will not be estimated, but assigned by sigmas.
            sigmas: the assigned standard deviations, will be useful when
                "sigma_type = 'custom'".
            tol: the tolerance of stopping iteration.
            max_iter: the maximum iterations.
            init_params: if is None, paramters will be randomly initialized.
            gamma_type: the type of categorical distribution. "equal" means the
                probabilities of all categories will be the same. "full" means
                the probabilities will be estimated.
        """
        assert sigma_type in ["full", "equal", "custom"]
        assert gamma_type in ["full", "equal"]
        if sigma_type == "custom":
            assert isinstance(sigmas, np.ndarray) and sigmas.ndim == 1 and \
                sigmas.shape[0] == n_components
        self.n_components = n_components
        self.sigma_type = sigma_type
        self.sigmas = sigmas
        self.tol = tol
        self.max_iter = max_iter
        self.init_params = init_params
        self.gamma_type = gamma_type

    def fit_predict(self, X):
        """
        Parameters:
            X: the samples, shape is (N,)
        Return:
            labels: the clusters to which each sample belongs
        """
        self.status = "fail"
        elbo = -np.infty
        gammas_, mus_, sigmas_ = self._init(X.shape[0])

        for i in range(self.max_iter):
            prev_elbo = elbo
            logp_norm, elbo = self._e_step(X, gammas_, mus_, sigmas_)
            gammas_, mus_, sigmas_ = self._m_step(X, logp_norm)
            if abs(prev_elbo - elbo) < self.tol:
                self.status = "success"
                break
        self.mus_, self.sigmas_, self.gammas_ = mus_, sigmas_, gammas_
        self.labels_ = logp_norm.argmax(axis=1)
        return self.labels_

    def _e_step(self, X, gammas_, mus_, sigmas_):
        """
        The Expectation step.
        Compute the posterior distribution and expectation of joint
            distribution.
        """
        logp = ss.norm.logpdf(X[:, None], mus_, sigmas_)  # p(x|z)
        if self.gamma_type == "full":
            logp = logp + np.log(gammas_)
        logpsum = logsumexp(logp, axis=1)
        logp_norm = logp - logpsum[:, None]               # p(z|x)
        return logp_norm, (logp * np.exp(logp_norm)).sum()

    def _m_step(self, X, logp):
        """
        The Maximization step.
        Maximize the expectation of joint distribution and compute the new
            parameters.
        """
        # 1. prepare
        p = np.exp(logp)
        # get rid of having NaN
        psum = p.sum(axis=0) + 10 * np.finfo(p.dtype).eps
        # 2. compute the parameters of categorical distribution
        if self.gamma_type == "full":
            gammas_ = psum / psum.sum()
        else:
            gammas_ = np.ones(self.n_components) / self.n_components
        # 3. compute the mean of gaussian components
        mus_ = (p * X[:, None]).sum(axis=0) / psum
        # 4. compute the stdandard deviations of gaussian components
        if self.sigma_type == "full":
            sigmas2_ = ((X[:, None] - mus_) ** 2 * p).sum(axis=0) / psum
            sigmas_ = np.sqrt(sigmas2_)
        elif self.sigma_type == "equal":
            sigmas2_ = ((X[:, None] - mus_) ** 2 * p).sum() / psum.sum()
            sigmas1_ = np.sqrt(sigmas2_)
            sigmas_ = np.full(self.n_components, sigmas1_)
        else:
            sigmas_ = self.sigmas
        return gammas_, mus_, sigmas_

    def _init(self, nsamples):
        """ Initializing """
        if self.init_params is not None:
            return self.init_params
        if self.gamma_type == "full":
            gammas_ = np.random.rand(self.n_components)
            gammas_ = gammas_ / gammas_.sum()
        else:
            gammas_ = np.ones(self.n_components) / self.n_components
        # don't set the initialized std close to zero.
        if self.sigma_type == "full":
            return (gammas_, np.random.randn(self.n_components),
                    np.random.uniform(0.25, 0.75, size=(self.n_components,)))
        elif self.sigma_type == "equal":
            return (gammas_, np.random.randn(self.n_components),
                    np.full(self.n_components, uniform(0.25, 0.75)))
        else:
            return (gammas_, np.random.randn(self.n_components), self.sigmas)
