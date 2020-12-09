import os
from multiprocessing import Pool
from itertools import product

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt

from gmm import GMM


def generate_data(mus=[0, 1], sigmas=[1, 1], nsamples=10):
    """ generate two group data. """
    X, Y = [], []
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        X.append(np.random.normal(mu, sigma, size=(nsamples,)))
        Y.append(np.full(nsamples, i))
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y


def gaussian_mixture(
    X, typ="full", sigma=None, init_params=None, gamma_type="equal"
):
    """ fit a GMM model and give the prdicted labels, estimated paramters """
    if typ in ["sk_kmeans", "sk_random"]:
        model = GaussianMixture(
            2, covariance_type="diag", max_iter=1000,
            init_params="random" if typ == "sk_random" else "kmeans"
        )
        pred = model.fit_predict(X.reshape(-1, 1))
        return pred, model.means_.squeeze(), model.covariances_.squeeze()
    model = GMM(2, sigma_type=typ, sigmas=sigma, max_iter=1000,
                init_params=init_params, gamma_type=gamma_type)
    pred = model.fit_predict(X)
    return pred, model.mus_, model.sigmas_, model.gammas_


def score(
    true_mus, true_sigmas, true_y, pred_y, pred_mus, pred_sigmas, pred_gammas
):
    """ calculate the metrics """
    # adjusted rand score
    adj_rand = adjusted_rand_score(true_y, pred_y)

    # calibration of predicted and true labels
    main_diag = sum(true_y == pred_y)
    mino_diag = true_y.shape[0] - main_diag
    if mino_diag > main_diag:
        pred_y = 1 - pred_y
        pred_mus = pred_mus[[1, 0]]
        pred_sigmas = pred_sigmas[[1, 0]]
    # calculate the difference of predicted and true means
    mu_diff = true_mus - pred_mus
    # calculate the difference of predicted and true standard deviations
    sigma_diff = true_sigmas - pred_sigmas
    # calculate the difference of predicted and true mean differences
    delta_diff = true_mus[1] - true_mus[0] - (pred_mus[1] - pred_mus[0])
    # calculate the difference of predicted and true gammas
    gamma_diff = pred_gammas - 1 / 2

    return np.r_[adj_rand, delta_diff, mu_diff, sigma_diff, gamma_diff]


def whole_process(
    nsamples, delta, typ="full", sigma=None, init_params=None,
    gamma_type="equal"
):
    """ the whole process from data generation to metrics calculation. """
    true_mus = np.array([0, delta])
    true_sigmas = np.array([1, 1])
    X, Y = generate_data(true_mus, true_sigmas, nsamples)
    pred_y, pred_mus, pred_sigmas, pred_gammas = gaussian_mixture(
        X, typ, sigma, init_params, gamma_type)
    scores = score(
        true_mus, true_sigmas, Y,
        pred_y, pred_mus, pred_sigmas, pred_gammas
    )
    return np.r_[scores, nsamples, delta]


def main():
    # configures
    nsamples = np.arange(25, 501, 25)  # sample size
    deltas = np.arange(0.5, 5.1, 0.5)  # the difference of means
    repeat = 100                       # times of repetition
    sigma_modes = {
        "noequal_var": "full",
        "equalvar": "equal",
        "freeze_var": "custom",
        # "sklearn_kmeans_init": "sk_kmeans",
        # "sklearn_random_init": "sk_random"
    }
    init_modes = ["true_init", "random_init"]
    gamma_modes = {"estimated_gamma": "full", "freeze_gamma": "equal"}

    num_workers = 50                   #
    save_dir = "./results"

    for (gamma_name, gamma_mode), init_mode, (k, v) in product(
        gamma_modes.items(), init_modes, sigma_modes.items()
    ):
        # if init_mode == "random_init" and v.startswith("sk_"):
        #     continue
        if v == "custom":
            sigma = np.array([1., 1.])
        else:
            sigma = None
        print("task: %s - %s - %s" % (gamma_name, init_mode, k))
        save_name = "%s-%s-%s" % (gamma_name, init_mode, k)

        res = []
        if num_workers > 1:
            pool = Pool(num_workers)
        for nsample in nsamples:
            for delta in deltas:
                if init_mode == "random_init":
                    init_params = None
                else:
                    init_params = (np.array([0, delta]), np.array([1, 1]))
                init_params = None
                for _ in range(repeat):
                    if num_workers > 1:
                        res.append(pool.apply_async(
                            whole_process,
                            (nsample, delta, v, sigma,
                             init_params, gamma_mode)))
                    else:
                        res.append(whole_process(
                            nsample, delta, v, sigma,
                            init_params, gamma_mode))
        if num_workers > 1:
            res = np.array([r.get() for r in res])

        df = pd.DataFrame(
            res, columns=["adj_rand", "delta_diff", "mu1_diff", "mu2_diff",
                          "sigma1_diff", "sigma2_diff", "gamma1_diff",
                          "gamma2_diff", "nsamples", "delta"])
        df.to_csv(os.path.join(save_dir, "%s.csv" % save_name))

        df = df.melt(id_vars=["nsamples", "delta"], var_name="score",
                     value_name="value")
        df = df.astype({"nsamples": "category"})
        sns.relplot(data=df, x="delta", y="value",
                    hue="nsamples", col="score", kind="line",
                    col_wrap=3, facet_kws={"sharey": False})
        plt.savefig(os.path.join(save_dir, "%s.png" % save_name))


if __name__ == "__main__":
    main()
