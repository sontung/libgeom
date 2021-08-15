import numpy as np
from numba import njit, prange


@njit("f8[:, :](f8[:, :, :], f8[:, :, :])")
def fast_sum(x, ty):
    res = np.sum((x - ty) ** 2, axis=2)
    return res


@njit("f8[:, :](f8[:, :], f8)", parallel=True)
def fast_exp(p, s):
    for i in prange(p.shape[0]):
        for j in prange(p.shape[1]):
            p[i, j] = np.exp(-p[i, j] / (2 * s))
    return p


def initialize_sigma2(x_mat, y_mat):
    (N, D) = x_mat.shape
    (M, _) = y_mat.shape
    diff = x_mat[None, :, :] - y_mat[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)


def expectation(x_mat, ty, sigma2, m, n, d):
    p_mat = fast_sum(x_mat[None, :, :], ty[:, None, :])
    c = (2 * np.pi * sigma2) ** (d / 2)
    c = c * m / n
    p_mat = fast_exp(p_mat, sigma2)
    den = np.sum(p_mat, axis=0)
    den[den == 0] = np.finfo(float).eps
    den += c
    p_mat = p_mat / den[None, :]
    pt1 = np.sum(p_mat, axis=0)
    p1_mat = np.sum(p_mat, axis=1)
    np_mat = np.sum(p1_mat)
    return p_mat, p1_mat, pt1, np_mat


def rigid_register_fast(x_mat, y_mat, max_iterations=100):
    """
    fast rigid registration by coherent point drift algorithm
    """
    sigma2 = initialize_sigma2(x_mat, y_mat)
    (N, D) = x_mat.shape
    (M, _) = y_mat.shape
    b_mat = np.eye(D)
    t = np.atleast_2d(np.zeros((1, D)))
    ty = np.dot(y_mat, b_mat) + np.tile(t, (M, 1))
    iteration = 0
    tolerance = 0.001
    old_angle = 0
    diff_all = []
    while iteration < max_iterations:
        iteration += 1

        # expectation
        p_mat, p1_mat, pt1, np_mat = expectation(x_mat, ty, sigma2, M, N, D)

        # update transform
        mu_x = np.divide(np.sum(np.dot(p_mat, x_mat), axis=0), np_mat)
        # source point cloud mean
        mu_y = np.divide(np.sum(np.dot(np.transpose(p_mat), y_mat), axis=0), np_mat)

        x_hat = x_mat - np.tile(mu_x, (N, 1))
        # centered source point cloud
        y_hat = y_mat - np.tile(mu_y, (M, 1))
        ypy = np.dot(np.transpose(p1_mat), np.sum(np.multiply(y_hat, y_hat), axis=1))

        a_mat = np.dot(np.transpose(x_hat), np.transpose(p_mat))
        a_mat = np.dot(a_mat, y_hat)

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        u_mat, _, v_mat = np.linalg.svd(a_mat, full_matrices=True)
        c_mat = np.ones((D, ))
        c_mat[D-1] = np.linalg.det(np.dot(u_mat, v_mat))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        r_mat = np.transpose(np.dot(np.dot(u_mat, np.diag(c_mat)), v_mat))
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        s = np.trace(np.dot(np.transpose(a_mat),
                                 np.transpose(r_mat))) / ypy
        t = np.transpose(mu_x) - s * \
            np.dot(np.transpose(r_mat), np.transpose(mu_y))

        rot_angle = np.rad2deg(np.arctan2(r_mat[1, 0], r_mat[0, 0]))

        if abs(rot_angle - old_angle) < 0.001 and iteration > 1000:
            iteration = max_iterations
        old_angle = rot_angle

        # transform pc
        ty = s * np.dot(y_mat, r_mat) + t

        # update variance
        tr_ar = np.trace(np.dot(a_mat, r_mat))
        x_px = np.dot(np.transpose(pt1), np.sum(
            np.multiply(x_hat, x_hat), axis=1))
        sigma2 = (x_px - s * tr_ar) / (np_mat * D)
        if sigma2 <= 0:
            sigma2 = tolerance / 10

    return ty, b_mat, t, diff_all


def affine_register_fast(x_mat, y_mat, max_iterations=100, threshold=0.1, stop_early=False):
    """
    fast affine registration by coherent point drift algorithm
    """
    sigma2 = initialize_sigma2(x_mat, y_mat)
    (N, D) = x_mat.shape
    (M, _) = y_mat.shape
    b_mat = np.eye(D)
    t = np.atleast_2d(np.zeros((1, D)))
    ty = np.dot(y_mat, b_mat) + np.tile(t, (M, 1))
    iteration = 0
    tolerance = 0.001
    q = np.inf
    diff_all = []
    while iteration < max_iterations:
        iteration += 1

        # expectation
        p_mat, p1_mat, pt1, np_mat = expectation(x_mat, ty, sigma2, M, N, D)

        # update transform
        mu_x = np.divide(np.sum(np.dot(p_mat, x_mat), axis=0), np_mat)
        mu_y = np.divide(
            np.sum(np.dot(np.transpose(p_mat), y_mat), axis=0), np_mat)

        x_hat = x_mat - np.tile(mu_x, (N, 1))
        y_hat = y_mat - np.tile(mu_y, (M, 1))

        a_mat = np.dot(np.transpose(x_hat), np.transpose(p_mat))
        a_mat = np.dot(a_mat, y_hat)

        ypy = np.transpose(y_hat)*p1_mat  # faster version of ypy = np.dot(np.transpose(y_hat), np.diag(p1_mat))
        ypy = np.dot(ypy, y_hat)

        try:
            b_mat = np.linalg.solve(np.transpose(ypy), np.transpose(a_mat))
        except np.linalg.LinAlgError:
            return None
        t = np.transpose(
            mu_x) - np.dot(np.transpose(b_mat), np.transpose(mu_y))

        # transform pc
        ty = np.dot(y_mat, b_mat) + np.tile(t, (M, 1))

        # update variance
        qprev = q

        tr_ab = np.trace(np.dot(a_mat, b_mat))
        x_px = np.dot(np.transpose(pt1), np.sum(
            np.multiply(x_hat, x_hat), axis=1))
        tr_bypyp = np.trace(np.dot(np.dot(b_mat, ypy), b_mat))
        q = (x_px - 2 * tr_ab + tr_bypyp) / (2 * sigma2) + \
                 D * np_mat / 2 * np.log(sigma2)
        diff = np.abs(q - qprev)
        diff_all.append(diff)
        if diff < threshold and stop_early:
            break

        sigma2 = (x_px - tr_ab) / (np_mat * D)

        if sigma2 <= 0:
            sigma2 = tolerance / 10
    return ty, b_mat, t, diff_all
