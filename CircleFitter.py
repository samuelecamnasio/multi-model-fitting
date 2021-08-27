import numpy as np
from math import sqrt

# from https://github.com/AlliedToasters/circle-fit/blob/master/demo.ipynb

def calc_R(x, y, xc, yc):
    """
    calculate the distance of each 2D points from the center (xc, yc)
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def f(c, x, y):
    """
    calculate the algebraic distance between the data points
    and the mean circle centered at c=(xc, yc)
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def residuals_and_variance(cluster_points, x, y, r):
    """Computes Sigma for circle fit."""
    dx, dy, sum_ = 0., 0., 0.
    residuals = []
    for i in range(len(cluster_points)):
        dx = cluster_points[i][1] - x
        dy = cluster_points[i][0] - y
        residuals.append(sqrt(dx * dx + dy * dy) - r)
        sum_ += (sqrt(dx * dx + dy * dy) - r) ** 2
    return residuals, sqrt(sum_ / len(cluster_points))


def hyper_fit(cluster_points, IterMax=99, verbose=False):
    """
    Fits cluster_points to circle using hyperfit algorithm.
    Inputs:
        - cluster_points, list or numpy array with len>2 of the form:
        [
    [x_coord, y_coord],
    ...,
    [x_coord, y_coord]
    ]
        or numpy array of shape (n, 2)
    Outputs:
        residuals - residuals of the input points
        sigma - variance of data wrt solution (float)
    """
    X, X = None, None
    if isinstance(cluster_points, np.ndarray):
        X = cluster_points[:, 0]
        Y = cluster_points[:, 1]
    elif isinstance(cluster_points, list):
        X = np.array([x[0] for x in cluster_points])
        Y = np.array([x[1] for x in cluster_points])
    else:
        raise Exception("Parameter 'cluster_points' is an unsupported type: " + str(type(cluster_points)))

    n = X.shape[0]

    Xi = X - X.mean()
    Yi = Y - Y.mean()
    Zi = Xi * Xi + Yi * Yi

    # compute moments
    Mxy = (Xi * Yi).sum() / n
    Mxx = (Xi * Xi).sum() / n
    Myy = (Yi * Yi).sum() / n
    Mxz = (Xi * Zi).sum() / n
    Myz = (Yi * Zi).sum() / n
    Mzz = (Zi * Zi).sum() / n

    # computing the coefficients of characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    Var_z = Mzz - Mz * Mz

    A2 = 4 * Cov_xy - 3 * Mz * Mz - Mzz
    A1 = Var_z * Mz + 4. * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy
    A22 = A2 + A2

    # finding the root of the characteristic polynomial
    y = A0
    x = 0.
    for i in range(IterMax):
        Dy = A1 + x * (A22 + 16. * x * x)
        xnew = x - y / Dy
        if xnew == x or not np.isfinite(xnew):
            break
        ynew = A0 + xnew * (A1 + xnew * (A2 + 4. * xnew * xnew))
        if abs(ynew) >= abs(y):
            break
        x, y = xnew, ynew

    det = x * x - x * Mz + Cov_xy
    Xcenter = (Mxz * (Myy - x) - Myz * Mxy) / det / 2.
    Ycenter = (Myz * (Mxx - x) - Mxz * Mxy) / det / 2.

    r = sqrt(abs(Xcenter ** 2 + Ycenter ** 2 + Mz))
    residuals, sigma = residuals_and_variance(cluster_points, x, y, r)

    return residuals, sigma


def fit_on_fly_circles(cluster_points):
    return hyper_fit(cluster_points)
