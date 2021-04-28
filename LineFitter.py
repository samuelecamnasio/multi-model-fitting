from statistics import mean, stdev
from numpy import *
from math import sqrt


# Fits the input points to a line,
# returns the m and b parameters of a line
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


# Returns the vector of the distances (between the points of the cluster
# and the fitted line) and the standard deviation of the residuals
def fit_on_fly_lines(cluster_points):
    m, b = best_fit_slope_and_intercept(cluster_points[:, 0], cluster_points[:, 1])
    distance = abs(b + m * cluster_points[:, 0] - cluster_points[:, 1]) / sqrt(1 + m ** 2)
    return distance, stdev(distance)

