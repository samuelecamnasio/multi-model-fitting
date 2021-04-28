from statistics import mean, stdev
from numpy import *
from math import sqrt


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


def compute_error_for_lines(cluster_points, m, b):
    # our noisy points
    distance = abs(b + m * cluster_points[:, 0] - cluster_points[:, 1]) / sqrt(1 + m ** 2)
    return distance, stdev(distance)


def fit_on_fly_lines(cluster_points):
    m, b = best_fit_slope_and_intercept(cluster_points[:, 0], cluster_points[:, 1])
    return compute_error_for_lines(cluster_points, m, b)

