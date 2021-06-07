from statistics import mean, stdev
from numpy import *
from math import sqrt


# Fits the input points to a line using the least squares,
# returns the m and b parameters of a line
def best_fit_slope_and_intercept(xs, ys):
    vertical_flag = False
    num = (mean(xs) * mean(ys)) - mean(xs * ys)
    den = ((mean(xs) * mean(xs)) - mean(xs * xs))
    if den == 0:
        den = 1
        vertical_flag = True
    m = num / den
    b = mean(ys) - m * mean(xs)
    #m, b = polyfit(xs, ys, 1)
    return m, b, vertical_flag


# Returns the vector of the distances (between the points of the cluster
# and the fitted line) and the standard deviation of the residuals
def fit_on_fly_lines(cluster_points):
    m, b, vertical_flag = best_fit_slope_and_intercept(cluster_points[:, 0], cluster_points[:, 1])
    if vertical_flag:
        distance = abs(mean(cluster_points[:, 0])-cluster_points[:, 0])
    else:
        distance = abs(b + m * cluster_points[:, 0] - cluster_points[:, 1]) / sqrt(1 + m ** 2)
    #print("Log debugg, distance: " + str(distance))
    return distance, sqrt(stdev(distance))

