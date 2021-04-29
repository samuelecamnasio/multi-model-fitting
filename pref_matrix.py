import scipy.io
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from Cluster import Cluster
from LineFitter import fit_on_fly_lines
from CircleFitter import fit_on_fly_circles


def distance_from_line(p1, p2,
                       p3):  # calculates the normal distance between a point p3 and a line passing through p1 and p2
    return (np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)) == 0


def get_preference_matrix(points):
    pref_mat = np.zeros((150, 15))
    # this is just for the lines
    for i in range(0, len(points) - 1, 10):  # iterates all the models
        p1 = points[i]
        p2 = points[i + 1]
        if (p1 == p2).all():  # checks if two points are equal to avoid the division by 0
            p2 = points[i + 2]
        i = int(i / 10)
        for k in range(0, len(points)):  # iterates all the points
            p3 = points[k]
            pref_mat[k][i] = distance_from_line(p1, p2, p3)  # populates the preference matrix

    # print(pref_mat[:, 14])
    return pref_mat

def get_preference_matrix_2(points, mode):
    K = 6  # temporary trials to do
    LINE_MSS = 2
    CIRCLE_MSS = 3

    threshold = 5  # to decide better

    num_samplings = K*len(points)
    pref_mat = []

    if mode == "Line":
        MSS = LINE_MSS
    elif mode == "Circle":
        MSS = CIRCLE_MSS

    for m in range(num_samplings):
        mss_indx = sample_points(points, MSS)
        for i in range(len(points)):
            if distance_from_line(points[mss_indx[0]], points[mss_indx[1]], points[i]) < threshold:
                pref_mat[i][m] = 1
            else:
                pref_mat[i][m] = 0

    return pref_mat

def sample_points(points, MSS):
    # avoid that the same point is taken two times
    indexes = np.zeros((2, 1))
    return indexes


"""
 The gric function should be used to compute the gric score for each cluster once it's created
 with the exception of the first iteration (in that case the score will be initialized to an 
 ideally infinite value)
 
"""
def gric(cluster):  # model_dimension = 2 for lines, = 3 for circumferences

    g = 0

    lambda1 = 1  # paper multilink, pag.6 (row 555/556)
    lambda2 = 2

    d = 1  # number of dimensions modeled (d=3 -> fund. matrix, d=2 -> homography, d=1 -> lines, circumferences)
    u = 2  # number of model paramters (u=2 for lines, u=3 for circumferences)

    if cluster.model_type == "Line":  # if model is a line
        err, sigma = fit_on_fly_lines(
            cluster.points)  # sigma è un multiplo della deviazione standard del rumore sui dati
    elif cluster.model_type == "Circle":  # if model is a circle
        err, sigma = fit_on_fly_circles(cluster.points)

    rho = rho_calculation(err)
    for k in range(0, len(cluster.points) - 1):
        g += rho[k] * (err[k] / sigma) ^ 2 + lambda1 * d * len(cluster) + lambda2 * u

    return g


def rho_calculation(
        error):  # ATM: binary, equals 1 for inliers (residuals < epsilon) and 0 for outliers. Should be done with M-estimators

    rho = np.zeros((1, len(error)))
    for k in range(0, len(error.points) - 1):  # iterates all the points
        if (error[k] > 4):
            rho[k] = 0
        else:
            rho[k] = 1

    return rho


# the .mat file is structured with 150 couples of points where from 10 to 10 they belong to the same line
mat = scipy.io.loadmat('punti_prova.mat')  # loads the .mat containing the points
mat = mat['A']

prova = Cluster(mat[0:5], 1000, "line")

print("Cluster prova: " + str(prova.points))

pm = get_preference_matrix(mat)  # preference matrix calculation
