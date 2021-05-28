import scipy.io
import numpy as np
from Cluster import Cluster
from LineFitter import fit_on_fly_lines
from CircleFitter import fit_on_fly_circles
from math import sqrt
from getPoints import show_pref_matrix


def distance_from_line(p1, p2,
                       p3):  # calculates the normal distance between a point p3 and a line passing through p1 and p2
    return (np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1))

# Function to find the circle on
# which the given three points lie
def find_circle(x1, y1, x2, y2, x3, y3) :
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
          ((y31) * (x12) - (y21) * (x13))));

    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
          (2 * ((x31) * (y12) - (x21) * (y13))));

    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1);

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c

    # r is the radius
    r = round(sqrt(sqr_of_r), 5)

    return h,k,r

def distance_from_circ(p1, p2, p3, p4):  # calculates the normal distance between a point p4 and a circle passing through p1, p2 and p3
    h,k,r = find_circle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]) #find centre and radius
    return abs(sqrt(pow(p4[0]-h,2) + pow(p4[1]-k,2)) - r)

# def get_preference_matrix(points):
#     pref_mat = np.zeros((150, 15))
#     # this is just for the lines
#     for i in range(0, len(points) - 1, 10):  # iterates all the models
#         p1 = points[i]
#         p2 = points[i + 1]
#         if (p1 == p2).all():  # checks if two points are equal to avoid the division by 0
#             p2 = points[i + 2]
#         i = int(i / 10)
#         for k in range(0, len(points)):  # iterates all the points
#             p3 = points[k]
#             pref_mat[k][i] = distance_from_line(p1, p2, p3)  # populates the preference matrix
#
#     # print(pref_mat[:, 14])
#     return pref_mat

def get_preference_matrix_2(points, mode):
    # TODO change the K to select an optimal number of sampling
    K = 6  # temporary trials to do
    LINE_MSS = 2
    CIRCLE_MSS = 3

    threshold = 1.5  # to decide better

    num_samplings = K*len(points)
    pref_mat = np.zeros((len(points), num_samplings))

    for m in range(num_samplings):
        if mode == "Line":
            mss_indx = sample_points(points, LINE_MSS, "localized")
            for i in range(len(points)):
                residue = distance_from_line(points[mss_indx[0]], points[mss_indx[1]], points[i])
                if residue < 5*threshold:
                    pref_mat[i][m] = np.exp(-residue/threshold)
                else:
                    pref_mat[i][m] = 0
        elif mode == "Circle":
            mss_indx = sample_points(points, CIRCLE_MSS, "localized")
            for i in range(len(points)):
                residue = distance_from_circ(points[mss_indx[0], :], points[mss_indx[1], :], points[mss_indx[2], :], points[i])
                if residue < 5*threshold:
                    pref_mat[i][m] = np.exp(-residue/threshold)
                else:
                    pref_mat[i][m] = 0
        else:
            raise Exception("get_preference_matrix -> no mode selected")

    show_pref_matrix(pref_mat, mode)
    return pref_mat


def sample_points(points, MSS, mode="uniform"):
    if mode == "uniform":
        return sample_points_uniform(points, MSS)
    elif mode == "localized":
        return sample_points_localized(points, MSS)
    else:
        raise Exception("sample_points-> no mode selected")


def sample_points_uniform(points, MSS):
    g = np.random.Generator(np.random.PCG64())
    mss_idx = np.array(g.choice(len(points), MSS, replace=False))
    return mss_idx

def sample_points_localized(src_pts, k, ni=1 / 3):
    num_of_pts = src_pts.shape[0]
    g = np.random.Generator(np.random.PCG64())

    mss0 = g.choice(num_of_pts, 1)

    prob_local = get_localized_prob(src_pts, src_pts[mss0], ni)

    prob = np.max([prob_local], axis=0)
    prob[mss0] = 0.0
    prob = prob / np.sum(prob)

    mss1 = g.choice(num_of_pts, k - 1, replace=False, p=prob)

    mss = mss0.tolist() + mss1.tolist()

    return np.array(mss)

def get_localized_prob(pts, pt, ni):
    d_squared = np.sum(np.square(np.subtract(pts, pt)), axis=1)

    sigma = ni * np.median(np.sqrt(d_squared))
    sigma_squared = sigma ** 2

    prob = np.exp(- (1 / sigma_squared) * d_squared)

    return prob

"""
 The gric function should be used to compute the gric score for each cluster once it's created
 with the exception of the first iteration (in that case the score will be initialized to an 
 ideally infinite value)
 
"""


#
#
##TODO
#def create_clusters(points):
#    cluster_list = np.array()
#    for i in range(len(points)):
#        ## create new cluster
#     print()
#    return cluster_list
#
##TODO
#def create_distance_matrix(points ):
#    distance_mat = np.zeros([len(points), len(points)])
#    cluster_array = []
#    # Dictionary creation
#    thisdict = {
#    }
#
#    # Population of the dictionary ("index_in_dist_matr": index)
#    for i in range(len(points)):
#        cluster_array.append(Cluster(str(i), [points[i]], 0, "line"))
#        thisdict[str(i)] = i
#    print(thisdict)
#
#    #Calculation of distances and population of the matrix
#    for c1 in cluster_array:
#        for c2 in cluster_array:
#            distance_mat[thisdict[c1.name], thisdict[c2.name]] = jaccard_distance(c1, c2)
#
#    print(distance_mat)
#    return 0
#


# the .mat file is structured with 150 couples of points where from 10 to 10 they belong to the same line
#mat = scipy.io.loadmat('punti_prova.mat')  # loads the .mat containing the points
#mat = mat['A']

#prova = Cluster(mat[0:5], 1000, "line")

#print("Cluster prova: " + str(prova.points))

#pm = get_preference_matrix(mat)  # preference matrix calculation
#create_distance_matrix([[0,0], [1,2], [1,3], [0,0], [4,5]])

