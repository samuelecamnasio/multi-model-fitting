import scipy.io
import numpy as np
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
    sqr_of_r = pow(h, 2) + pow(k, 2) - c
    if sqr_of_r<0:
        r = "err"
    else:
        # r is the radius
        r = round(sqrt(sqr_of_r), 5)
    return h,k,r

def distance_from_circ(h, k, r, p4):  # calculates the normal distance between a point p4 and a circle passing through p1, p2 and p3
    return abs(sqrt(pow(p4[0]-h,2) + pow(p4[1]-k,2)) - r)


def get_preference_matrix_2(points, mode, K):
    # TODO change the K to select an optimal number of sampling
    #K = 3  # temporary trials to do
    LINE_MSS = 2
    CIRCLE_MSS = 3

    threshold = 1.5  # to decide better

    num_samplings = K*len(points)
    pref_mat = np.zeros((len(points), num_samplings))

    last_percentage = 0
    for m in range(num_samplings):
        curr_percentage = int((m / num_samplings) * 100)
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
            while on_a_line(points[mss_indx[0], :], points[mss_indx[1], :], points[mss_indx[2], :]):
                mss_indx = sample_points(points, CIRCLE_MSS, "localized")
            h,k,r = find_circle(points[mss_indx[0], 0], points[mss_indx[0], 1], points[mss_indx[1], 0], points[mss_indx[1], 1], points[mss_indx[2], 0], points[mss_indx[2], 1])
            while r == "err":
                mss_indx = sample_points(points, CIRCLE_MSS, "localized")
                while on_a_line(points[mss_indx[0], :], points[mss_indx[1], :], points[mss_indx[2], :]):
                    mss_indx = sample_points(points, CIRCLE_MSS, "localized")
                h,k,r = find_circle(points[mss_indx[0], 0], points[mss_indx[0], 1], points[mss_indx[1], 0], points[mss_indx[1], 1], points[mss_indx[2], 0], points[mss_indx[2], 1])

            for i in range(len(points)):
                residue = distance_from_circ(h, k, r, points[i])
                if residue < 5*threshold:
                    pref_mat[i][m] = np.exp(-residue/threshold)
                else:
                    pref_mat[i][m] = 0
        else:
            raise Exception("get_preference_matrix -> no mode selected")
        # printing progress percentages
        if curr_percentage != last_percentage:
            # change precentage
            print("\033[A                             \033[A", end = "\r")
            print("Progress : "+ str(curr_percentage)+"%", end = "")
            last_percentage = curr_percentage

    print("\r", end="Progress : 100%\n")
    show_pref_matrix(pref_mat, mode)
    return pref_mat


def on_a_line(p1, p2, p3):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    return (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) == 0

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

