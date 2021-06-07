import itertools
import numpy as np
from pref_matrix import *
from math import *

def tanimoto_distance(i, j):
    pq = np.inner(i, j)

    if pq == 0:
        return 1

    p_square = np.inner(i, i)
    q_square = np.inner(j, j)

    t_distance = 1 - pq / (p_square + q_square - pq)
    return t_distance


# TODO
def jaccard_distance(A, B):  # A and B are points clusters
    # print(str(A.name) + " : " + str(A.points))
    # print(str(B.name) + " : " + str(B.points))

    # union = Cluster("", [], 0, A.type)
    union = A
    # for point in A.points:
    #    union.points += [point]
    # intersection = Cluster("", [], 0, A.type)
    intersection = []
    for point in B:
        if point in A:
            intersection += [point]
        if point not in A:
            union += [point]
    return (len(union) - len(intersection)) / len(union)


def measure(x):
    return x[-1]


def clustering(pref_m, points, criteria, dist_type="Tanimoto" ):
    num_of_pts = pref_m.shape[0]
    pts = range(num_of_pts)
    clusters = [[i] for i in pts]
    new_idx = pref_m.shape[0]

    pos = {i: i for i in pts}

    x0 = list(itertools.combinations(range(num_of_pts), 2))

    if dist_type == "Tanimoto":
        # Using Tanimoto distance
        x0 = [(cl_i, cl_j, tanimoto_distance(pref_m[cl_i], pref_m[cl_j])) for cl_i, cl_j in x0]
    elif dist_type == "Jaccard":
        # Using Jaccard distance
        # TODO
        x0 = [(cl_i, cl_j, jaccard_distance(clusters[pos[cl_i]], clusters[pos[cl_j]])) for cl_i, cl_j in x0]

    while pref_m.shape[0] > 1:
        x0.sort(key=measure)
        cl_0, cl_1, min_distance = x0[0]
        if min_distance >= 1:
            break

        print("Trying to fuse clusters "+str(clusters[pos[cl_0]])+" and "+str(clusters[pos[cl_1]]))

        union_line_score = model_selection(clusters[pos[cl_0]] + clusters[pos[cl_1]], "Line", points, criteria)
        line_1_score = model_selection(clusters[pos[cl_0]], "Line", points, criteria)
        line_2_score = model_selection(clusters[pos[cl_1]], "Line", points, criteria)
        union_circle_score = model_selection(clusters[pos[cl_0]] + clusters[pos[cl_1]], "Circle", points, criteria)
        circle_1_score = model_selection(clusters[pos[cl_0]], "Circle", points, criteria)
        circle_2_score = model_selection(clusters[pos[cl_1]], "Circle", points, criteria)
        #union_circle_score = 10
        #circle_1_score = 1
        #circle_2_score = 1
        print("Line scores: "+str(union_line_score)+" union, "+str(line_2_score+line_1_score) + " single")
        print("Circle scores: " + str(union_circle_score) + " union, " + str(circle_1_score + circle_2_score) + " single")



        #out = 0
        #for i in range(len(pref_m)):
        #    if new_pf[i] > 0:
        #        out = 1
        #        print("found at lest one")
        #        break


        if union_line_score < (line_1_score + line_2_score) or union_circle_score < (circle_1_score + circle_2_score) :

            new_pf = np.minimum(pref_m[pos[cl_0]], pref_m[pos[cl_1]])  # element-wise min
            ##--------
            new_cluster = clusters[pos[cl_0]] + clusters[pos[cl_1]]

            pref_m = np.delete(pref_m, (pos[cl_0], pos[cl_1]), axis=0)
            pref_m = np.vstack((pref_m, new_pf))
            clusters = [c for idx_c, c in enumerate(clusters) if
                        idx_c not in (pos[cl_0], pos[cl_1])]  # delete C_i and C_j
            clusters = clusters + [new_cluster]
            new_cluster.sort()

            pos0 = pos[cl_0]
            pos1 = pos[cl_1]
            del pos[cl_0]
            del pos[cl_1]

            for k in pos:
                if pos[k] >= pos0:
                    pos[k] -= 1
                if pos[k] >= pos1:
                    pos[k] -= 1

            pos[new_idx] = pref_m.shape[0] - 1

            pts = [p for p in pts if p not in (cl_0, cl_1)]
            x0 = [(cl_i, cl_j, d) for cl_i, cl_j, d in x0
                  if cl_i not in (cl_0, cl_1) and cl_j not in (cl_0, cl_1)]

            new_comb = [(p, new_idx) for p in pts]
            pts.append(new_idx)
            new_idx += 1
            x1 = [(cl_i, cl_j, tanimoto_distance(pref_m[pos[cl_i]], pref_m[pos[cl_j]])) for cl_i, cl_j in new_comb]
            x0 += x1

            print("[CLUSTERING] New cluster: " + str(new_cluster)
                  + "\t-\tTanimoto distance: " + str(min_distance))
        else:
            el=list(x0[0])
            el[-1] = 2  # set distance at an infinite value
            x0[0] = tuple(el)


    return clusters


def model_selection(cluster, mode, points, criteria):  # model_dimension = 2 for lines, = 3 for circumferences

    score = 0
    # cluster contains the indexes of the points that are in the cluster
    p_of_cluster = [points[cluster[0]]]
    for i in range(1, len(cluster)):
        p_of_cluster = np.vstack((p_of_cluster, points[cluster[i]]))

    lambda1 = 1  # paper multilink, pag.6 (row 555/556)
    lambda2 = 2
    L = 200*200
    d = 1  # number of dimensions modeled (d=3 -> fund. matrix, d=2 -> homography, d=1 -> lines, circumferences)
    if mode == "Line":
        u = 2  # number of model paramters (u=2 for lines, u=3 for circumferences)
    elif mode == "Circle":
        u = 3

    if (len(p_of_cluster) > 1 and mode == "Line") or (len(p_of_cluster) > 2 and mode == "Circle"):

        if mode == "Line":  # if model is a line
            err, sigma = fit_on_fly_lines(
                p_of_cluster)  # sigma è un multiplo della deviazione standard del rumore sui dati
            print("Line residue sum "+str(sum(err)) + " ,line residue variance " + str(sigma))
        elif mode == "Circle":  # if model is a circle (needs at leats 3 points)
            err, sigma = fit_on_fly_circles(p_of_cluster)
            print("Circle residue sum " + str(sum(err)) + " ,circle residue variance " + str(sigma))

        
        # err -> r
        # sigma -> delta
        # u -> P
        # len(clusters) -> N

        #criteria = 3  # 0 -> GRIC, 1 -> MDL, 2 -> GIC, 3 -> GMDL

        if criteria == 0 :
            score = gric(p_of_cluster, err, sigma, lambda1, lambda2, d, len(cluster), u)
        elif criteria == 1:
            score = mdl(p_of_cluster, err, sigma, len(cluster), u)
        elif criteria == 2:
            score = gic()
        elif criteria == 3:
            score = gmdl(p_of_cluster, err, len(cluster), u, sigma, d, L)

    else :
        score = inf

    return score

def gric(p_of_cluster,r, delta, lambda1, lambda2, d, N, u):
    g=0
    if (delta == 0):
        if (len(p_of_cluster) == 2):
            g = inf
        else:
            print("\n\nentrato (g=100)\n\n")
            g = 100
    else:
        rho = rho_calculation(r)
        for k in range(0, len(p_of_cluster)):
            # TODO: case sigma=0 (same error for multiple points)
            # TODO: needs to work also on the parameters, there are problem when recognizing a model in front of another
            g += (float(rho[k]) * (float(r[k]) / delta) ** 2)
        g += (lambda1 * d * N + lambda2 * u)
    return g

def mdl(p_of_cluster,r, delta, N, u):
    score=0
    for k in range(0, len(p_of_cluster)):
        # TODO: case sigma=0 (same error for multiple points)
        # TODO: needs to work also on the parameters, there are problem when recognizing a model in front of another
        score += float(r[k])
    score += (u/2)*np.log(N)*delta**2
    return score

def gic():
    score=0
    return score

def gmdl(p_of_cluster, r, N, P, delta, d, L):
    score=0
    for k in range(0, len(p_of_cluster)):
        score += float(r[k])**2
    score -= (N * d + P)*(delta**2) * (np.log(delta/L)**2)
    return score


def rho_calculation(
        error):  # ATM: binary, equals 1 for inliers (residuals < epsilon) and 0 for outliers. Should be done with M-estimators

    rho = np.zeros((len(error), 1))
    for k in range(0, len(error)):  # iterates all the points
        if (error[k] > 4):
            rho[k] = 0
        else:
            rho[k] = 1

    return rho

def delete_outliers(clusters, threshold):

    clusters[:] = [cluster for cluster in clusters if not len(cluster)<threshold]

    return clusters
