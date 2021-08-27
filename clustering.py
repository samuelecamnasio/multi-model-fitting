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





def measure(x):
    return x[-1]


def clustering(pref_m, points, criteria, lambda1, lambda2, verbose=False ):
    clustering_try = 0
    clustering_suc = 0
    medium_first_cont = 0
    medium_second_cont = 0
    num_of_pts = pref_m.shape[0]
    pts = range(num_of_pts)
    clusters = [[i] for i in pts]
    new_idx = pref_m.shape[0]

    pos = {i: i for i in pts}

    temp = list(itertools.combinations(range(num_of_pts), 2))

    print("Starting computing intercluster distance...")
    # Using Tanimoto distance
    last_percentage = 0
    tot = len(temp)
    count = 0
    x0 = []
    for cl_i, cl_j in temp:
        curr_percentage = int((count/tot) * 100)
        count = count + 1
        x0.append((cl_i, cl_j, tanimoto_distance(pref_m[cl_i], pref_m[cl_j])))
        if curr_percentage != last_percentage:
            # change precentage
            print("\033[A                             \033[A", end="\r")
            print("Progress : " + str(curr_percentage) + "%", end="")
            last_percentage = curr_percentage
    print("\r", end="Progress : 100%\n")
    temp = []

    last_percentage = 0
    pr_count = 0
    tot = pref_m.shape[0]
    print("Starting clustering...")
    while pref_m.shape[0] > 1:
        curr_percentage = int(((tot - pref_m.shape[0])/tot) * 100)
        x0.sort(key=measure)
        cl_0, cl_1, min_distance = x0[0]
        if min_distance >= 1:
            break
        if verbose:
            print("Trying to fuse clusters "+str(clusters[pos[cl_0]])+" and "+str(clusters[pos[cl_1]]))
        clustering_try = clustering_try + 1
        union_line_score, first_line_cont, second_line_cont = model_selection(clusters[pos[cl_0]] + clusters[pos[cl_1]], "Line", points, criteria, lambda1, lambda2)
        line_1_score, dump1, dump2 = model_selection(clusters[pos[cl_0]], "Line", points, criteria, lambda1, lambda2)
        line_2_score, dump1, dump2 = model_selection(clusters[pos[cl_1]], "Line", points, criteria, lambda1, lambda2)

        union_circle_score, first_circ_contr, second_circ_contr = model_selection(clusters[pos[cl_0]] + clusters[pos[cl_1]], "Circle", points, criteria, lambda1, lambda2)
        circle_1_score, dump1, dump2 = model_selection(clusters[pos[cl_0]], "Circle", points, criteria, lambda1, lambda2)
        circle_2_score, dump1, dump2 = model_selection(clusters[pos[cl_1]], "Circle", points, criteria, lambda1, lambda2)

        if verbose:
            print("Line scores: "+str(union_line_score)+" union, "+str(line_2_score+line_1_score) + " single")
            print("Circle scores: " + str(union_circle_score) + " union, " + str(circle_1_score + circle_2_score) + " single")
        if union_line_score < (line_1_score + line_2_score):
            # accepted for line
            medium_first_cont += first_line_cont
            medium_second_cont += second_line_cont
        if union_circle_score < (circle_1_score + circle_2_score):
            # accepted for circ
            medium_first_cont += first_circ_contr
            medium_second_cont += second_line_cont

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
            clustering_suc = clustering_suc + 1
            if verbose:
                print("[CLUSTERING] New cluster: " + str(new_cluster) + "\t-\tTanimoto distance: " + str(min_distance))
        else:
            el=list(x0[0])
            el[-1] = 2  # set distance at an infinite value
            x0[0] = tuple(el)

        if curr_percentage != last_percentage:
            # change precentage
            print("\033[A                             \033[A", end = "\r")
            print("Progress : "+ str(curr_percentage)+"%", end = "")
            last_percentage = curr_percentage
    print("\r", end="Progress : 100%\n")
    medium_first_cont = medium_first_cont/clustering_suc
    medium_second_cont = medium_second_cont/clustering_suc
    return clusters, clustering_try, clustering_suc, medium_first_cont, medium_second_cont


def model_selection(cluster, mode, points, criteria, lambda1, lambda2, verbose=False):  # model_dimension = 2 for lines, = 3 for circumferences

    score = 0
    # cluster contains the indexes of the points that are in the cluster
    p_of_cluster = [points[cluster[0]]]
    for i in range(1, len(cluster)):
        p_of_cluster = np.vstack((p_of_cluster, points[cluster[i]]))
    #print(str(p_of_cluster) + " len " + str(len(p_of_cluster)))
    L = 200
    d = 1  # number of dimensions modeled (d=3 -> fund. matrix, d=2 -> homography, d=1 -> lines, circumferences)
    if mode == "Line":
        u = 2  # number of model parameters (u=2 for lines, u=3 for circumferences)
    elif mode == "Circle":
        u = 3
    u_max = 3

    if (len(p_of_cluster) > 1 and mode == "Line") or (len(p_of_cluster) > 2 and mode == "Circle"):

        if mode == "Line":  # if model is a line
            #print("In line evaluation")
            err, sigma = fit_on_fly_lines(
                p_of_cluster)  # sigma è un multiplo della deviazione standard del rumore sui dati
            if verbose:
                print("Line residue sum "+str(sum(err)) + " ,line residue variance " + str(sigma))
        elif mode == "Circle":  # if model is a circle (needs at leats 3 points)
            err, sigma = fit_on_fly_circles(p_of_cluster)
            if verbose:
                print("Circle residue sum " + str(sum(err)) + " ,circle residue variance " + str(sigma))


            #print("Jumped everything")

        # err -> r
        # sigma -> delta
        # u -> P
        # len(clusters) -> N

        #criteria = 3  # 0 -> GRIC, 1 -> MDL, 2 -> GIC, 3 -> GMDL

        if criteria == 0 :
            score, first_cont, second_cont = gric(p_of_cluster, err, sigma, lambda1, lambda2, d, len(cluster), u, mode)
            #print("Computed score: "+str(score))
        elif criteria == 1:
            score, first_cont, second_cont = mdl(p_of_cluster, err, sigma, len(cluster), u, u_max)
        elif criteria == 2:
            score, first_cont, second_cont = gic(p_of_cluster, err, sigma, lambda1, lambda2, d, len(cluster), u, u_max)
        elif criteria == 3:
            score, first_cont, second_cont = gmdl(p_of_cluster, err, len(cluster), u, u_max, sigma, d, L, mode)

    else:
        score = inf
        first_cont = 0
        second_cont = 0

    return score, first_cont, second_cont

def gric(p_of_cluster,r, sigma, lambda1, lambda2, d, N, u, mode):
    first_cont = 0
    rho = rho_calculation(r)
    for k in range(0, len(p_of_cluster)):
        first_cont += (float(rho[k]) * (float(r[k])) ** 2)
    if sigma == 0:
        sigma = 0.01
    first_cont = first_cont/(sigma**2)
    second_cont = lambda1 * d * N + lambda2 * u
    g = first_cont + second_cont
    return g, first_cont, second_cont

def mdl(p_of_cluster, r, sigma, N, u, u_max):
    first_cont = 0
    for k in range(0, len(p_of_cluster)):
        first_cont += ((float(r[k])) ** 2)
    if sigma == 0:
        sigma = 0.01
    first_cont = first_cont / (sigma ** 2)
    if N - u_max <= 0:
        den = N
    else:
        den = N - u_max
    delta = first_cont/den
    second_cont = (u/2)*np.log(N)*delta
    g = first_cont + second_cont
    return g, first_cont, second_cont

def gic(p_of_cluster, r, sigma, lambda1, lambda2, d, N, u, u_max):
    first_cont = 0
    for k in range(0, len(p_of_cluster)):
        first_cont += ((float(r[k])) ** 2)
    if sigma == 0:
        sigma = 0.01
    first_cont = first_cont / (sigma ** 2)
    if N - u_max <= 0:
        den = N
    else:
        den = N - u_max
    delta = first_cont / den
    second_cont = (lambda1 * d * N + lambda2 * u) * delta
    g = first_cont + second_cont
    return g, first_cont, second_cont

def gmdl(p_of_cluster, r, N, P, u_max, sigma, d, L, mode):
    first_cont = 0
    for k in range(0, len(p_of_cluster)):
        first_cont += ((float(r[k])) ** 2)
    if sigma == 0:  # case of perfect in the middle (perfect fit is covered)
        sigma = 0.01
    first_cont = first_cont / (sigma ** 2)
    if N - u_max <= 0:
        den = N
    else:
        den = N - u_max
    delta = first_cont / den
    if first_cont == 0:
        second_cont = 0
    else:
        second_cont = (N * d + P)*delta * (np.log(delta/(L**2)))
    g = first_cont - second_cont
    return g, first_cont, second_cont

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

def performance_evaluation(real_clusters, predicted_clusters):
    misclass_errors=[]
    for real_cluster in real_clusters:
        max_common_el = 0
        for predicted_cluster in predicted_clusters:
            intersection = set(real_cluster).intersection(predicted_cluster)
            intersection_as_list = list(intersection)
            if len(intersection_as_list) > max_common_el:
                max_common_el=len(intersection_as_list)
        misclass_errors.append(format(1-(max_common_el/len(real_cluster)), '.3f'))

    print("Misclassification errors:")
    for misclass_error in misclass_errors:
        print(misclass_error)
