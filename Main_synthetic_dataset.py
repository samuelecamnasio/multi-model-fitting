from getPoints import *
from pref_matrix import *
from numpy import *
from clustering import *
from pointExtractor import *
import time
if __name__ == "__main__":
    start_time = time.time()
    cluster_res_try = []
    cluster_res_suc = []

    # Tunable parameters for model selection criteria
    lambda1 = 1
    lambda2 = 2
    K = 2  # multiple of the sampling number
    noise = 1
    scene = 4 # 1 -> one line one circle, 2 -> three lines one circle, 3 -> three circles one line, 4 -> three circles three lines
    cutoff_threshold = 35   # cluster point threshold for outlier rejection

    # Generation of points
    points, real_clusters = generate_points2(scene, noise)

    # Compute the preference matrix for both lines and circles
    print("computing preference matrix for lines...")
    pref_mat = get_preference_matrix_2(points, "Line", K)
    print("computing preference matrix for circles...")
    pref_mat = hstack((pref_mat, get_preference_matrix_2(points, "Circle", K)))
    partial_before_clustering = time.time() - start_time

    # GRIC
    print("\nGRIC:")
    start_time = time.time()
    predicted_clusters_gric, a, b, first, second = clustering(pref_mat, points, 0, lambda1, lambda2)
    predicted_clusters_gric = delete_outliers(predicted_clusters_gric, cutoff_threshold)    # deletes clusters with n° of points below the set threshold
    print("In : %s seconds" % (time.time() - start_time + partial_before_clustering))
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    print("\nMedium first contr: ")
    print(first)
    print("\nMedium second contr: ")
    print(second)
    performance_evaluation(real_clusters, predicted_clusters_gric)

    # MDL
    print("\nMDL:")
    start_time = time.time()
    predicted_clusters_mdl, a, b, first, second= clustering(pref_mat, points, 1, lambda1, lambda2)
    predicted_clusters_mdl = delete_outliers(predicted_clusters_mdl, cutoff_threshold)
    print("In : %s seconds" % (time.time() - start_time + partial_before_clustering))
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    print("\nMedium first contr: ")
    print(first)
    print("\nMedium second contr: ")
    print(second)
    performance_evaluation(real_clusters, predicted_clusters_mdl)

    # GIC
    print("\nGIC:")
    start_time = time.time()
    predicted_clusters_gic, a, b, first, second  = clustering(pref_mat, points, 2, lambda1, lambda2)
    predicted_clusters_gic = delete_outliers(predicted_clusters_gic, cutoff_threshold)
    print("In : %s seconds" % (time.time() - start_time + partial_before_clustering))
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    print("\nMedium first contr: ")
    print(first)
    print("\nMedium second contr: ")
    print(second)
    performance_evaluation(real_clusters, predicted_clusters_gic)

    # GMDL
    print("\nGMDL:")
    start_time = time.time()
    predicted_clusters_gmdl, a, b, first, second = clustering(pref_mat, points, 3, lambda1, lambda2)
    predicted_clusters_gmdl = delete_outliers(predicted_clusters_gmdl, cutoff_threshold)
    print("In : %s seconds" % (time.time() - start_time + partial_before_clustering))
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    print("\nMedium first contr: ")
    print(first)
    print("\nMedium second contr: ")
    print(second)
    performance_evaluation(real_clusters, predicted_clusters_gmdl)

    print(cluster_res_try)
    print(cluster_res_suc)

    # plot all the results
    visualize_clusters_all_methods(predicted_clusters_gric, predicted_clusters_mdl, predicted_clusters_gic, predicted_clusters_gmdl, points)


