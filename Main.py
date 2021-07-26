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
    lambda1 = 1
    lambda2 = 2
    K = 3  # multiple of the sampling number
    noise = 1.5
    # Generation of points
    #points, real_clusters = generate_points(2, noise) # 1 complete, 2 test line and circle, anything else is test with only one line, 2nd param -> noise
    points, real_clusters = generate_points2(4, noise) # 1 one line one circle, 2 more lines than circle, 3 more circles than lines, anything else same circles and lines

    # Compute the preference matrix for both lines and circles
    print("computing preference matrix for lines...")
    pref_mat = get_preference_matrix_2(points, "Line", K)
    print("computing preference matrix for circles...")
    pref_mat = hstack((pref_mat, get_preference_matrix_2(points, "Circle", K)))
    partial_before_clustering = time.time() - start_time

    # Clustering (criteria: 0 -> GRIC, 1 -> MDL, 2 -> GIC, 3 -> GMDL)
    print("\nGRIC:")
    start_time = time.time()
    predicted_clusters_gric, a, b, first, second = clustering(pref_mat, points, 0)
    print("In : %s seconds" % (time.time() - start_time + partial_before_clustering))
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    print("\nMedium first contr: ")
    print(first)
    print("\nMedium second contr: ")
    print(second)
    performance_evaluation(real_clusters, predicted_clusters_gric)
    print("\nMDL:")
    start_time = time.time()
    predicted_clusters_mdl, a, b, first, second= clustering(pref_mat, points, 1)
    print("In : %s seconds" % (time.time() - start_time + partial_before_clustering))
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    print("\nMedium first contr: ")
    print(first)
    print("\nMedium second contr: ")
    print(second)
    performance_evaluation(real_clusters, predicted_clusters_mdl)
    print("\nGIC:")
    start_time = time.time()
    predicted_clusters_gic, a, b, first, second  = clustering(pref_mat, points, 2)
    print("In : %s seconds" % (time.time() - start_time + partial_before_clustering))
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    print("\nMedium first contr: ")
    print(first)
    print("\nMedium second contr: ")
    print(second)
    performance_evaluation(real_clusters, predicted_clusters_gic)
    print("\nGMDL:")
    start_time = time.time()
    predicted_clusters_gmdl, a, b, first, second = clustering(pref_mat, points, 3)
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
    # deletes clusters with n° of points below a threshold (2nd parameter)
    predicted_clusters_gric = delete_outliers(predicted_clusters_gric, 15)
    predicted_clusters_mdl = delete_outliers(predicted_clusters_mdl, 15)
    predicted_clusters_gic = delete_outliers(predicted_clusters_gic, 15)
    predicted_clusters_gmdl = delete_outliers(predicted_clusters_gmdl, 15)

    #visualize_clusters(predicted_clusters , points)
    visualize_clusters_all_methods(predicted_clusters_gric, predicted_clusters_mdl, predicted_clusters_gic, predicted_clusters_gmdl, points)


