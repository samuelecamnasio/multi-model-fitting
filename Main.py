from getPoints import *
from pref_matrix import *
from numpy import *
from clustering import *
from pointExtractor import *

if __name__ == "__main__":
    cluster_res_try = []
    cluster_res_suc = []
    K = 3  # multiple of the sampling number
    # Generation of points
    points, real_clusters = generate_points(2, 1.5) # 1 complete, 2 test line and circle, anything else is test with only one line, 2nd param -> noise
    # points, real_clusters = generate_points2(4, 1.5) # 1 one line one circle, 2 more lines than circle, 3 more circles than lines, anything else same circles and lines

    # Compute the preference matrix for both lines and circles
    print("computing preference matrix for lines...")
    pref_mat = get_preference_matrix_2(points, "Line", K)
    print("computing preference matrix for circles...")
    pref_mat = hstack((pref_mat, get_preference_matrix_2(points, "Circle", K)))

    # Clustering (criteria: 0 -> GRIC, 1 -> MDL, 2 -> GIC, 3 -> GMDL)
    print("\nGRIC:")
    predicted_clusters_gric, a, b = clustering(pref_mat, points, 0)
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    performance_evaluation(real_clusters, predicted_clusters_gric)
    print("\nMDL:")
    predicted_clusters_mdl, a, b = clustering(pref_mat, points, 1)
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    performance_evaluation(real_clusters, predicted_clusters_mdl)
    print("\nGIC:")
    predicted_clusters_gic, a, b  = clustering(pref_mat, points, 2)
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
    performance_evaluation(real_clusters, predicted_clusters_gic)
    print("\nGMDL:")
    predicted_clusters_gmdl, a, b = clustering(pref_mat, points, 3)
    cluster_res_suc.append(b)
    cluster_res_try.append(a)
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

    ##neds a data structure that sets all distances (truth is new distances have to be computed each time)

