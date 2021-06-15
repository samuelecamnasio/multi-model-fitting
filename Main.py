from getPoints import *
from pref_matrix import *
from numpy import *
from clustering import *
from pointExtractor import *

if __name__ == "__main__":
    K = 3  # multiple of the sampling number
    # Generation of points
    points, real_clusters = generate_points(2, 1.5) # 1 complete, 2 test line and circle, anything else is test with only one line, 2nd param -> noise

    # Compute the preference matrix for both lines and circles
    print("computing preference matrix for lines...")
    pref_mat = get_preference_matrix_2(points, "Line", K)
    print("computing preference matrix for circles...")
    pref_mat = hstack((pref_mat, get_preference_matrix_2(points, "Circle", K)))

    # Clustering (criteria: 0 -> GRIC, 1 -> MDL, 2 -> GIC, 3 -> GMDL)
    print("\nGRIC:")
    predicted_clusters_gric = clustering(pref_mat, points, 0)
    performance_evaluation(real_clusters, predicted_clusters_gric)
    print("\nMDL:")
    predicted_clusters_mdl = clustering(pref_mat, points, 1)
    performance_evaluation(real_clusters, predicted_clusters_mdl)
    print("\nGIC:")
    predicted_clusters_gic = clustering(pref_mat, points, 2)
    performance_evaluation(real_clusters, predicted_clusters_gic)
    print("\nGMDL:")
    predicted_clusters_gmdl = clustering(pref_mat, points, 3)
    performance_evaluation(real_clusters, predicted_clusters_gmdl)


    # deletes clusters with n° of points below a threshold (2nd parameter)
    predicted_clusters_gric = delete_outliers(predicted_clusters_gric, 15)
    predicted_clusters_mdl = delete_outliers(predicted_clusters_mdl, 15)
    predicted_clusters_gic = delete_outliers(predicted_clusters_gic, 15)
    predicted_clusters_gmdl = delete_outliers(predicted_clusters_gmdl, 15)

    #visualize_clusters(predicted_clusters , points)
    visualize_clusters_all_methods(predicted_clusters_gric, predicted_clusters_mdl, predicted_clusters_gic, predicted_clusters_gmdl, points)

    ##neds a data structure that sets all distances (truth is new distances have to be computed each time)
