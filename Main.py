from getPoints import *
from pref_matrix import *
from numpy import *
from clustering import *
from pointExtractor import *

if __name__ == "__main__":
    K = 3  # multiple of the sampling number
    # Extract points from the .mat or the generated points
    #points = generate_points(1, 1.5) # 1 complete, 2 test line and circle, anything else is test with only one line, 2nd param -> noise
    points = point_from_image(400, "shape", "2")
    # Compute the preference matrix for both lines and circles
    print("computing preference matrix for lines...")
    pref_mat = get_preference_matrix_2(points, "Line", K)
    print("computing preference matrix for circles...")
    pref_mat = hstack((pref_mat, get_preference_matrix_2(points, "Circle", K)))

    # Clustering
    clusters = clustering(pref_mat, points, 0) #criteria: 0 -> GRIC, 1 -> MDL, 2 -> GIC, 3 -> GMDL
    clusters = delete_outliers(clusters, 15)    #deletes clusters with n° of points below a threshold (2nd parameter)

    visualize_clusters(clusters, points)

    ##neds a data structure that sets all distances (truth is new distances have to be computed each time)

