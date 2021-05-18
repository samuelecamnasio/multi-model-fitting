from getPoints import *
from pref_matrix import *
from numpy import *
from clustering import *

if __name__ == "__main__":
    # Extract points from the .mat or the generated points
    points = generate_points(3) # 1 complete, 2 test line and circle, anything else is test with only one line

    # Compute the preference matrix for both lines and circles
    pref_mat = get_preference_matrix_2(points, "Line")
    pref_mat = hstack((pref_mat, get_preference_matrix_2(points, "Circle")))

    # Clustering
    clusters = clustering(pref_mat, points)

    visualize_clusters(clusters, points)

    ##neds a data structure that sets all distances (truth is new distances have to be computed each time)

