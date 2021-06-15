from getPoints import *
from pref_matrix import *
from numpy import *
from clustering import *
from pointExtractor import *

if __name__ == "__main__":
    K = 3  # multiple of the sampling number
    # Extract edge points from a real image
    points = point_from_image(400, "shape", "1")

    # Compute the preference matrix for both lines and circles
    print("computing preference matrix for lines...")
    pref_mat = get_preference_matrix_2(points, "Line", K)
    print("computing preference matrix for circles...")
    pref_mat = hstack((pref_mat, get_preference_matrix_2(points, "Circle", K)))

    # Clustering
    predicted_clusters = clustering(pref_mat, points, 0)  # criteria: 0 -> GRIC, 1 -> MDL, 2 -> GIC, 3 -> GMDL

    predicted_clusters = delete_outliers(predicted_clusters , 15)    # deletes clusters with n° of points below a threshold (2nd parameter)

    visualize_clusters(predicted_clusters , points)

    ##neds a data structure that sets all distances (truth is new distances have to be computed each time)

