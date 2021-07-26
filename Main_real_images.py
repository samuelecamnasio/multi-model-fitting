from getPoints import *
from pref_matrix import *
from numpy import *
from clustering import *
from pointExtractor import *

if __name__ == "__main__":
    K = 3  # multiple of the sampling number
    # Extract edge points from a real image
    wrong_figure = True
    minVal = 300
    maxVal = 500
    while wrong_figure:
        points = point_from_image(400, "complex", "1", minVal, maxVal)
        response = input("Has the edge detection found the right edges with low noise? y/n ")
        if response == "y":
            wrong_figure = False
        elif response == "n":
            minVal = int(input("insert low threshold: "))
            maxVal = int(input("insert high threshold: "))
        else:
            print("Wrong input")

    # Compute the preference matrix for both lines and circles
    print("computing preference matrix for lines...")
    pref_mat = get_preference_matrix_2(points, "Line", K)
    print("computing preference matrix for circles...")
    pref_mat = hstack((pref_mat, get_preference_matrix_2(points, "Circle", K)))
    # Clustering
    predicted_clusters = clustering(pref_mat, points, 0)  # criteria: 0 -> GRIC, 1 -> MDL, 2 -> GIC, 3 -> GMDL

    # deletes clusters with n° of points below a threshold (2nd parameter)
    wrong_result = True
    temp_predicted_clusters = predicted_clusters
    cutoff_threshold = 25
    while wrong_result:
        temp_predicted_clusters = delete_outliers(temp_predicted_clusters, cutoff_threshold)
        visualize_clusters(predicted_clusters, points)
        response = input("Have the outliers been cleared? y/n ")
        if response == "y":
            wrong_result = False
        elif response == "n":
            cutoff_threshold = int(input("insert new cutoff threshold: "))
        else:
            print("Wrong input")

    predicted_clusters = temp_predicted_clusters






