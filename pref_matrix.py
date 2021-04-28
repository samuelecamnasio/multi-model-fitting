import scipy.io
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from Cluster import Cluster

def distance_from_line(p1,p2,p3):   #calculates the normal distance between a point p3 and a line passing through p1 and p2
    return (np.linalg.norm(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)) == 0

def get_preference_matrix(points):
    pref_mat = np.zeros((150, 15))
    for i in range(0, len(points) - 1, 10):  # iterates all the models
        p1 = points[i]
        p2 = points[i + 1]
        if (p1 == p2).all():  # checks if two points are equal to avoid the division by 0
            p2 = points[i + 2]
        i = int(i / 10)
        for k in range(0, len(points)):  # iterates all the points
            p3 = points[k]
            pref_mat[k][i] = distance_from_line(p1, p2, p3)  # populates the preference matrix

    #print(pref_mat[:, 14])
    return pref_mat

def Gric(cluster, model):   #model_dimension = 2 for lines, = 3 for circumferences
    g=0
    rho=rho_calculation(cluster, model)
    sigma=1 # sigma è un multiplo della deviazione standard del rumore sui dati

    lambda1=1 #paper multilink, pag.6 (row 555/556)
    lambda2=2

    d=1 # number of dimensions modeled (d=3 -> fund. matrix, d=2 -> homography, d=1 -> lines, circumferences)
    u=2 #number of model paramters (u=2 for lines, u=3 for circumferences)

    for k in range(0, len(cluster.points) - 1):
        if(len(model)==2):
            err=distance_from_line(model[0], model[1], cluster.points[k])
        elif(len(model)==3):
            err = distance_from_circ(model[0], model[1], cluster.points[k])
       g+= rho[k]*(err/sigma)^2+lambda1*d*len(cluster)+lambda2*u

    return g

def sigma_calculation(cluster, model):
    return 0

def rho_calculation(cluster, model):    # ATM: binary, equals 1 for inliers (residuals < epsilon) and 0 for outliers. Should be done with M-estimators

    rho=np.zeros((1, len(cluster.points)))
    for k in range(0, len(cluster.points)-1):  # iterates all the points
        rho[k]= distance_from_line(model[0], model[1], cluster.points[k])  # populates the preference matrix

    return rho



# the .mat file is structured with 150 couples of points where from 10 to 10 they belong to the same line
mat = scipy.io.loadmat('punti_prova.mat')   # loads the .mat containing the points
mat=mat['A']


prova=Cluster(mat[0:5],1000, "line")

print("Cluster prova: "+str(prova.points))

a=rho_calculation(1)

print("Rho: "+str(a))



