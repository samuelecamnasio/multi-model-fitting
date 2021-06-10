import math
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generate_points(model = 1, noise = 0.5):
    real_clusters=[]
    # -------------------------------------------------------------------------------
    # Generate points on inner circle
    # -------------------------------------------------------------------------------
    ang_circ_in = np.linspace(0, 2 * math.pi, 100)  ## linspace(start, end, number)
    C_in = [0, 0]
    r_in = 30
    x_circ_in = r_in * np.cos(ang_circ_in) + C_in[0]
    y_circ_in = r_in * np.sin(ang_circ_in) + C_in[1]

    circ_ideal_in = np.column_stack((x_circ_in, y_circ_in))
    circ_in = circ_ideal_in
    circ_in += np.random.normal(0, noise, circ_in.shape)
    # -------------------------------------------------------------------------------
    # Generate points on outer circle
    # -------------------------------------------------------------------------------
    ang_circ_out = np.linspace(0, 2 * math.pi, 100)  ## linspace(start, end, number)
    C_out = [0, 0]
    r_out = 50
    x_circ_out = r_out * np.cos(ang_circ_out) + C_out[0]
    y_circ_out = r_out * np.sin(ang_circ_out) + C_out[1]

    circ_ideal_out = np.column_stack((x_circ_out, y_circ_out))
    circ_out = circ_ideal_out
    circ_out += np.random.normal(0, noise, circ_out.shape)

    # -------------------------------------------------------------------------------
    # Generate points on lines
    # -------------------------------------------------------------------------------
    m_ideal1 = 1
    q_ideal1 = 45 + C_out[1]

    x1 = np.linspace(-45, 0, 30)
    y1 = m_ideal1 * x1 + q_ideal1
    line_ideal1 = np.column_stack((x1, y1))
    line1 = line_ideal1
    line1 += np.random.normal(0, noise, line1.shape)

    m_ideal2 = -1
    q_ideal2 = 45 + C_out[1]

    x2 = np.linspace(0, 45, 30)
    y2 = m_ideal2 * x2 + q_ideal2
    line_ideal2 = np.column_stack((x2, y2))
    line2 = line_ideal2
    line2 += np.random.normal(0, noise, line2.shape)

    m_ideal3 = 1
    q_ideal3 = -45 + C_out[1]

    x3 = np.linspace(0, 45, 30)
    y3 = m_ideal3 * x3 + q_ideal3
    line_ideal3 = np.column_stack((x3, y3))
    line3 = line_ideal3
    line3 += np.random.normal(0, noise, line3.shape)

    m_ideal4 = -1
    q_ideal4 = -45 + C_out[1]

    x4 = np.linspace(-45, 0, 30)
    y4 = m_ideal4 * x4 + q_ideal4
    line_ideal4 = np.column_stack((x4, y4))
    line4 = line_ideal4
    line4 += np.random.normal(0, noise, line4.shape)

    # add random noise
    random_noise = np.random.uniform(-100, 100, (100, 2))

    # -------------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------------
    plt.scatter(x1, y1)
    plt.scatter(line1[:, 0], line1[:, 1])

    plt.scatter(x2, y2)
    plt.scatter(line2[:, 0], line2[:, 1])

    plt.scatter(x3, y3)
    plt.scatter(line3[:, 0], line3[:, 1])

    plt.scatter(x4, y4)
    plt.scatter(line4[:, 0], line4[:, 1])

    plt.title('Generated Points')

    plt.scatter(x_circ_in, y_circ_in)
    plt.scatter(circ_in[:, 0], circ_in[:, 1])

    plt.scatter(x_circ_out, y_circ_out)
    plt.scatter(circ_out[:, 0], circ_out[:, 1])

    plt.scatter(x_circ_in, y_circ_in)
    plt.scatter(circ_in[:, 0], circ_in[:, 1])

    plt.scatter(random_noise[:, 0], random_noise[:, 1])
    plt.show()

    if model == 1:  # 4 lines 2 circles
        figure = line1
        figure = np.vstack((figure, line2))
        figure = np.vstack((figure, line3))
        figure = np.vstack((figure, line4))
        figure = np.vstack((figure, circ_in))
        figure = np.vstack((figure, circ_out))
        figure = np.vstack((figure, random_noise))

    elif model == 2:

        figure = line4
        figure = np.vstack((figure, circ_in))
        figure = np.vstack((figure, random_noise))

    else:
        figure = line4
        figure = np.vstack((figure, random_noise))

    real_clusters=real_cluster_composition(model, line1, line2, line3, line4, circ_in, circ_out)

    return figure, real_clusters


def visualize_clusters(clusters, points):
    plt.figure(2)
    plt.title('Clusters')

    for cluster in clusters:
        print("N° of points: " + str(len(cluster)))
        color=[(rand.random(),rand.random(),rand.random())]
        rand.random()
        for index in cluster:
            #print("caio: "+str(points[index][0]))
            plt.scatter(points[index][0], points[index][1], color=color)

    plt.show()


def show_pref_matrix(pref_m, label_k):
    fig, ax = plt.subplots(figsize=(5, 1.5))
    matr = ax.imshow(pref_m, cmap='Blues', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(matr, cax=cax)
    fig.suptitle(label_k)
    fig.tight_layout()
    plt.show()

def real_cluster_composition(model, line1, line2, line3, line4, circ_in, circ_out):
    real_clusters=[]
    nc = []
    k = 0
    if model == 1:  # 4 lines 2 circles
        for i in range(k, line1.shape[0]):
            nc.append(i)
        real_clusters.append(nc)
        nc = []
        k += line1.shape[0]
        for i in range(k, k + line2.shape[0]):
            nc.append(i)
        real_clusters.append(nc)
        nc = []
        k += line2.shape[0]
        for i in range(k, k + line3.shape[0]):
            nc.append(i)
        real_clusters.append(nc)
        nc = []
        k += line3.shape[0]
        for i in range(k, k + line4.shape[0]):
            nc.append(i)
        real_clusters.append(nc)
        nc = []
        k += line4.shape[0]
        for i in range(k, k + circ_in.shape[0]):
            nc.append(i)
        real_clusters.append(nc)
        nc = []
        k += circ_in.shape[0]
        for i in range(k, k + circ_out.shape[0]):
            nc.append(i)
        real_clusters.append(nc)

    elif model == 2:
        for i in range(k, line4.shape[0]):
            nc.append(i)
        real_clusters.append(nc)
        nc = []
        k += line4.shape[0]
        for i in range(k, k + circ_in.shape[0]):
            nc.append(i)
        real_clusters.append(nc)

    else:
        for i in range(k, line4.shape[0]):
            nc.append(i)
        real_clusters.append(nc)

    return real_clusters