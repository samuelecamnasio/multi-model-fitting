import math
import matplotlib.pyplot as plt
import numpy as np


def generate_points():
    #-------------------------------------------------------------------------------
    # Generate points on inner circle
    #-------------------------------------------------------------------------------
    ang_circ_in = np.linspace(0, 2*math.pi,300) ## linspace(start, end, number)
    C_in = [0,0]
    r_in = 30
    x_circ_in = r_in * np.cos(ang_circ_in) + C_in[0]
    y_circ_in = r_in*np.sin(ang_circ_in) + C_in[1]

    circ_ideal_in = np.column_stack((x_circ_in,y_circ_in))
    circ_in = circ_ideal_in
    circ_in += np.random.normal(0, 2.4, circ_in.shape)
    #-------------------------------------------------------------------------------
    # Generate points on outer circle
    #-------------------------------------------------------------------------------
    ang_circ_out = np.linspace(0, 2*math.pi,300) ## linspace(start, end, number)
    C_out = [0,0]
    r_out = 50
    x_circ_out = r_out * np.cos(ang_circ_out) + C_out[0]
    y_circ_out = r_out * np.sin(ang_circ_out) + C_out[1]

    circ_ideal_out = np.column_stack((x_circ_out,y_circ_out))
    circ_out = circ_ideal_out
    circ_out += np.random.normal(0, 2.4, circ_out.shape)

    #-------------------------------------------------------------------------------
    # Generate points on lines
    #-------------------------------------------------------------------------------
    m_ideal1 = 1
    q_ideal1 = 45 + C_out[1]

    x1 = np.linspace(-45, 0, 30)
    y1 = m_ideal1*x1 + q_ideal1
    line_ideal1 = np.column_stack((x1,y1))
    line1 = line_ideal1
    line1 += np.random.normal(0, 2.4, line1.shape)

    m_ideal2 = -1
    q_ideal2 = 45 + C_out[1]

    x2 = np.linspace(0, 45, 30)
    y2 = m_ideal2*x2 + q_ideal2
    line_ideal2 = np.column_stack((x2,y2))
    line2 = line_ideal2
    line2 += np.random.normal(0, 2.4, line2.shape)

    m_ideal3 = 1
    q_ideal3 = -45 + C_out[1]

    x3 = np.linspace(0, 45, 30)
    y3 = m_ideal3*x3 + q_ideal3
    line_ideal3 = np.column_stack((x3,y3))
    line3 = line_ideal3
    line3 += np.random.normal(0, 2.4, line3.shape)

    m_ideal4 = -1
    q_ideal4 = -45 + C_out[1]

    x4 = np.linspace(-45, 0, 30)
    y4 = m_ideal4*x4 + q_ideal4
    line_ideal4 = np.column_stack((x4,y4))
    line4 = line_ideal4
    line4 += np.random.normal(0, 2.4, line4.shape)

    #add random noise
    random_noise=np.random.uniform(-100,100,(50,2))

    #-------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------
    plt.scatter(x1, y1)
    plt.scatter(line1[:, 0], line1[:,1])

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

    figure = line1
    figure = np.vstack((figure, line2))
    figure = np.vstack((figure, line3))
    figure = np.vstack((figure, line4))
    figure = np.vstack((figure, circ_in))
    figure = np.vstack((figure, circ_out))
    figure = np.vstack((figure, random_noise))
    return figure