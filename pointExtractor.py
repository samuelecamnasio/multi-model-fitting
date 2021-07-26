import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def point_from_image(basewidth, imgType, imgNumber, minVal, maxVal, verbose =False):

    cwd = os.getcwd()
    # read/load an image
    img_path = os.path.join(cwd, "resources\img_" + imgType + "_" + imgNumber)
    if imgType == "complex":
        img_path += ".jpg"
    else:
        img_path += ".png"
    image = cv2.imread(img_path)

    wpercent = (basewidth / float(image.shape[1]))
    hsize = int((float(image.shape[0]) * float(wpercent)))
    dim = (basewidth, hsize)

    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    if verbose:
        print(image.shape)

    # detection of the edges
    plt.cla()
    img_edge = cv2.Canny(image, minVal, maxVal, apertureSize = 5)
    plt.imshow(img_edge)
    plt.show()
    # TODO: maybe there are too many points taken from the edge detection
    # now trying to halve them (should still maintain enough points)
    ans = []
    for y in reversed(range(0, img_edge.shape[0])):
        for x in range(0, img_edge.shape[1]):
            if img_edge[y, x] != 0 and x % 2 == 0:
                ans = ans + [[x, img_edge.shape[0]-y]]
    ans = np.array(ans)

    if verbose:
        print(ans.shape)
        print(ans[0:10, :])
    s = [0.1 for n in range(ans.shape[0])]
    plt.scatter(ans[:, 0], ans[:, 1], s=s)

    return ans
