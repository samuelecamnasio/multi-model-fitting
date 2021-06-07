import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def point_from_image(basewidth, imgType, imgNumber):

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
    print(image.shape)

    # detection of the edges
    img_edge = cv2.Canny(image,200,600,apertureSize = 5)

    ans = []
    for y in range(0, img_edge.shape[0]):
        for x in range(0, img_edge.shape[1]):
            if img_edge[y, x] != 0:
                ans = ans + [[x, y]]
    ans = np.array(ans)

    print(ans.shape)
    print(ans[0:10, :])
    s=[0.1 for n in range(ans.shape[0])]
    plt.scatter(ans[:,0],ans[:,1],s=s)

    return ans
