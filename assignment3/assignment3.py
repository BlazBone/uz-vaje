import numpy as np
from matplotlib import pyplot as plt
from a3_utils import *
import cv2 as cv2
from UZ_utils import *
import math
import os


def gauss_kernel(sigma):
    sigma3 = np.ceil(3*sigma).astype(int)
    kernel = np.arange(-sigma3, sigma3+1, 1).astype(float)
    kernel = np.exp(-(np.square(kernel))/(2*np.square(sigma)))
    return kernel/np.sum(kernel)

# derivative of gaussian


def gaussdx(sigma):
    sigma3 = np.ceil(3*sigma).astype(int)
    kernel = np.arange(-sigma3, sigma3+1, 1).astype(float)

    kernel = -(kernel/(np.square(sigma))) * \
        np.exp(-(np.square(kernel))/(2*np.square(sigma)))
    return kernel/np.sum(np.abs(kernel))


def oneB():
    print("Exercise 1A")
    print(np.exp(2))
    print(np.sum(np.abs(gaussdx(1))))
    print(np.sum(np.abs(gaussdx(20))))

    plt.plot(gaussdx(1))
    plt.show()

    plt.plot(gaussdx(20))
    plt.show()

    plt.plot(gauss_kernel(1))
    plt.show()

    plt.plot(gauss_kernel(20))
    plt.show()


def oneC():
    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1
    gaus = np.array([gauss_kernel(4)])
    gaus_dx = np.array([gaussdx(4)])
    # gaus is symetric, derivetivce is not
    gaus_dx = np.flip(gaus_dx)

    g_gt = cv2.filter2D(impulse, -1, gaus)
    g_gt = cv2.filter2D(g_gt, -1, gaus.T)

    g_dt = cv2.filter2D(impulse, -1, gaus)
    g_dt = cv2.filter2D(g_dt, -1, gaus_dx.T)

    # order of convolution is not important
    d_gt = cv2.filter2D(impulse, -1, gaus.T)
    d_gt = cv2.filter2D(d_gt, -1, gaus_dx)

    gt_d = cv2.filter2D(impulse, -1, gaus.T)
    gt_d = cv2.filter2D(gt_d, -1, gaus_dx)

    dt_g = cv2.filter2D(impulse, -1, gaus_dx.T)
    dt_g = cv2.filter2D(dt_g, -1, gaus)

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(impulse, cmap='gray')
    ax[0, 0].set_title("Impulse")

    ax[0, 1].imshow(g_dt, cmap='gray')
    ax[0, 1].set_title("Gauss then Derivative T")

    ax[1, 0].imshow(g_gt, cmap='gray')
    ax[1, 0].set_title("Gauss then Gauss T")

    ax[1, 1].imshow(gt_d, cmap='gray')
    ax[1, 1].set_title("Gauss T then Derivative")

    ax[0, 2].imshow(d_gt, cmap='gray')
    ax[0, 2].set_title("Derivative then Gauss T")

    ax[1, 2].imshow(dt_g, cmap='gray')
    ax[1, 2].set_title("Derivative T then Gauss")

    plt.show()


def get_image_derivetive(image, sigma):
    gaus = np.array([gauss_kernel(sigma)])

    gaus_dx = np.array([gaussdx(sigma)])
    gaus_dx = np.flip(gaus_dx)

    I_x = cv2.filter2D(image, -1, gaus.T)
    I_x = cv2.filter2D(I_x, -1, gaus_dx)

    I_y = cv2.filter2D(image, -1, gaus)
    I_y = cv2.filter2D(I_y, -1, gaus_dx.T)

    return I_x, I_y


def get_second_image_derivative(image, sigma):
    gaus = np.array([gauss_kernel(sigma)])
    gaus_dx = np.array([gaussdx(sigma)])
    gaus_dx = np.flip(gaus_dx)

    I_x, I_y = get_image_derivetive(image, sigma)

    I_xx = cv2.filter2D(I_x, -1, gaus.T)
    I_xx = cv2.filter2D(I_xx, -1, gaus_dx)

    I_yy = cv2.filter2D(I_y, -1, gaus)
    I_yy = cv2.filter2D(I_yy, -1, gaus_dx.T)

    I_xy = cv2.filter2D(I_x, -1, gaus)
    I_xy = cv2.filter2D(I_xy, -1, gaus_dx.T)

    return I_xx, I_yy, I_xy


def gradient_magnitude(image, sigma):
    I_x, I_y = get_image_derivetive(image, sigma)
    I_mag = np.sqrt(np.square(I_x) + np.square(I_y))
    I_dir = np.arctan2(I_y, I_x)
    return I_mag, I_dir


def oneD(image_path="./images/museum.jpg"):
    image = imread_gray(image_path)
    sigma = 1

    I_x, I_y = get_image_derivetive(image, sigma=sigma)
    I_mag, I_dir = gradient_magnitude(image, sigma=sigma)
    I_xx, I_yy, I_xy = get_second_image_derivative(image, sigma=sigma)

    fig, ax = plt.subplots(2, 4)
    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title("Original")

    ax[0, 1].imshow(I_x, cmap='gray')
    ax[0, 1].set_title("I_x")

    ax[0, 2].imshow(I_y, cmap='gray')
    ax[0, 2].set_title("I_y")

    ax[0, 3].imshow(I_mag, cmap='gray')
    ax[0, 3].set_title("I_mag")

    ax[1, 0].imshow(I_xx, cmap='gray')
    ax[1, 0].set_title("I_xx")

    ax[1, 1].imshow(I_xy, cmap='gray')
    ax[1, 1].set_title("I_xy")

    ax[1, 2].imshow(I_yy, cmap='gray')
    ax[1, 2].set_title("I_yy")

    ax[1, 3].imshow(I_dir, cmap='gray')
    ax[1, 3].set_title("I_dir")

    plt.show()
    print("Exercise 1D")


def euclidean_distance(x, y):
    return np.sqrt((np.sum(np.square(x - y))))


def chi_square(x, y):
    return 0.5 * np.sum(np.square(x - y) / (x + y + 1e-6))


def intersection(x, y):
    return 1 - np.sum(np.minimum(x, y))


def helliger(x, y):
    return np.sqrt(0.5 * np.sum(np.square(np.sqrt(x) - np.sqrt(y))))

# https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def gradiant_magnitude_hist(image):
    # split image in 8 block for some reason
    I_mag, I_dir = gradient_magnitude(image, sigma=1)
    magBlock = blockshaped(I_mag, 8, 8)
    dirBlock = blockshaped(I_dir, 8, 8)

    dirRAnge = np.linspace(-np.pi, np.pi, 8)  # 8 sosedi

    hist = []
    # walk throuigh every block
    for i in range(magBlock.shape[0]):
        temp_mag = magBlock[i]
        temp_dir = dirBlock[i]

        temp_hist = np.zeros(8)
        # bin direction
        dir_bin = np.digitize(temp_dir, dirRAnge)
        for j in range(8):
            temp_hist[dir_bin[j] - 1] += temp_mag[j]
            # da z druge
        hist.extend(temp_hist/np.sum(temp_hist))
    return hist/np.sum(hist)


def oneE():
    # simmilar to assigment 2
    # Could be  fucntion parameters
    # its nicer this way in a notebook
    ##############################
    directory = "./dataset/"
    ##############################

    histograms = dict()
    for filename in os.listdir(directory):
        # color doesnt matter
        I_temp = imread_gray(directory + filename)
        histograms[filename] = gradiant_magnitude_hist(I_temp)

    # image_name, hist = random.choice(list(histograms.items()))
    image_name = "object_05_4.png"
    hist = histograms[image_name]

    l2 = []
    hell = []
    inter = []
    chi = []

    for k, v in histograms.items():
        l2.append((k, euclidean_distance(hist, v)))
        hell.append((k, helliger(hist, v)))
        inter.append((k, intersection(hist, v)))
        chi.append((k, chi_square(hist, v)))

    chi.sort(key=lambda x: x[1])
    hell.sort(key=lambda x: x[1])
    l2.sort(key=lambda x: x[1])
    inter.sort(key=lambda x: x[1])

    chi = chi[:6]
    hell = hell[:6]
    l2 = l2[:6]
    inter = inter[:6]

    every_distance = [(chi, "chi"), (hell, "hell"),
                      (l2, "l2"), (inter, "inter")]

    for i, (distance, name_of_distance) in enumerate(every_distance):
        a, two = plt.subplots(2, 6)
        a.suptitle(name_of_distance)

        for j, (name, hist) in enumerate(distance):
            two[0, j].imshow(imread(directory + name))
            two[0, j].set(title=name[5:])

            two[1, j].bar(height=histograms[name], x=range(
                len(histograms[name])), width=5)
            two[1, j].set(title=str(hist)[:5])

    plt.show()


def findedges(image, sigma, theta):
    image, _ = gradient_magnitude(image, sigma)
    image = np.where(image > theta, 1, 0)
    return image


def twoA():
    print("Exercise 2A")
    thetas = np.arange(0.02, 0.3, 0.05)
    for theta in thetas:
        im = findedges(imread_gray("./images/museum.jpg"), 1, theta)
        plt.imshow(im, cmap='gray')
        plt.title("theta = " + str(theta))
        plt.show()
    # mybe 0.1 is a good value for theta


def closest_side(angle):
    sides = [-math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi /
             4, 0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]
    return min(sides, key=lambda x: abs(x - angle))


def myOtsu(I):
    # max threshold is max value in image (B&W) since we pass in normalized images we just expand them
    max_val = max(I.reshape(-1))
    # we could make algo check more threshold
    possible_threshholds = np.arange(0, max_val, 0.005)

    # for first iteration
    min_var = -1
    min_var_threshold = -1

    for thresh in possible_threshholds:
        # split image into two parts on threshold
        foreground = I[I > thresh]
        weight_foreground = len(foreground) / len(I.reshape(-1))
        variance_foreground = np.var(foreground)

        background = I[I <= thresh]
        weight_background = len(background) / len(I.reshape(-1))
        variance_background = np.var(background)

        # calculate variance
        variance = weight_foreground * variance_foreground + \
            weight_background * variance_background

        # save if variance is smaller
        if min_var == -1 or variance < min_var:
            min_var = variance
            min_var_threshold = thresh

    return min_var_threshold


def twoB(image):
    print("Exercise 2B")
    """
    Using magnitude produces only a first approximation of detected edges. Unfortunately,
    these are often wide and we would like to only return edges one pixel wide.
    Therefore, you will implement non-maxima suppression based on the image derivative
    magnitudes and angles. Iterate through all the pixels and for each search its
    8-neighborhood. Check the neighboring pixels parallel to the gradient direction and
    set the current pixel to 0 if it is not the largest in the neighborhood (based on
    derivative magnitude). You only need to compute the comparison to actual pixels,
    interpolating to more accuracy is not required.
    """

    I_mag, I_dir = gradient_magnitude(image, sigma=1)
    I_mag_copy = np.copy(I_mag)
    # walk through all pixels with indexes

    for i in range(1, I_mag.shape[0] - 1):
        for j in range(1, I_mag.shape[1] - 1):
            # get the angle of the pixel
            angle = I_dir[i, j]
            # get the closest side
            # get the magnitude of the pixel
            mag = I_mag[i, j]
            # get the magnitude of the pixels in the direction of the side
            if (np.pi / 8 <= angle <= 3 * np.pi / 8) or (-7 * np.pi / 8 <= angle <= -5 * np.pi / 8):
                mag1 = I_mag[i - 1][j - 1]
                mag2 = I_mag[i + 1][j + 1]

            elif (3 * np.pi / 8 <= angle <= 5 * np.pi / 8) or (-5 * np.pi / 8 <= angle <= -3 * np.pi / 8):
                mag1 = I_mag[i - 1][j]
                mag2 = I_mag[i + 1][j]
            elif (5 * np.pi / 8 <= angle <= 7 * np.pi / 8) or (-3 * np.pi / 8 <= angle <= -np.pi / 8):
                mag1 = I_mag[i - 1][j + 1]
                mag2 = I_mag[i + 1][j - 1]
            else:
                mag1 = I_mag[i][j - 1]
                mag2 = I_mag[i][j + 1]

            if mag < mag1 or mag < mag2:
                I_mag_copy[i, j] = 0
            # if the magnitude of the pixel is not the largest, set it to 0

    return I_mag_copy


def twoC():
    t_low = 0.08
    t_high = 0.50
    image = np.where(twoB(imread_gray("./images/museum.jpg")) < t_low, 0, 1)
    image = image.astype(np.uint8)
    test, labels, _, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8, ltype=cv2.CV_32S)

    image.any()
    for i in range(test):
        if((np.greater(image[labels == i], t_high).any())):
            image[labels == i] = 1
        else:
            image[labels == i] = 0

    plt.imshow(image, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.show()


def hugh(x, y):
    image = np.zeros((300, 300))
    thetas = np.linspace(-np.pi/2, np.pi, num=300)
    rohs = x * np.cos(thetas) + y * np.sin(thetas) + 150

    for i in range(300):
        image[int(rohs[i]), i] += 1
    return image


def threeA():
    print("Exercise 3A")
    a = hugh(10, 10)
    b = hugh(30, 60)
    c = hugh(50, 20)
    d = hugh(80, 90)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Hough Transform')
    ax[0, 0].imshow(a, cmap='gray')
    ax[0, 0].set_title("10, 10")
    ax[0, 1].imshow(b, cmap='gray')
    ax[0, 1].set_title("30, 60")
    ax[1, 0].imshow(c, cmap='gray')
    ax[1, 0].set_title("50, 20")
    ax[1, 1].imshow(d, cmap='gray')
    ax[1, 1].set_title("80, 90")
    plt.show()


def hough_find_lines(image,  thetha_num_of_bins, ro_num_of_bins, threshold):
    # treshold
    image = np.where(image < threshold, 0, image)
    # prepare values
    diagonal = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    accMatrix = np.zeros(
        (ro_num_of_bins, thetha_num_of_bins), dtype=np.uint64)
    theta = np.linspace(-np.pi/2, np.pi/2, num=thetha_num_of_bins)
    rho_range = np.linspace(-diagonal, diagonal, num=ro_num_of_bins)
    # get indexes of non zero pixels
    y_s, x_s = np.nonzero(image)
    # hvala prijatelj za speed hack
    # update still slowaf
    cos = np.cos(theta)
    sin = np.sin(theta)

    for x, y in zip(x_s, y_s):
        rohs = np.round(x * cos + y * sin).astype(np.int32)
        binnes = np.digitize(rohs, rho_range) - 1  # outofv

        for j in range(thetha_num_of_bins):
            accMatrix[binnes[j], j] += 1

    return accMatrix


def threeB():
    print("Exercise 3B")
    images_names = ["./images/oneline.png", "./images/rectangle.png"]
    test_image = np.zeros((100, 100)).astype(np.uint8)
    test_image[10, 10] = 1
    test_image[10, 20] = 1
    acc_matrix = hough_find_lines(test_image, 200, 200, 0.16)

    plt.imshow(acc_matrix)
    plt.show()

    for image_name in images_names:
        image = imread_gray(image_name)
        image = findedges(image, 1, 0.16)
        acc_matrix = hough_find_lines(
            image, 200, 200, 0.16)
        plt.imshow(acc_matrix)
        plt.title(image_name)
        plt.show()


def non_maxima_box(image):
    for i in range(1, len(image)-1):
        for j in range(1, len(image[0])-1):
            neigbours = image[-1+i:i+2, j-1:j+2]
            if image[i, j] < np.max(neigbours):
                image[i, j] = 0
    return image


def threeC():
    test_image = np.zeros((100, 100)).astype(np.uint8)
    test_image[10, 10] = 1
    test_image[10, 20] = 1
    acc_matrix = hough_find_lines(test_image, 200, 200, 0.16)
    res = non_maxima_box(acc_matrix)
    plt.imshow(res)
    plt.title("non maxima box")
    plt.show()


def get_pairs(image, hugh, t_bins, r_bins):
    diagonal = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    theta = np.linspace(-np.pi/2, np.pi/2, num=t_bins)
    rho_range = np.linspace(-diagonal, diagonal, num=r_bins)

    y_s, x_s = np.nonzero(hugh)

    twos = []
    for x, y in zip(x_s, y_s):
        t = theta[x]
        r = rho_range[y]
        twos.append((t, r))
    return twos


def threeD():
    synthetic = np.zeros((100, 100)).astype(np.uint8)
    synthetic[10, 10] = 1
    synthetic[10, 20] = 1
    oneline = cv2.Canny(cv2.imread("./images/oneline.png",
                                   cv2.IMREAD_GRAYSCALE), 0.16, 1)
    rectangle = cv2.Canny(cv2.imread(
        "./images/rectangle.png", cv2.IMREAD_GRAYSCALE), 0.16, 1)
    images = [synthetic, oneline, rectangle]

    for image in images:
        acc_matrix = hough_find_lines(image, 200, 200, 0.4)
        points = non_maxima_box(acc_matrix)
        plt.imshow(points)
        plt.show()
        points = np.where(points > np.max(points)*0.3, 1, 0)
        twos = get_pairs(image, points, 200, 200)
        plt.imshow(image)
        for t, r in twos:
            draw_line(r, t, image.shape[0], image.shape[1])
        plt.show()


def getNbestParirs(image, hugh, t_bins, r_bins, n):
    np_twos = np.array(get_pairs(image, hugh, t_bins, r_bins))
    argpar = np.argpartition(np_twos, len(np_twos)-n-1, 0)[-n:]

    np_twos = np_twos[argpar]


def threeE():
    print("Exercise 3E")
    brick_clr = imread("./images/bricks.jpg")
    pier_clr = imread("./images/pier.jpg")
    brick = imread_gray("./images/bricks.jpg")
    pier = imread_gray("./images/pier.jpg")

    brick_edge = findedges(brick, 1, 0.16)
    pier_edge = findedges(pier, 1, 0.06)

    brick_hugh_matrix = hough_find_lines(brick_edge, 200, 200, 0.16)
    pier_hugh_matrix = hough_find_lines(pier_edge, 360, 360, 0.16)

    brick_hugh_nonmaxima = non_maxima_box(brick_hugh_matrix)
    plt.imshow(brick_hugh_matrix)
    plt.show()
    plt.imshow(pier_hugh_matrix)
    plt.show()
    pier_hugh_nonmaxima = non_maxima_box(pier_hugh_matrix)

    brick_hug_10 = np.zeros_like(brick_hugh_nonmaxima)
    pier_hug_10 = np.zeros_like(pier_hugh_nonmaxima)

    best_points_brick = []
    best_points_pier = []

    for i in range(10):
        best_points_brick.append(np.unravel_index(
            np.argmax(brick_hugh_matrix), brick_hugh_matrix.shape))
        brick_hugh_matrix[best_points_brick[-1]] = 0
        brick_hug_10[best_points_brick[-1]] = 1
        best_points_pier.append(np.unravel_index(
            np.argmax(pier_hugh_matrix), pier_hugh_matrix.shape))
        pier_hugh_matrix[best_points_pier[-1]] = 0
        pier_hug_10[best_points_pier[-1]] = 1

    a = get_pairs(brick, brick_hug_10, 360, 360)
    plt.imshow(brick_clr)

    for t, r in a:
        draw_line(r, t, brick_clr.shape[0], brick_clr.shape[1])
    plt.show()

    plt.imshow(pier_clr)
    for t, r in get_pairs(pier, pier_hug_10, 360, 360):
        draw_line(r, t, pier_clr.shape[0], pier_clr.shape[1])
    plt.show()
    print("hola amigos")


def exercise1():
    print("Exercise 1")
    oneB()
    oneC()
    oneD()
    oneE()


def exercise2():
    print("Exercise 2")
    twoA()
    NMS = twoB(imread_gray("./images/museum.jpg"))
    NMS = np.where(NMS > 0.16, 1, 0)
    plt.imshow(NMS, cmap='gray')
    plt.title("Non-Maxima Suppression, threshold = 0.16")
    plt.show()
    twoC()


def exercise3():
    print("Exercise 3")
    threeA()
    threeB()
    threeC()
    threeD()
    threeE()


def main():
    print("Hello World!")
    exercise1()
    exercise2()
    exercise3()


if __name__ == "__main__":
    main()
