import numpy as np
from matplotlib import pyplot as plt
from a3_utils import *
import cv2 as cv2
from UZ_utils import *
import math


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


def exercise1():
    print("Exercise 1")
    oneB()
    oneC()
    oneD()


def findedges(image, sigma, theta):
    image, _ = gradient_magnitude(image, sigma)
    image = np.where(image > theta, 1, 0)
    plt.imshow(image, cmap='gray')
    plt.title("Fine Edges, sigma = {}, theta = {}".format(sigma, theta))
    plt.show()
    return image


def twoA():
    print("Exercise 2A")
    thetas = np.arange(0, 0.5, 0.05)
    for theta in thetas:
        findedges(imread_gray("./images/museum.jpg"), 1, theta)

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


def twoB():
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

    image = imread_gray("./images/museum.jpg")
    I_mag, I_dir = gradient_magnitude(image, sigma=1)
    plt.imshow(I_mag, cmap='gray')
    plt.title("Magnitude")
    plt.show()

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

    # I_mag_copy = np.where(I_mag_copy > 0.16, 1, 0)
    return I_mag_copy
    # mybe 0.1 is a good value for theta


def twoC():
    t_low = 0.08
    t_high = 0.50
    image = np.where(twoB() < t_low, 0, 1)
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


def exercise2():
    print("Exercise 2")
    twoA()
    NMS = twoB()
    NMS = np.where(NMS > 0.16, 1, 0)
    plt.imshow(NMS, cmap='gray')
    plt.title("Non-Maxima Suppression, threshold = 0.16")
    plt.show()
    twoC()


def bb(x, y):
    # Create an accumulator array defined by the resolution on ro and theta values. Calculate
    # the sinusoid that represents all the lines that pass through some nonzero point.

    ro = x * np.cos(y) + y * np.sin(y)


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


def hough_find_lines(image, ro_num_of_bins, thetha_num_of_bins, threshold):
    """
Implement the function hough_find_lines that accepts a binary image, the number
# and  (allow the possibility of them being different) and a threshold.
of bins for
Create an accumulator matrix A for the parameter space (; #). Parameter # is
defined in the interval from 􀀀=2 to =2,  is defined on the interval from 􀀀D to
D, where D is the length of the image diagonal. For each nonzero pixel in the image,
generate a curve in the (; #) space by using the equation (7) for all possible values
of # and increase the corresponding cells in A. Display the accumulator matrix. Test
the method on your own synthetic images ((e.g. 100  100 black image, with two
white pixels at (10; 10) and (10; 20)).
Finally, test your function on two synthetic images oneline.png and rectangle.
png. First, you should obtain an edge map for each image using either your
function findedges or some inbuilt function. Run your implementation of the Hough
algorithm on the resulting edge maps.
    """

    diagonal = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    accumulator_matrix = np.zeros((ro_num_of_bins, thetha_num_of_bins))
    theta = np.linspace(-np.pi/2, np.pi/2, num=thetha_num_of_bins)
    rho_range = np.linspace(-diagonal, diagonal, num=ro_num_of_bins)

    nono_zero_indexes = np.where(image == 1)
    x_s, y_s = nono_zero_indexes[0], nono_zero_indexes[1]
    for x, y in zip(x_s, y_s):
        rohs = np.round(x * np.cos(theta) + y * np.sin(theta).astype(np.int64))
        binnes = np.digitize(rohs, rho_range)
        for j in range(thetha_num_of_bins):
            accumulator_matrix[binnes[j], j] += 1

    return accumulator_matrix


def hough_find_lines2(image,
                      n_bins_theta, n_bins_rho, treshold):
    """"
    Accepts: bw image with lines, n_bins_theta, n_bins_rho, treshold
    Returns: image points above treshold transformed into hough space
    """
    image = image.copy()
    image[image < treshold] = 0
    theta_values = np.linspace(-np.pi/2, np.pi/2, n_bins_theta)
    D = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
    rho_values = np.linspace(-D, D, n_bins_rho)
    accumulator = np.zeros((n_bins_rho, n_bins_theta), dtype=np.uint64)

    cos_precalculated = np.cos(theta_values)
    sin_precalculated = np.sin(theta_values)

    y_s, x_s = np.nonzero(image)

    # Loop through all nonzero pixels above treshold
    for i in range(len(y_s)):
        x = x_s[i]
        y = y_s[i]

        # Precalculate rhos
        rhos = np.round(x * cos_precalculated + y *
                        sin_precalculated).astype(np.int64)

        # Bin the rhos
        binned = np.digitize(rhos, rho_values) - 1

        # Add to accumulator
        for theta in range(n_bins_theta):
            accumulator[binned[theta], theta] += 1

    return accumulator


def threeB():
    print("Exercise 3B")
    images_names = ["./images/oneline.png", "./images/rectangle.png"]
    test_image = np.zeros((100, 100)).astype(np.uint8)
    test_image[10, 10] = 1
    test_image[10, 20] = 1
    acc_matrix = hough_find_lines2(test_image, 200, 200, 1)

    plt.imshow(acc_matrix)
    plt.show()

    for image_name in images_names:
        image = imread_gray(image_name)
        image = findedges(image, 1, 0.16)
        acc_matrix = hough_find_lines2(
            image, 200, 200, 1)
        plt.imshow(acc_matrix)
        plt.title(image_name)
        plt.show()


def non_maxima_box(image):
    # loop through image check 8 neigbourhoodge
    ima = image.copy()
    for i in range(1, len(image)-1):
        for j in range(1, len(image[0])-1):
            neigbours = image[-1+i:i+1, j-1:j+1]
            neigbours[1, 1] = 0
            if image[i, j] != np.max(neigbours):
                ima[i, j] == 0
    return image


def threeC():
    test_image = np.zeros((100, 100)).astype(np.uint8)
    test_image[10, 10] = 1
    test_image[10, 20] = 1
    acc_matrix = hough_find_lines(test_image, 200, 200, 1)
    res = non_maxima_box(acc_matrix)
    plt.imshow(res)
    plt.show()


def threeD():
    synthetic = np.zeros((100, 100)).astype(np.uint8)
    oneline = findedges(imread_gray("./images/oneline.png"), 1, 0.16)
    rectangle = findedges(imread_gray("./images/rectangle.png"), 1, 0.16)
    images = [synthetic, oneline, rectangle]

    for image in images:
        acc_matrix = hough_find_lines(image, 200, 200, 0.16)
        points = non_maxima_box(acc_matrix)
        points = np.where(points == 0, 0, 1)
        print(np.where(points == 1))


def exercise3():
    print("Exercise 3")
    # threeA()
    threeB()
    # threeC()
    threeD()


def main():
    print("Hello World!")
    # exercise1()
    # exercise2()
    exercise3()


if __name__ == "__main__":
    main()
