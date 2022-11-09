import numpy as np
from matplotlib import pyplot as plt
from a3_utils import *
import cv2 as cv2
from UZ_utils import *


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
   # order of convolution is important
    g_gt = cv2.filter2D(impulse, -1, gaus)
    g_gt = cv2.filter2D(g_gt, -1, gaus.T)

    g_dt = cv2.filter2D(impulse, -1, gaus)
    g_dt = cv2.filter2D(g_dt, -1, gaus_dx.T)

    d_gt = cv2.filter2D(impulse, -1, gaus_dx)
    d_gt = cv2.filter2D(d_gt, -1, gaus.T)

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


def oneD():
    print("Exercise 1D")


def exercise1():
    print("Exercise 1")
    # oneB()
    # oneC()
    oneD()


def main():
    print("Hello World!")
    exercise1()


if __name__ == "__main__":
    main()
