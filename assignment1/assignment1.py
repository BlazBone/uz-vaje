from hashlib import new
from operator import invert
from random import random
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt


def firstA():
    """
    Read the image from the file umbrellas.jpg and display it using the following
    snippet:
    """
    I = imread("./images/umbrellas.jpg")
    imshow(I)
    return I


def firstB():
    """
    Convert the loaded image to grayscale.
    """
    I = firstA()
    red = I[:, :, 0]
    green = I[:, :, 1]
    blue = I[:, :, 2]

    together = (red + green + blue) / 3
    imshow(together)
    return together


def firstC():
    I = imread("./images/umbrellas.jpg")
    # Cut and display a specific part of the loaded image.
    cutout = I[130:260, 240:450, 1]
    imshow(cutout)
    cutout = I[130:260, 240:450, 0]
    imshow(cutout)
    cutout = I[130:260, 240:450, 2]
    imshow(cutout)
    return cutout


def firstD():

    I = imread("./images/umbrellas.jpg")
    invers = I.copy()
    invers[130:260, 240:450, :] = np.ones_like(
        invers[130:260, 240:450, :]) - invers[130:260, 240:450, :]

    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(invers)
    plt.title('inverted')

    plt.show()


def firstE():
    I = imread_gray("./images/umbrellas.jpg")
# already grayscale and floating point
    ICpy = (I.copy() * 63).astype(np.uint8)

    print(ICpy)
    plt.subplot(1, 2, 1)
    plt.imshow(ICpy, vmax=255, cmap='gray')
    plt.title('uint8')

    plt.subplot(1, 2, 2)
    plt.imshow(I, cmap='gray')
    plt.title('float64')
    plt.show()


def first():
    firstA()
    firstB()
    firstC()
    firstD()
    firstE()


def secondA():
    I = imread_gray("./images/bird.jpg")
    mask = I.copy()
    threshold = 0.2
    mask[I < threshold] = 0
    mask[I > threshold] = 1
    plot2(I, 'original', mask, 'mask')
    i = myhist1(I, 255)
    plt.plot(i)
    plt.show()
    print(sum(i))
    print(len(i))
    i = myhist1(I, 10)
    print(sum(i))
    print(len(i))

    plt.plot(i)
    plt.show()
    i = myhist2(I, 25)
    plt.plot(i)
    plt.show()
    print(sum(i))
    print(len(i))

    i = myhist2(I, 255)
    plt.plot(i)
    plt.show()
    print(sum(i))
    print(len(i))


def diffBetweenMyHist():
    print('hej')
    I1 = imread_gray("./images/bird.jpg")
    I2 = imread_gray("./images/coins.jpg")
    I3 = imread_gray("./images/eagle.jpg")
    I4 = imread_gray("./images/umbrellas.jpg")

    images = [I1, I2, I3, I4]

    for image in images:
        rand = 50
        h1 = myhist1(image, rand)
        h2 = myhist2(image, rand)

        plt.plot(h1, label='myhist1')
        plt.plot(h2, label='myhist2')
        plt.show()


def myhist2(I, numofbins):
    I = I.reshape(-1)
    max_v = max(I)
    min_v = min(I)
    interval = (max_v - min_v) / numofbins
    hist = [0 for i in range(numofbins)]
    for i in I:
        if i == max_v:
            hist[-1] += 1
        else:
            hist[int((i - min_v) // interval)] += 1
    return hist


def myhist1(I, numofbins):
    I = I.reshape(-1)
    I = I*255
    a = np.bincount(I.astype(np.uint8))
    interval = 255//(numofbins)
    hist = []
    for i in range(numofbins):
        hist.append(sum(a[i*interval:(i+1) * interval]))

    print(hist)
    return hist


def plot2(one, title1, two, title2, cmap='gray'):
    plt.subplot(1, 2, 1)
    plt.imshow(one, cmap=cmap)
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(two, cmap=cmap)
    plt.title(title2)

    plt.show()


if __name__ == "__main__":
    # secondA()
    diffBetweenMyHist()
