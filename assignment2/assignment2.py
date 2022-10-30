# %% [markdown]
# Firstly, you will implement the function myhist3 that computes a 3-D histogram
# from a three channel image. The images you will use are RGB, but the function
# should also work on other color spaces. The resulting histogram is stored in a 3-D
# matrix. The size of the resulting histogram is determined by the parameter n_bins.
# 1
# The bin range calculation is exactly the same as in the previous assignment, except
# now you will get one index for each image channel. Iterate through the image pixels
# and increment the appropriate histogram cells. You can create an empty 3-D numpy
# array with H = np.zeros((n_bins,n_bins,n_bins)). Take care that you normalize
# the resulting histogram.

# %%
from array import array
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from a2_utils import *


def myhist2(I, numofbins):
    I = I.reshape(-1)
    max_v = max(I)
    min_v = min(I)
    interval = (max_v - min_v) / numofbins
    hist = np.zeros(numofbins)
    for i in I:
        if i == max_v:
            hist[-1] += 1
        else:
            hist[int((i - min_v) // interval)] += 1

    return np.array(hist)/sum(hist)


def myhist3(image, numofbins):
    print("myhist3")
    hist = np.zeros(numofbins)
    if len(image.shape) == 3:
        H = np.zeros((numofbins, numofbins, numofbins)).astype(np.int64)
        max_v = np.max(image)
        min_v = np.min(image)
        image = image.reshape(-1, 3)
        interval = (max_v - min_v) / numofbins
        counter = 0
        for r, g, b in image:
            r_i = int((r - min_v) / interval)
            g_i = int((g - min_v) / interval)
            b_i = int((b - min_v) / interval)
            if r_i == numofbins:
                r_i -= 1
            if g_i == numofbins:
                g_i -= 1
            if b_i == numofbins:
                b_i -= 1
            H[r_i, g_i, b_i] += 1

        return H/np.sum(H)
    else:
        return myhist2(image, numofbins)


def firstA():
    myhist3(imread("images/lena.png"), 50)


I = imread("images/lena.png")
myhist3(I, 8)


# %% [markdown]
# In order to perform image comparison using histograms, we need to implement
# some distance measures. These are defined for two input histograms and return a
# single scalar value that represents the similarity (or distance) between the two histograms.
# Implement a function compare_histograms that accepts two histograms
# and a string that identifies the distance measure you wish to calculate. You can
# start with the L2 metric.
# The L2 metric (commonly known as Euclidean distance)
# Also implement the following measures that are more suitable for histogram comparison:
# * Chi-square distance
# * Intersection
# * Hellinger distance
# Try to avoid looping over histogram values and instead use vector operations on
# entire matrices at once.

# %%
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


one = myhist3(imread("images/lena.png"), 50)
two = myhist3(imread("images/lincoln.jpg"), 50)
print(euclidean_distance(one, two))


def chi_square(x, y):
    return np.sum((x - y)**2 / (x + y + 1e-6)) / 2


print(chi_square(one, two))


def intersection(x, y):
    return 1 - np.sum(np.minimum(x, y))


def helliger(x, y):
    return np.sum((np.sqrt(x) - np.sqrt(y))**2) / 2


print(intersection(one, two))
print(helliger(one, two))


# %% [markdown]
# Test your function with the following images:
#
# * dataset/object_01_1.png,
# * dataset/object_02_1.png,
# * dataset/object_03_1.png.
#
# Compute a 888-bin 3-D histogram for each image. Reshape each of them into a
# 1-D array. Using plt.subplot(), display all three images in the same window as well
# as their corresponding histograms. Compute the L2 distance between histograms of
# object 1 and 2 as well as L2 distance between histograms of objects 1 and 3.

# %%
def firstC():
    I1 = imread("./dataset/object_01_1.png")
    I2 = imread("./dataset/object_02_1.png")
    I3 = imread("./dataset/object_03_1.png")
    hist1 = myhist3(I1, 8).reshape(-1)
    hist2 = myhist3(I2, 8).reshape(-1)
    hist3 = myhist3(I3, 8).reshape(-1)

    a, two = plt.subplots(2, 3)
    a.suptitle("histograms")

    two[0, 0].imshow(I1, cmap="gray")
    two[0, 0].set(title="object_01_1.png")

    two[0, 1].imshow(I2, cmap="gray")
    two[0, 1].set(title="object_02_1.png")

    two[0, 2].imshow(I3, cmap="gray")
    two[0, 2].set(title="object_03_1.png")

    two[1, 0].bar(height=hist1, x=range(len(hist1)), width=10)
    two[1, 0].set(title="l2(h1,h1) = " + str(euclidean_distance(hist1, hist1)))

    two[1, 1].bar(height=hist2, x=range(len(hist2)), width=10)
    two[1, 1].set(title="l2(h1,h2) = " + str(euclidean_distance(hist1, hist2)))

    two[1, 2].bar(height=hist3, x=range(len(hist3)), width=10)
    two[1, 2].set(title="l2(h1,h3) = " + str(euclidean_distance(hist1, hist3)))

    plt.show()


firstC()
