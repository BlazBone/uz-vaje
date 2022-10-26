from array import array
from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from a2_utils import *


def first_A(s):
    print("hej")


def myhist3(image: npt.NDArray[np.float64], numofbins):
    print("myhist3")
    print(image.shape)
    image.

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


print("hej")
if __name__ == "__main__":
    myhist3(imread("images/lena.png"), 50)
    myhist3(imread_gray("images/lena.png"), 50)
