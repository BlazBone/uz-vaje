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
    print(I.shape)
    print(I.dtype)
    imshow(I)
    return I


def firstB(I):
    """
    Convert the loaded image to grayscale.
    """
    red = I[:, :, 0]
    green = I[:, :, 1]
    blue = I[:, :, 2]

    together = (red + green + blue) / 3
    imshow(together)
    return together


def firstC(I):

    # Cut and display a specific part of the loaded image.
    cutout = I[130:260, 240:450, 1]
    imshow(cutout)
    plt.imshow(cutout, cmap='gray')
    plt.show()
    # cutout = I[130:260, 240:450, 0]
    # imshow(cutout)
    # cutout = I[130:260, 240:450, 2]
    # imshow(cutout)
    return cutout


def firstD(I):
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


def firstE(I):
    # already grayscale and floating point
    ICpy = (I.copy() * 63).astype(np.uint8)

    plt.subplot(1, 2, 1)
    plt.imshow(ICpy, vmax=255, cmap='gray')
    plt.title('uint8')

    plt.subplot(1, 2, 2)
    plt.imshow(I, cmap='gray')
    plt.title('float64')
    plt.show()


def first():
    I = firstA()
    I_gray = firstB(I)
    cutout = firstC(I)
    firstD(I)
    firstE(I_gray)


def secondA():
    I = imread_gray("./images/bird.jpg")
    mask = I.copy()
    threshold = 0.29  # otsu method returned this value
    mask[I < threshold] = 0
    mask[I > threshold] = 1
    plot2(I, 'original', mask, 'mask')
    return mask


def secondB():
    I = imread_gray("./images/bird.jpg")
    # bar za histograme
    i = myhist1(I*255, 255)
    plt.bar(height=i, x=range(255))
    plt.title("myhist1, image bird.jpg, bins=255")
    plt.show()

    i = myhist1(I*255, 10)

    plt.bar(height=i, x=range(10))
    plt.title("myhist1, image bird.jpg, bins=10")

    plt.show()
    i = myhist2(I, 25)
    plt.bar(height=i, x=range(25))
    plt.title("myhist2, image bird.jpg, bins=25")
    plt.show()

    i = myhist2(I, 50)
    plt.bar(height=i, x=range(50))
    plt.title("myhist2, image bird.jpg, bins=50")
    plt.show()


def secondC():
    # mej neko sliko ki ima recimo vrednosti samo 150 in en hist bo meu nekje na sredi en pa bo lepo raztegnu
    I4 = imread_gray("./images/umbrellas.jpg")

    images = [I4]

    for image in images:
        rand = 50
        h1 = myhist1(image * 150, rand)
        h2 = myhist2(image, rand)

        plt.bar(x=range(50), height=h1, label='myhist1')
        plt.title("myhist1 samo do 150")
        plt.show()

        plt.bar(x=range(50), height=h2, label='myhist2')
        plt.title("myhist2 razsiri cez celo")

        plt.show()


def secondD():
    I1 = imread_gray("./images/lightest.jpg")
    I2 = imread_gray("./images/light.jpg")
    I3 = imread_gray("./images/dark.jpg")

    h1 = myhist2(I1, 150)
    h11 = myhist2(I1, 30)

    h2 = myhist2(I2, 150)
    h21 = myhist2(I2, 30)

    h3 = myhist2(I3, 150)
    h31 = myhist2(I3, 30)

    a, axis = plt.subplots(3, 3)
    axis[0, 0].imshow(I1, cmap='gray')
    axis[0, 0].set_title("lightest")
    axis[0, 1].bar(x=range(150), height=h1)
    axis[0, 2].bar(x=range(30), height=h11)

    axis[1, 0].imshow(I2, cmap='gray')
    axis[1, 0].set_title("light")
    axis[1, 1].bar(x=range(150), height=h2)
    axis[1, 2].bar(x=range(30), height=h21)

    axis[2, 0].imshow(I3, cmap='gray')
    axis[2, 0].set_title("dark")
    axis[2, 1].bar(x=range(150), height=h3)
    axis[2, 2].bar(x=range(30), height=h31)

    plt.show()


def secondE():
    I1 = imread_gray("./images/eagle.jpg")
    I2 = imread_gray("./images/bird.jpg")
    I3 = imread_gray("./images/coins.jpg")
    I4 = imread_gray("./images/umbrellas.jpg")

    t1 = myOtsu(I1)
    t2 = myOtsu(I2)
    t3 = myOtsu(I3)
    t4 = myOtsu(I4)

    a, two = plt.subplots(2, 2)
    a.suptitle("comparison of myOtsu")

    two[0, 1].imshow(I2, cmap="gray")
    two[0, 1].set(title="bird image, threshold: " + str(t2))

    two[0, 0].imshow(I3, cmap="gray")
    two[0, 0].set(title="coins image, threshold: " + str(t3))

    two[1, 0].imshow(I4, cmap="gray")
    two[1, 0].set(title="umbrellas image, threshold: " + str(t4))

    two[1, 1].imshow(I1, cmap="gray")
    two[1, 1].set(title="eagle image, threshold: " + str(t1))

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
    return np.array(hist)/sum(hist)


def myhist1(I, numofbins):
    I = I.reshape(-1)
    a = np.bincount(I.astype(np.uint8))
    interval = 255//(numofbins)
    hist = []
    for i in range(numofbins):
        hist.append(sum(a[i*interval:(i+1) * interval]))
    return np.array(hist)/sum(hist)


def plot2(one, title1, two, title2, cmap='gray'):
    plt.subplot(1, 2, 1)
    plt.imshow(one, cmap=cmap)
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(two, cmap=cmap)
    plt.title(title2)

    plt.show()


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


def second():
    secondA()
    secondB()
    secondC()
    secondD()
    secondE()


def thirdA(n=5):

    for n in [1, 2, 3, 4]:
        I = imread("./images/mask.png")
        SE = np.ones((n, n), np.uint8)
        I_eroded = cv2.erode(I, SE, iterations=1)
        I_dilate = cv2.dilate(I, SE, iterations=1)
        I_closing = cv2.erode(I_dilate, SE, iterations=1)
        I_opening = cv2.dilate(I_eroded, SE, iterations=1)

        plt.subplot(1, 5, 1)
        plt.imshow(I)
        plt.title('original')

        plt.subplot(1, 5, 2)
        plt.imshow(I_eroded)
        plt.title('eroded ')

        plt.subplot(1, 5, 3)
        plt.imshow(I_dilate)
        plt.title('dilated')

        plt.subplot(1, 5, 4)
        plt.imshow(I_closing)
        plt.title('closing')

        plt.subplot(1, 5, 5)
        plt.imshow(I_opening)
        plt.title('opening')

        plt.show()


def thirdB():
    mask = secondA()
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    I = cv2.dilate(mask, se, iterations=5)
    I = cv2.erode(I, se, iterations=4)
    imshow(I)
    return mask


def imask(image, mask):
    mask = np.expand_dims(mask, axis=2)
    cut_image = image * mask
    imshow(cut_image)
    return cut_image


def thirdD():
    I = imread_gray("./images/eagle.jpg")
    imshow(I)
    mask = I.copy()

    threshold = myOtsu(I)

    mask[I < threshold] = 0
    mask[I > threshold] = 1
    # fix mask
    imshow(mask)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, se, iterations=4)
    mask = cv2.dilate(mask, se, iterations=3)

    plot2(imread("./images/eagle.jpg"), 'original', mask, 'mask')

    imask(imread("./images/eagle.jpg"), mask)
    # invert the image
    I = 1 - I

    mask[I < threshold] = 0
    mask[I > threshold] = 1
    imshow(mask)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=5)

    plot2(I, 'original', mask, 'mask')
    imask(imread("./images/eagle.jpg"), mask)

    return mask


def thirdE():
    I = imread_gray("./images/coins.jpg")
    I = 1 - I
    threshold = 0.1
    mask = I.copy()
    mask[I < threshold] = 0
    mask[I > threshold] = 1
    plot2(imread("./images/coins.jpg"), 'original', mask, 'mask')
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    I_eroded = cv2.erode(mask, SE, iterations=1)
    I_dilate = cv2.dilate(I_eroded, SE, iterations=1)
    imshow(I_dilate)
    connectivity = 8
    a = cv2.connectedComponentsWithStats(
        np.array(mask*255).astype(np.uint8), connectivity, cv2.CV_32S)
    imshow(a[1])

    b = np.array(a[1]).reshape(-1)

    change = set()
    for i, x in enumerate(np.bincount(b)):
        if x > 700:
            change.add(i)

    for i, x in enumerate(a[1].reshape(-1)):
        if x in change:
            a[1].reshape(-1)[i] = 0
        else:
            a[1].reshape(-1)[i] = 1

    mask_end = np.expand_dims(np.array(a[1]), axis=2)
    imshow(mask_end)
    coins_color = imread("./images/coins.jpg")
    coins_color = coins_color * mask_end
    coins_color[coins_color == 0] = 255
    imshow(coins_color)


def third():
    thirdA()
    mask = thirdB()
    cut_image = imask(imread("./images/bird.jpg"), mask)  # third C
    thirdD()
    thirdE()


if __name__ == "__main__":
    print('hej')
    first()
    second()
    third()
