# %% [markdown]
# # Assignment 5: Epipolar geometry and triangulation

# %%
#imports
import numpy as np
from matplotlib import pyplot as plt
from a5_utils import *
import cv2 as cv2
from UZ_utils import *

plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 150 

# %% [markdown]
# ## Exercise 1: Disparity
# 
# In this assignment we will focus on calculating disparity from a two-camera system. Our
# analysis will be based on a simplified stereo system, where two identical cameras are
# aligned with parallel optical axes and their image planes (CCD sensors) lie on the same
# plane (Image 1a).

# %% [markdown]
# #### A)
# the further the object is the lower the disparity

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### B)
# Write a script that computes the disparity for a range of values of pz. Plot the values
# to a figure and set the appropriate units to axes. Use the following parameters of
# the system: focal length is f = 2:5mm and stereo system baseline is T = 12cm.

# %%

def disparity(pz, FOCAL_LENGTH=0.25, BASELINE=12):
  return BASELINE * FOCAL_LENGTH / pz


pz_s = np.linspace(0.1, 10, 100)
d_s = disparity(pz_s)

plt.plot(pz_s, d_s)
plt.xlabel('Z (m)')
plt.ylabel('Disparity')
plt.title('Disparity vs. Z')
plt.show()


# %% [markdown]
# ### C)
# In order to get a better grasp on the idea of distance and disparity, you will calculate
# the numbers for a specific case. We will take the parameters from a specification of
# a commercial stereo camera Bumblebee2 manufactured by the company PointGray:
# f = 2.5mm, T = 12cm, whose image sensor has a resolution of 648x488 pixels that
# are square and the width of a pixel is 7.4um. We assume that there is no empty
# space between pixels and that both cameras are completely parallel and equal. Lets
# say that we use this system to observe a (point) object that is detected at pixel 550
# in x axis in the left camera and at the pixel 300 in the right camera. How far is the
# object (in meters) in this case? How far is the object if the object is detected at
# pixel 540 in the right camera? Solve this task analytically and bring your solution
# to the presentation of the exercise.

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### D)
# Write a script that calculates the disparity for an image pair. Use
# the images in the directory disparity. Since the images were pre-processed we can
# limit the search for the most similar pixel to the same row in the other image. Since
# just the image intensity carries too little information, we will instead compare small
# image patches. A simple way of finding a matching patch is to use normalized cross
# correlation.
# A patch
# from the second image is considered a match if it has the highest NCC value. The
# difference in x axis between the point from the first image and the matching point
# from the second image is the disparity estimate in terms of pixels1. The disparity
# search is performed in a single direction.

# %%
def normalized_cross_correlation(img1, img2):
  mean1 = np.mean(img1)
  mean2 = np.mean(img2)
  top = np.sum((img1 - mean1) * (img2 - mean2))
  bottom = np.sqrt(np.sum(np.square(img1 - mean1)) * np.sum(np.square(img2 - mean2)))
  return top / bottom


def disparity(image1, image2):
  image1 = cv2.resize(image1, None, fx=0.4, fy=0.4)
  image2 = cv2.resize(image2, None, fx=0.4, fy=0.4)
  window_size = 21
  patch_size = 5
  disparity_map = np.zeros_like(image1)

  # walk through each pixel in image1
  for y in range(patch_size, image1.shape[0] - patch_size):
    # get patches from image1 for the whole row
    patches1 = []
    patches2 = []
    for x in range(patch_size, image1.shape[1] - patch_size):
      patches1.append(image1[y - patch_size:y + patch_size + 1, x - patch_size:x + patch_size + 1])
      patches2.append(image2[y - patch_size:y + patch_size + 1, x - patch_size:x + patch_size + 1])

    pathces_len = len(patches1)

    for i in range(pathces_len):
      pixel = patches1[i] 
      # get the window around the pixel
      window = patches2[max(0, i - window_size) : min(image2.shape[1], i + window_size)]
      # calculate the normalized cross correlation for each window
      nccs = [normalized_cross_correlation(pixel, w) for w in window]
      # find the best match
      best_match = np.argmax(nccs)
      # calculate the disparity
      disparity_map[y, i + patch_size] = abs( window_size - best_match)

  return disparity_map

image1 = cv2.imread('./data/disparity/office_left.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./data/disparity/office_right.png', cv2.IMREAD_GRAYSCALE)

disparity_map = disparity(image1, image2)
plt.imshow(disparity_map, cmap='gray')
plt.show()



# %% [markdown]
# ## Exercise 2: Fundamental matrix, epipoles, epipolar lines

# %% [markdown]
# ### A)
# ![image.png](attachment:image.png)
# 
# 
# presecisce teh dveh premic je torej tocka `T(1,1)`

# %% [markdown]
# ### B)
# Implement a function fundamental_matrix that is given a set of (at least) eight
# pairs of points from two images and computes the fundamental matrix using the
# eight-point algorithm.
# As the eight-point algorithm can be numerically unstable, it is usually not executed
# directly on given pairs of points. Instead, the input is first normalized by centering
# them to their centroid and scaling their positions so that the average distance to the
# centroid sqrt(2). To achieve this, you can use the function normalize_points from
# the supplementary material.

# %%
def fundamental_matrix(p1, p2):
  # normalize points
  p1, T1 = normalize_points(p1)
  p2, T2 = normalize_points(p2)
  #generate A matrix
  A = np.zeros((p1.shape[0], 9))
  for i in range(p1.shape[0]):
    u1 = p1[i, 0]
    v1 = p1[i, 1]
    u2 = p2[i, 0]
    v2 = p2[i, 1]
    A[i] = np.array([u1 * u2, u2 * v1, u2, v2 * u1, v1 * v2, v2, u1, v1, 1])

  # decompose A into U, D, V
  U, D, V = np.linalg.svd(A)
  # F is the last column of V
  F = V[-1,:].reshape(3, 3)

  # decoimpose F into U, D, V
  U, D, V = np.linalg.svd(F)
  # set smallest singular value to 0
  D[-1] = 0
  # recompose F
  F = U @ np.diag(D) @ V
  # denormalize F
  F = T2.T @ F @ T1
  return F


house_1 = cv2.imread('./data/epipolar/house1.jpg', cv2.IMREAD_GRAYSCALE)
house_2 = cv2.imread('./data/epipolar/house2.jpg', cv2.IMREAD_GRAYSCALE)


points = np.loadtxt("./data/epipolar/house_points.txt")
points_1 = points[:, :2]
points_2 = points[:, 2:]

F = fundamental_matrix(points_1, points_2)
print(F)


#draw epipolar lines
def draw_epipolar_lines(img1, img2, points1, points2, F):
  # draw points
  plt.subplot(1, 2, 1)
  plt.imshow(img1, cmap='gray')
  plt.scatter(points1[:, 0], points1[:, 1], c='r')
  plt.subplot(1, 2, 2)
  plt.imshow(img2, cmap='gray')
  plt.scatter(points2[:, 0], points2[:, 1], c='r')
  # draw epipolar lines
  for i in range(points1.shape[0]):
    p1 = points1[i]
    p2 = points2[i]
    l = F @ np.array([p1[0], p1[1], 1])
    x = np.linspace(0, img2.shape[1], 100)
    y = -(l[0] * x + l[2]) / l[1]
    plt.subplot(1, 2, 2)
    plt.plot(x, y, c='r')

    l_ = F.T @ np.array([p2[0], p2[1], 1])
    x_ = np.linspace(0, img1.shape[1], 100)
    y_ = -(l_[0] * x_ + l_[2]) / l_[1]
    plt.subplot(1, 2, 1)
    plt.plot(x_, y_, c='r')

  plt.show()



draw_epipolar_lines(house_1, house_2, points_1, points_2, F)




# %% [markdown]
# ### C)
# We use the reprojection error as a quantitative measure of the quality of the estimated
# fundamental matrix.
# Write a function reprojection_error that calculates the reprojection error of a
# fundamental matrix F given two matching points. For each point, the function
# should calculate the corresponding epipolar line from the point’s match in the other
# image, then calculate the perpendicular distance between the point and the line
# using the equation:

# %%
def reprojection_error(F, point1, point2):
  # convert to homogeneous coordinates
  p1 = np.array([point1[0], point1[1], 1])
  p2 = np.array([point2[0], point2[1], 1])
  # compute epipolar line
  l = F @ p1
  # compute distance between point and line
  left_to_right = np.abs(l @ p2) / np.sqrt(l[0]**2 + l[1]**2)

  l_ = F.T @ p2
  right_to_left = np.abs(l_ @ p1) / np.sqrt(l_[0]**2 + l_[1]**2)
  # print(left_to_right+ right_to_left)
  return (left_to_right + right_to_left)/2

test_error = reprojection_error(F, [85,233], [67,219])
print(test_error)

sum_error = 0
for i in range(points_1.shape[0]):
  sum_error += reprojection_error(F, points_1[i], points_2[i])

print(sum_error / points_1.shape[0])



# %% [markdown]
# ### D)
# Perform fully automatic fundamental matrix estimation on a pair
# of images from the directory desk. Detect the correspondence points using your
# preferred method. As some of the matches might be incorrect, extend the RANSAC
# algorithm so that it will work for fundamental matrix estimation. You can measure
# the quality of a solution by using the point-to-line reprojection error. Display the
# correspondences to check whether all of them are correct. Calculate the fundamental
# matrix on the final set of inliers and show correspondences with epipolar lines.

# %%
desk1 = cv2.imread('./data/desk/DSC02638.JPG', cv2.IMREAD_GRAYSCALE)
desk2 = cv2.imread('./data/desk/DSC02639.JPG', cv2.IMREAD_GRAYSCALE)

# detect features
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(desk1, None)
kp2, des2 = orb.detectAndCompute(desk2, None)

# match features
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# filter matches
good = []
for m, n in matches:
  if m.distance < 0.75 * n.distance:
    good.append(m)
    
# draw matches
img3 = cv2.drawMatches(desk1, kp1, desk2, kp2, good, None, flags=2)
plt.imshow(img3)
plt.show()

# extract points
points1 = np.array([kp1[m.queryIdx].pt for m in good])
points2 = np.array([kp2[m.trainIdx].pt for m in good])

print(points1.shape)
print(points2.shape)


# RANSAC
def ransac(points1, points2, n, k, t, d):
  best_F = None
  best_inliers = None
  best_inliers_points1 = None
  best_inliers_points2 = None
  for i in range(n):
    # randomly select k points
    idx = np.random.choice(points1.shape[0], k, replace=False)
    # estimate fundamental matrix
    F = fundamental_matrix(points1[idx], points2[idx])
    # count inliers
    inliers = 0
    inliers_points1 = []
    inliers_points2 = []
    for j in range(points1.shape[0]):
      if reprojection_error(F, points1[j], points2[j]) < t:
        inliers += 1
        inliers_points1.append(points1[j])
        inliers_points2.append(points2[j])
    # update best model
    if inliers > d:
      print('Found better model with %d inliers' % inliers)
      best_F = F
      best_inliers = inliers
      best_inliers_points1 = inliers_points1
      best_inliers_points2 = inliers_points2
  # re-estimate fundamental matrix on all inliers
  best_F = fundamental_matrix(np.array(best_inliers_points1), np.array(best_inliers_points2))
  return best_F, best_inliers_points1, best_inliers_points2

# run RANSAC
F, inliers1, inliers2 = ransac(points1, points2, 1000, 5, 15, 40)

# draw epipolar lines 
draw_epipolar_lines(desk1, desk2, np.array(inliers1), np.array(inliers2), F)

# compute reprojection error
sum_error = 0
for i in range(len(inliers1)):
  sum_error += reprojection_error(F, inliers1[i], inliers2[i])
  
print(sum_error / len(inliers1))

# %% [markdown]
# ## Exercise 3: Triangulation

# %% [markdown]
# ### A)
# Implement the function triangulate that accepts a set of correspondence points
# and a pair of calibration matrices as an input and returns the triangulated 3D
# points. Test the triangulation on the ten points from the file house_points.txt.
# Visualize the result using plt.plot or plt.scatter. Also plot the index of the
# point in 3D space (use plt.text) so the results will be easier to interpret. Plot the
# points interactively, so that you can rotate the visualization.
# Note: The coordinate system used for plotting in 3D space is usually not the same
# as the camera coordinate system. In order to make the results easier to interpret, the
# ordering of the axes can be modified by using a transformation matrix.

# %%
## vecotr product in matrix form

def vec_prod(a):
  return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def triangulate(points_1, points_2, camera1, camera2):  
  # homogunes points
  points_1 = np.hstack((points_1, np.ones((points_1.shape[0], 1))))
  points_2 = np.hstack((points_2, np.ones((points_2.shape[0], 1))))

  # triangulate
  P = []
  for one, two in zip(points_1, points_2):
    A1 = vec_prod(one) @ camera1
    A2 = vec_prod(two) @ camera2
    A = np.vstack((A1[:2, :], A2[:2, :]))
    U, S, Vt = np.linalg.svd(A)
    # Last column of V
    X = Vt.T[:, -1]
    # Normalize
    X = X / X[-1]
    # Append to list
    P.append(X[:3])
  
  return np.array(P)

#callibration matrix
camera1 = np.loadtxt("./data/epipolar/house1_camera.txt")
camera2 = np.loadtxt("./data/epipolar/house2_camera.txt")

# house points
points = np.loadtxt("./data/epipolar/house_points.txt")
points_1 = points[:, :2]
points_2 = points[:, 2:]


#plot

T = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])

# triangulate points
X = triangulate(points_1, points_2, camera1, camera2)
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = (T @ X.T).T
# Plot points into 3d space
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
for i in range(X.shape[0]):
  ax.text(X[i, 0], X[i, 1], X[i, 2], str(i))

# plot pictures
# # plot points on pictures with labels
fig, ax = plt.subplots(1, 2)
ax[0].imshow(plt.imread("./data/epipolar/house1.jpg"))
ax[0].scatter(points_1[:, 0], points_1[:, 1], c='r', s=1)
ax[1].imshow(plt.imread("./data/epipolar/house2.jpg"))
ax[1].scatter(points_2[:, 0], points_2[:, 1], c='r', s=1)

for i in range(points_1.shape[0]):
  ax[0].text(points_1[i, 0], points_1[i, 1], str(i))
  ax[1].text(points_2[i, 0], points_2[i, 1], str(i))

plt.show()