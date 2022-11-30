# %% [markdown]
# # Assignment 4: Feature points, matching, homography
# 

# %% [markdown]
# ## Exercise 1: Feature points detectors
# In this exercise you will implement two frequently used feature point detectors: the Hessian
# algorithm and the Harris algorithm.

# %%
#imports
import numpy as np
from matplotlib import pyplot as plt
from a4_utils import *
import cv2 as cv2
from UZ_utils import *
import math
import os

# %% [markdown]
# ### A)
# Implement a function hessian_points, that computes a Hessian determinant using
# the equation for each pixel of the input image. As this computation can be
# very slow if done pixel by pixel, you have to implement it using vector operations
# (without explicit for loops). Test the function using image from test_points.jpg
# as your input (do not forget to convert it to grayscale) and visualize the result.
# 

# %%
def get_image_derivetive(image, sigma):
    gaus = np.array(gauss(sigma))

    gaus_dx = np.array(gaussdx(sigma))
    gaus_dx = np.flip(gaus_dx)

    I_x = cv2.filter2D(image, -1, gaus.T)
    I_x = cv2.filter2D(I_x, -1, gaus_dx)

    I_y = cv2.filter2D(image, -1, gaus)
    I_y = cv2.filter2D(I_y, -1, gaus_dx.T)

    return I_x, I_y


def get_second_image_derivative(image, sigma):
    gaus = np.array(gauss(sigma))
    gaus_dx = np.array(gaussdx(sigma))
    gaus_dx = np.flip(gaus_dx)

    I_x, I_y = get_image_derivetive(image, sigma)

    I_xx = cv2.filter2D(I_x, -1, gaus.T)
    I_xx = cv2.filter2D(I_xx, -1, gaus_dx)

    I_yy = cv2.filter2D(I_y, -1, gaus)
    I_yy = cv2.filter2D(I_yy, -1, gaus_dx.T)

    I_xy = cv2.filter2D(I_x, -1, gaus)
    I_xy = cv2.filter2D(I_xy, -1, gaus_dx.T)

    return I_xx, I_yy, I_xy

# %%
def non_maxima_box(image):
  for i in range(1, len(image)-1):
      for j in range(1, len(image[0])-1):
          neigbours = image[-1+i:i+2, j-1:j+2]
          if image[i, j] < np.max(neigbours):
              image[i, j] = 0
  return image

def non_maxima_box_n(image,n):
  for i in range(n, len(image)-n):
      for j in range(n, len(image[0])-n):
          neigbours = image[-n+i:i+n+1, j-n:j+n+1]
          if image[i, j] < np.max(neigbours):
              image[i, j] = 0
  return image

def hessian_points(image, sigma, treshold):
  I_xx, I_yy, I_xy = get_second_image_derivative(image, sigma)
  det = np.multiply(I_xx, I_yy) - np.square(I_xy)
  det[det<treshold] = 0
  return det

  
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

fig, ax = plt.subplots(2, 3)

for i in range(1,4):
  print(i)
  image = imread_gray("./data/graf/graf_a.jpg")
  test = hessian_points(image, i*3, 0.004)
  ax[0,i-1].imshow(test)
  testa = test.copy()
  det = non_maxima_box(test)
  det[det>0] = 1
  ax[1, i-1].imshow(image, cmap="gray")
  ax[1, i-1].scatter(np.where(det != 0)[1], np.where(det != 0)[0],marker = "o", c="red", s=2)
  ax[1, i-1].set_title("sigma = " + str(i  * 3))



# %% [markdown]
# #### Question: What kind of structures in the image are detected by the algorithm? How does the parameter sigma affect the result?
# #### Answer: It detects blobs
# 

# %% [markdown]
# ### B)
# Implement the Harris feature point detector. This detector is based on the autocorrelation
# matrix C that measures the level of self-similarity for a pixel neighborhood
# for small deformations. At the lectures, you have been told that the Harris
# detector chooses a point (x; y) for a feature point if both eigenvalues of the
# auto-correlation matrix for that point are large. This means that the neighborhood
# of (x; y) contains two well-defined rectangular structures – i.e. a corner. Autocorrelation
# matrix can be computed using the first partial derivatives at (x; y) that
# 2
# are subsequently smoothed using a Gaussian filter

# %%
def auto_corelation(image,sigma_picture,sigma_derivetive):
  image = image.copy()
  I_x, I_y = get_image_derivetive(image, sigma_derivetive)
  gaus_kernel = np.array(gauss(sigma_picture))
  i_xx = I_x**2
  i_yy = I_y**2
  i_xy = I_x*I_y
  # det of C
  one = cv2.filter2D(i_xx, -1, gaus_kernel)
  one = cv2.filter2D(one, -1, gaus_kernel.T)
  two = cv2.filter2D(i_xy, -1, gaus_kernel)
  two = cv2.filter2D(two, -1, gaus_kernel.T)
  three = cv2.filter2D(i_xy, -1, gaus_kernel)
  three = cv2.filter2D(three, -1, gaus_kernel.T)
  four = cv2.filter2D(i_yy, -1, gaus_kernel)
  four = cv2.filter2D(four, -1, gaus_kernel.T)
  det = np.multiply(one, four) - np.multiply(two, three)
  trace = one + four
  return det - 0.04 * np.square(trace)
  

fig, ax = plt.subplots(2, 3)
for i in range(1,4):
  print(i)
  image = imread_gray("./data/graf/graf_a.jpg")
  test = auto_corelation(image, sigma_picture=3*i, sigma_derivetive=(3*i)/1.6)
  test_cpy = test.copy()
  test_cpy[test_cpy<0.00004] = 0
  test_cpy = non_maxima_box_n(test_cpy, 3)
  ax[0,i-1].imshow(test)
  ax[1, i-1].imshow(image, cmap="gray")
  ax[1, i-1].scatter(np.where(test_cpy != 0)[1], np.where(test_cpy != 0)[0], c="red", s=2, marker="x")
  ax[1, i-1].set_title("sigma = " + str(i  * 3))


# %% [markdown]
# #### Question : Do the feature points of both detectors appear on the same structures in the image?
# 
# #### Answer: yes in most cases they do. We are looking for corners blobs,....

# %% [markdown]
# ## Exercise 2: Matching local regions
# One of the uses of feature points is searching for similar structures in different images.
# To do this, we will need descriptors of the regions around these points. In this
# assignment you will implement some simple descriptors as well as their matching.

# %% [markdown]
# ### A)
# Use the function simple_descriptors from a4_utils.py to calculate descriptors
# for a list of feature points. Then, write a function find_correspondences which
# calculates similarities between all descriptors in two given lists. Use Hellinger
# distance (see Assignment 2). Finally, for each descriptor from the first list, find
# the most similar descriptor from the second list. Return a list of [a; b] pairs, where
# a is the index from the first list, and b is the index from the second list.
# Write a script that loads images graf/graf_a_small.jpg and graf/graf_b_small.jpg,
# runs the function find_correspondences and visualizes the result. Use the function
# display_matches from the supplementary material for visualization. Experiment
# with different parameters for descriptor calculation and report on the
# changes that occur.

# %%
def helliger(x, y):
    return np.sqrt(0.5 * np.sum(np.square(np.sqrt(x) - np.sqrt(y))))

# %%
def find_correspondences(desc1, desc2):
  print("find correspondences")
  # calculates similarities between all descriptors in two given lists. Use Hellinger
# distance (see Assignment 2). Finally, for each descriptor from the first list, find
# the most similar descriptor from the second list. Return a list of [a; b] pairs, where
# a is the index from the first list, and b is the index from the second list.
  correspondences = []
  for i in range(len(desc1)):
    distances = []
    for j in range(len(desc2)):
      distances.append(helliger(desc1[i], desc2[j]))
    correspondences.append([i, np.argmin(distances)])
  return correspondences


image1 = imread_gray("./data/graf/graf_a_small.jpg")
image2 = imread_gray("./data/graf/graf_b_small.jpg")
print("here")
sigma = 3
feature1 = hessian_points(image1, sigma, 0.004)
feature1_tmp  = feature1.copy()
# feature1_tmp[feature1_tmp<0.00004] = 0
feature1_tmp = non_maxima_box_n(feature1_tmp, 3)

feature2 = hessian_points(image2, sigma, 0.004)
feature2_tmp = feature2.copy()
# feature2_tmp[feature2_tmp<0.00004] = 0
feature2_tmp = non_maxima_box_n(feature2_tmp, 3)

# plot images and show the correspondences
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image1, cmap="gray")
ax[0].scatter(np.where(feature1_tmp != 0)[1], np.where(feature1_tmp != 0)[0], c="red", s=2, marker="x")
ax[1].imshow(image2, cmap="gray")
ax[1].scatter(np.where(feature2_tmp != 0)[1], np.where(feature2_tmp != 0)[0], c="red", s=2, marker="x")

print("dva")
f1_x, f1_y = np.where(feature1_tmp > 0.00004)
f2_x, f2_y = np.where(feature2_tmp > 0.00004)
print("tri")
descriptors_1 = simple_descriptors(image1, f1_x, f1_y)
descriptors_2 = simple_descriptors(image2, f2_x, f2_y)
print("stiri")


# %%
corespondences_indexes = find_correspondences(descriptors_1, descriptors_2)
print(corespondences_indexes)
print(find_correspondences(descriptors_2, descriptors_1))
points_1 = []
points_2 = []
for i in range(len(corespondences_indexes)):
  points_1.append([f1_y[corespondences_indexes[i][0]], f1_x[corespondences_indexes[i][0]]])
  points_2.append([f2_y[corespondences_indexes[i][1]], f2_x[corespondences_indexes[i][1]]])

display_matches(image1,points_1, image2, points_2)

# %% [markdown]
# When we set sigma and non_maxima_box rly high we only get a few points but from my experimenting they are connected correctly always. If we put sigma/threshold to lower scale we get much more points but some of them will be connected wrong. So when we would need to merge the image we would check what the translation is for the majority.

# %% [markdown]
# ### B)
# Implement a simple feature point matching algorithm. Write a function find_matches
# that is given two images as an input and returns a list of matched feature points
# from image 1 to image 2. The function should return a list of index pairs, where
# the first element is the index for feature points from the first image and the second
# element is the index for feature points from the second image.
# 
# - Execute a feature point detector to get stable points for both images (you
# can experiment with both presented detectors),
# - Compute simple descriptors for all detected feature points
# - Find best matches between descriptors in left and right images using the
# Hellinger distance, i.e. compute the best matches from the left to right image
# and then the other way around. In a post-processing step only select symmetric
# matches. This way we get a set of point pairs where each point from the left image
# is matched to exactly one point in the right image as well as the other way
# around.
# 
# Use the function display_matches from the supplementary material to display all
# the symmetric matches. Write a script that loads images graf/graf_a_small.jpg
# and graf/graf_b_small.jpg, runs the function find_matches and visualizes the
# result.
# 

# %%
# its the same we just inverse the order of the descriptors
# and then match the ones that are (x,y) in the first image and (y,x) in the second image
# simple
def find_matches(image1, image2):
  print("find matches")

  sigma = 1
  feature1 = hessian_points(image1, sigma, 0.004)
  feature1_tmp  = feature1.copy()
  feature1_tmp = non_maxima_box_n(feature1_tmp, 3)

  feature2 = hessian_points(image2, sigma, 0.004)
  feature2_tmp = feature2.copy()
  feature2_tmp = non_maxima_box_n(feature2_tmp, 3)

  f1_x, f1_y = np.where(feature1_tmp > 0.00004)
  f2_x, f2_y = np.where(feature2_tmp > 0.00004)

  descriptors_1 = simple_descriptors(image1, f1_x, f1_y)
  descriptors_2 = simple_descriptors(image2, f2_x, f2_y)

  corespondences_indexes = find_correspondences(descriptors_1, descriptors_2)
  corr_indexes_2 = find_correspondences(descriptors_2, descriptors_1)

  points_1 = []
  points_2 = []

  for x,y in corespondences_indexes:
    for x1,y1 in corr_indexes_2:
      if x == y1 and y == x1:
        points_1.append([f1_y[x], f1_x[x]])
        points_2.append([f2_y[y], f2_x[y]])
  return points_1, points_2

i1 = imread_gray("./data/graf/graf_a_small.jpg")
i2 = imread_gray("./data/graf/graf_b_small.jpg")

p1, p2 = find_matches(i1, i2)
display_matches(i1, p1, i2, p2)
print(p1)
print(p2)







# %% [markdown]
# #### Question: What do you notice when visualizing the correspondences? How accurate are the matches?
# 
# #### Answer: very accurate here we can have more points and we dont need to lower sigma and non maxima box like before to get good results.

# %% [markdown]
# ### C)
# (5 points) Incorrect matches can occur when matching descriptors. Suggest
# and implement a simple method for eliminating at least some of these incorrect
# matches. You can use the ideas that were presented at the lectures or test your
# own ideas. Either way, you need to explain the idea behind the method as well
# as demonstrate that the number of incorrect matches is lowered when using the
# proposed method.

# %% [markdown]
# I have an idea. Just calculate the average traslation of the points. Remove any outliers. This could be done with normal distribution and revmocig 2sigma or mybe just 1sigma. Or we could just hardcode.

# %%
avg_distance_x = 0
avg_distance_y = 0
tmp_x= 0
tmp_y = 0
for (x1,y1),(x2,y2) in zip(p1, p2):
  tmp_x = x1 - x2
  tmp_y = y1 - y2
  avg_distance_x += tmp_x
  avg_distance_y += tmp_y

avg_distance_x = avg_distance_x / len(p1)
avg_distance_y = avg_distance_y / len(p1)

print(avg_distance_x)
print(avg_distance_y)

new_p1 = []
new_p2 = []
for (x1,y1),(x2,y2) in zip(p1, p2):
  ## remove the outliers
  if abs(x1 - x2) < 10 and abs(y1 - y2) < 10:
    new_p1.append([x1,y1])
    new_p2.append([x2,y2])

display_matches(i1, new_p1, i2, new_p2)



# %% [markdown]
# we can clearly see the differences even though this is very harsh way of doing it. Example would be armpits, mouth....

# %% [markdown]
# ### E)
# Record a video with your phone or webcam. Use OpenCV to detect
# keypoints of your choice, display them using cv2.drawKeypoints and save the
# video with the displayed keypoints. The video must demonstrate the keypoint
# detector’s robustness to rotation and scale change. Make the video at least 15s
# long.

# %%
# WE DONT WANT  TO RUN THIS EVERY TIME

# cap = cv2.VideoCapture(0)

# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# # out = cv2.VideoWriter('output.mp4', fourcc, 20.0,
#                       # (int(cap.get(3)), int(cap.get(4))))

# print(cap)
# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("cant read frame")
#         break

#     sift = cv2.SIFT_create(200)
#     keypoint, descriptor = sift.detectAndCompute(frame, None)
#     frame_with_keypoints = cv2.drawKeypoints(
#         frame, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     # print(frame_with_keypoints.shape)
#     cv2.imshow("frame", frame_with_keypoints)
#     # out.write(frame_with_keypoints)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# # out.release()
# cv2.destroyAllWindows()


# %% [markdown]
# ## Exercise 3: Homography estimation
# In this assignment we are dealing with planar images, therefore we can try and estimate
# a homography matrix H that maps one image to another using planar correspondences.
# You will implement an algorithm that computes such a transformation using the minimization
# of the mean square error. For additional information about the method, consult
# the lecture notes as well as the course literature (in the literature, the term direct
# linear transform (DLT) is frequently used to describe the idea).

# %% [markdown]
# #### Question: Looking at the equation above, which parameters account for translation and which for rotation and scale?
# #### Answer: p1,p2 rotation and scale, p3,p4 translation

# %% [markdown]
# #### Question: Write down a sketch of an algorithm to determine similarity transform from a set of point correspondences P = [(xr1; xt1); (xr2; xt2); : : : (xrn; xtn)]. 
# 
# #### Answer: 

# %% [markdown]
# ### A)
# Write function estimate_homography, that approximates a homography between
# two images using a given set of matched feature points following the algorithm
# below.
# - Construct a matrix **A**
# - Perform a matrix decomposition using the SVD algorithm:
# U, S, VT = np.linalg.svd(A).
# - Compute vector **h**
# - reorder the elements of **h** to a 3x3 matrix **H**

# %%
def estimate_homography(points):
  # print("estimate homography")
  A = np.zeros((2*len(points), 9))

  for i, line in enumerate(points):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    A[2*i] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
    A[2*i+1] = [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2]
  
  U, S, VT = np.linalg.svd(A)
  V = VT.T
  H = V[:,8]
  H = H / H[8]
  return H.reshape(3,3)


im1 = imread_gray("./data/newyork/newyork_a.jpg")
im2 = imread_gray("./data/newyork/newyork_b.jpg")


points_new_york = np.loadtxt("./data/newyork/newyork.txt")
H_new_york = estimate_homography(points_new_york)

display_matches(im1, points_new_york[:,0:2], im2, points_new_york[:,2:4])

im = cv2.warpPerspective(im1, H_new_york, (im1.shape[1], im1.shape[0]))
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Default")
ax[0].imshow(im, cmap='gray')
ax[1].imshow(im2, cmap='gray')
plt.show()

im1 = imread_gray("./data/graf/graf_a.jpg")
im2 = imread_gray("./data/graf/graf_b.jpg")
points_graph = np.loadtxt("./data/graf/graf.txt")
H_graph = estimate_homography(points_graph)


print(H_graph)

display_matches(im1, points_graph[:,0:2], im2, points_graph[:,2:4])

im = cv2.warpPerspective(im1, H_graph, (im1.shape[1], im1.shape[0]))
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Default")
ax[0].imshow(im, cmap='gray')
ax[1].imshow(im2, cmap='gray')
plt.show()

# %% [markdown]
# ### B)
# Using the find_matches function that you have implemented in the previous
# exercise, find a set of matched points from either the graf or newyork image pairs.
# Then, implement the RANSAC algorithm for robust estimation of the homography matrix. For that you will need the reprojection error for each of the proposed
# solutions. The reprojection error for each point can be calculated by multiplying
# the point’s coordinates with the homography matrix and comparing the result
# to the reference point from the other image. You can use Euclidean distance for
# that. The final reprojection error for a solution should then be the average of
# reprojection errors for all included points.
# Find a subset of points that produce a high quality homography estimation and
# use that matrix H to transform one image to the other (you can again use
# cv2.warpPerspective).

# %%
import sys
def ransac(points, max_iterations, threshold):
  best_H = None
  # max value for int
  min_error = sys.maxsize  

  for i in range(max_iterations):
    random_points = np.random.choice(points.shape[0], 4, replace=False)
    random_points = points[random_points]
    H = estimate_homography(random_points)
    inliers = []
    h_errors = []
    for point in points:
      x1 = point[0]
      y1 = point[1]
      x2 = point[2]
      y2 = point[3]
      translated = np.dot(H, np.array([x1, y1, 1]))
      translated = translated / translated[2]
      distance = np.sqrt((translated[0] - x2)**2 + (translated[1] - y2)**2)
      h_errors.append(distance)
    
  # sort errors
    h_errors.sort()
    five_worst = len(h_errors) * 0.04

    err = sum(h_errors[:-int(five_worst)])
    if err < min_error:
      print("new best",i)
      min_error = err
      best_H = H

  return best_H

image1 = imread_gray("./data/newyork/newyork_a.jpg")
image2 = imread_gray("./data/newyork/newyork_b.jpg")
points1, points2 = find_matches(image1, image2)
display_matches(image1, points1, image2, points2)

points1 = np.array(points1)
points2 = np.array(points2)
# join points one and 2 to one array
points = np.hstack((points1, points2))

H = ransac(points, 100, 10)
print(H)

im1 = cv2.warpPerspective(image1, H, (image1.shape[1], image1.shape[0]))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Default")
ax[0].imshow(im1, cmap='gray')
ax[1].imshow(image2, cmap='gray')
plt.show()

# turn to array
# 0.76807 -0.63756 108.59988
# 0.64246 0.74179 -33.09045
# 0.00003 -0.00008 1.00000

H_perfect = np.array([[0.76807, -0.63756, 108.59988], [0.64246, 0.74179, -33.09045], [0.00003, -0.00008, 1.00000]])

print("difference of matrixes")
diff = H_perfect - H

print(diff/H)



# %% [markdown]
# #### Question: How many iterations on average did you need to find a good solution? How does the parameter choice for both the keypoint detector and RANSAC itself influence the performance (both quality and speed)?
# 
# #### Answer: What do we classify as good? i would say about 100. if we have good keypoint detector that detects only a few points that are symetrically linked we will need very few iterations. to get great results.

# %% [markdown]
# ### C)
# (5 points) Calculate the number of expected iterations for RANSAC using
# the formula mentioned in the instructions. Estimate the missing values (i.e. the
# inlier probability) using the example images used in this assignment. Propose and
# implement a method that will try to stop the algorithm as soon as a good enough
# solution is found.

# %% [markdown]
# Clean set of inliners. About 70%. 
# - k=10 p_fail = 0.06%
# - k=20 p_fail = 0.004%
# - k=100 p_fail = 0.00000000012%

# %% [markdown]
# Simple solution would be to stop after we improved our inital solution n times. We Could also say when none of the pixels is more than n pixels away from its intendet postion. Or just set a global treshold on error

# %%
def ransac2(points, max_iterations, num_improvements):
  best_H = None
  # max value for int
  min_error = sys.maxsize  
  count = 0
  for i in range(max_iterations):
    random_points = np.random.choice(points.shape[0], 4, replace=False)
    random_points = points[random_points]
    H = estimate_homography(random_points)
    inliers = []
    h_errors = []
    for point in points:
      x1 = point[0]
      y1 = point[1]
      x2 = point[2]
      y2 = point[3]
      translated = np.dot(H, np.array([x1, y1, 1]))
      translated = translated / translated[2]
      distance = np.sqrt((translated[0] - x2)**2 + (translated[1] - y2)**2)
      h_errors.append(distance)
    
  # sort errors
    h_errors.sort()
    five_worst = len(h_errors) * 0.04

    err = sum(h_errors[:-int(five_worst)])
    if err < min_error:
      print("new best",i)
      count+=1
      min_error = err
      best_H = H
      if count == num_improvements:
        break

  return best_H

H = ransac2(points, 100, 10)
print(H)

im1 = cv2.warpPerspective(image1, H, (image1.shape[1], image1.shape[0]))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Default")
ax[0].imshow(im1, cmap='gray')
ax[1].imshow(image2, cmap='gray')
plt.show()


# %% [markdown]
# ### E)
# (10 points) Write your own function for mapping points using the homography
# matrix. It should work like OpenCV’s warpPerspective(). It should accept an
# image and a homography matrix and return the input image as remapped by the
# homography matrix. The size of the output should match the size of the input
# image.
# Hint: You will need to use homogeneous coordinates.
# Note: The mapping should be without holes or artifacts. Additional post-processing
# (or interpolation) is not allowed. The solution is possible using only homography
# mapping.

# %%
def myWarpPerspective(image, H):
 
  height = image.shape[0]
  width = image.shape[1]

  new_image = np.zeros((height, width, 3))
  
  H_inv = np.linalg.inv(H)
  
  for x in range(width):
    for y in range(height):
      translated = np.dot(H_inv, np.array([x, y, 1]))
      translated = translated / translated[2]

      x1 = int(translated[0])
      y1 = int(translated[1])
      if x1 >= 0 and x1 < width and y1 >= 0 and y1 < height:
        new_image[y][x] = image[y1][x1]
  return new_image


graph_image = imread("./data/graf/graf_a.jpg")
new_graph_image = myWarpPerspective(graph_image, H_graph)
graph_image_2 = imread("./data/graf/graf_b.jpg")
fig, ax = plt.subplots(2, 2)
fig.suptitle("myWarpPerspective")
ax[0,0].imshow(new_graph_image)
ax[0,1].imshow(graph_image_2)
ax[1,0].imshow(graph_image)
ax[1,1].imshow(graph_image_2)
plt.show()
