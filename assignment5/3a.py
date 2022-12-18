import numpy as np
import matplotlib.pyplot as plt

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
