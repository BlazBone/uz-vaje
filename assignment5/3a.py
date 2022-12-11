import numpy as np
import matplotlib.pyplot as plt

def triangulate(points1, points2, cal_mtx1, cal_mtx2):
  T = np.array([[-1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]])
  # convert points to homogeneous coordinates
  points1 = np.array([points1[0], points1[1], 1])
  points2 = np.array([points2[0], points2[1], 1])

  points_1 = T @ points1
  points_2 =  T @ points2

  # points to matrix
  a= points1
  points1 = np.array([[0, -1, a[1]],
                      [1, 0, -a[0]],
                      [-a[1], a[0], 0]]) 
  b = points2
  points2 = np.array([[0, -1, b[1]],
                      [1, 0, -b[0]],
                      [-b[1], b[0], 0]])
  A1 =points1 @ cal_mtx1
  A2 = points2 @ cal_mtx2
  A = np.vstack((A1, A2))
  
  # decompose A into U, D, V
  U, D, V = np.linalg.svd(A)
  # X is the last column of V
  X = V[-1,:]
  X = X / X[-1]
  return X[:3]

#callibration matrix
camera1 = np.loadtxt("./data/epipolar/house1_camera.txt")
camera2 = np.loadtxt("./data/epipolar/house2_camera.txt")

# house points
points = np.loadtxt("./data/epipolar/house_points.txt")
points_1 = points[:, :2]
points_2 = points[:, 2:]


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

# triangulate points
X = []
for i in range(points_1.shape[0]):
  X.append(triangulate(points_1[i], points_2[i], camera1, camera2))
  point = X[i]
  ax.scatter(point[0], point[1], point[2])



X = np.array(X)
print(X.shape)
print(X)


# plt.scatter(X[:,0],X[:,1],X[:,2])
# plt.show()




# ax.scatter(2,3,4) # plot the point (2,3,4) on the figure
plt.show()