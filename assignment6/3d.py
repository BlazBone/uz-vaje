import numpy as np
import matplotlib.pyplot as plt
import os
from UZ_utils import imread_gray

def prepare_data(data_path):
  images = np.array([])
  for image in os.listdir(data_path):
    img = imread_gray(data_path + image)
    img = np.reshape(img, (8064,1))
    #append to the images
    if images.size == 0:
      images = img
    else:
      images = np.hstack((images, img))
  return images

def dualPCA(points, n_components):
    """
    :param points: 2 x N matrix of points
    :param n_components: number of components to keep
    :return: U, eigenvalues, mean
    """
    X = points.T.copy()
    mean = np.mean(X, axis=1)
    #center the data
    X = X - mean[:, np.newaxis]
    # dual covariance matrix
    cov2 = (1/(X.shape[1] - 1)) * (X.T @ X)
    # SVD
    U,S,V_t = np.linalg.svd(cov2)
    # S = S + 1e-13
    #basis eigenvectors space
    U = X @ U @ np.sqrt(np.diag(1/(S * (X.shape[1]-1))))
    return U, S, V_t, mean 

def main():
  image_size = (96,84)
  # get the average face from the dataset
  data = prepare_data("./data/faces/2/")

  fig = plt.figure()
  view = fig.add_subplot(111)
  plt.ion()
  fig.show()

  # mean of the data
  mean = np.mean(data, axis=1)
  # pick image 1
  image = data[:,0]
  x = np.linspace(-10, 10, 100)
  sinx = np.sin(x)
  cosx = np.cos(x)
  U, S, V_t, mean = dualPCA(data.T, 1)
  
  for sin,cos in zip(sinx, cosx):
    # project the first image to the subspace space. 
    # The first eigenvector is the most important one
    y_i = U.T @ (image - mean)
    # Then, select one of the more important eigenvectors and manually set its corresponding weight in the projected vector to some value of your choice. 
    # For example, you can set the weight of the first eigenvector to 0.
    y_i[0] = sin * 3000
    y_i[1] = cos * 3000
    # reconstruct the first image
    # Project the modified vector back to image space and observe the change.
    recon = U @ y_i + mean
    recon = np.reshape(recon, image_size)
    view.clear()
    view.imshow(recon, cmap='gray')
    plt.pause(0.05)
    fig.canvas.draw()


if __name__ == "__main__":
  main()