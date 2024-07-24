import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import det, qr
import math
from scipy.linalg import svd
from scipy.stats import rv_discrete
import time


### Functions to Fix Matrix size


def gram_calculate_volume(simplex):
    # Subtract the first vertex from all other vertices
    simplex = simplex[1:] - simplex[0]
    # Compute the Gram matrix
    gram_matrix = np.dot(simplex, simplex.T)
    # Calculate the volume
    return np.sqrt(np.abs(det(gram_matrix))) / math.factorial(simplex.shape[0]-1)

def find_furthest_rows(A):
    # Calculate the pairwise distances
    distances = pdist(A, 'euclidean')
    # Get the index of the maximum distance
    max_idx = np.argmax(distances)
    # Calculate the total number of elements in the distance matrix
    n = len(distances)
    # Calculate the number of rows in the original matrix
    m = int((-1 + np.sqrt(1 + 8 * n)) // 2) + 1
    # Convert the index in the condensed distance matrix to indices in the original matrix
    i = m - 2 - int((np.sqrt(-8 * max_idx + 4 * m * (m - 1) - 7) - 1) // 2)
    j = int(max_idx + i + 1 - m * (m - 1) // 2 + (m - i) * ((m - i) - 1) // 2)
    return i, j

def find_largest_simplex(matrix,k):
	'''
	Find the largest simplex in a set of points

	Parameters:
		matrix (array): array of points to find the largest simplex in

	Returns:
		order_indicies (array): array of indicies of the points in the largest simplex
	
	Example:
		.. code-block:: python

			matrix = np.array([[0,0],[1,0],[0,1],[1,1]])
			find_largest_simplex(matrix)
	'''
	matrix = np.asarray(matrix)
	x0, x1 = find_furthest_rows(matrix)
	simplex = [x0, x1]
	simplex_matrix = matrix[simplex]
	order_indicies = []
	n=matrix.shape[0]
	for _ in range(k):
		max_volume = 0
		for i in range(n):
			if i in simplex:
				continue
			temp_simplex = np.vstack([simplex_matrix, matrix[i]])
        	# Calculate the volume
			volume = gram_calculate_volume(temp_simplex)
			if volume > max_volume:
				max_volume = volume
				max_volume_index = i
		simplex_matrix = np.vstack([simplex_matrix, matrix[max_volume_index]])
		order_indicies.append(max_volume_index)
	return matrix[order_indicies]


