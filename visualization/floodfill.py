import numpy as np
import numpy.random as npr
from scipy.stats import mode
import matplotlib.pyplot as plt


npr.seed(1)

height = 40
width = 60
size = [height, width]
low = 6
high = 12


def get_neighbors(node):
    neighbors = []
    for i, j in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        a, b = node[0]+i, node[1]+j
        if 0 <= a <= height-1 and 0 <= b <= width-1:
            neighbors.append((a, b))
    return neighbors


def smear(A, p=0.8, n=None):
    B = np.copy(A)
    if n is None:
        n = 8*B.size
    for i in range(n):
        if npr.rand() > p:
            node = (npr.randint(height), npr.randint(width))
            B[node] = mode([B[neighbor] for neighbor in get_neighbors(node)]).mode.item()
    return B


def floodfill(image, node, fill_value):
    visited = set()

    def touch(node):
        original_color = image[node]
        image[node] = fill_value
        for neighbor in get_neighbors(node):
            if image[neighbor] == original_color and neighbor not in visited:
                visited.add(neighbor)
                touch(neighbor)

    touch(node)


A = npr.randint(low=low, high=high+1, size=size).astype(float)
B = smear(A)
F = np.copy(B)
floodfill(F, node=(20, 10), fill_value=0)
floodfill(F, node=(30, 40), fill_value=3)

fig, ax = plt.subplots(3)
cmap = 'RdBu'
ax[0].imshow(A, cmap=cmap, vmin=0, vmax=high)
ax[1].imshow(B, cmap=cmap, vmin=0, vmax=high)
ax[2].imshow(F, cmap=cmap, vmin=0, vmax=high)
