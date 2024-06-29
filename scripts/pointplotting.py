import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# BRISK feature: left screen botom left corner
k000 = [-0.24295129, -0.18650152,  0.17425673]
k060 = [-0.24444155, -0.19517508,  0.19492936]
k064 = [-0.23474595, -0.17455789,  0.17460956]
k070 = [-0.23474595, -0.17455789,  0.17460956]
k076 = [-0.23376344, -0.16826524,  0.17070825]
k090 = [-0.29532263, -0.19570422,  0.18614803]
k126 = [-0.23474595, -0.17455789,  0.17460956]
k177 = [-0.23100455, -0.1857615,   0.18019448]
k216 = [-0.2409305,  -0.17810372,  0.17129338]
k245 = [-0.32481399, -0.19584675,  0.1739295 ]



# Sample 3D points
points = np.array([
    k000,
    k060,
    k064,
    k070,
    k076,
    k090,
    k126,
    k177,
    k216,
    k245,
])

# Euclidean distances
def euclidean_distance_matrix(points):
    dist_matrix = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
    return dist_matrix

euclidean_distances = euclidean_distance_matrix(points)

# Mean and standard deviation
mean = np.mean(points, axis=0)
std_dev = np.std(points, axis=0)

# Z-scores
z_scores = (points - mean) / std_dev

# Mahalanobis distance
cov_matrix = np.cov(points, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)
mahalanobis_distances = [mahalanobis(point, mean, inv_cov_matrix) for point in points]

# DBSCAN
scaler = StandardScaler()
points_scaled = scaler.fit_transform(points)
dbscan = DBSCAN(eps=1.5, min_samples=2)
dbscan.fit(points_scaled)
labels = dbscan.labels_  # -1 indicates outliers

# Plotting
fig = plt.figure(figsize=(18, 10))

# 3D Scatter Plot
ax = fig.add_subplot(231, projection='3d')
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='bwr', s=50)
ax.set_title('3D Scatter Plot with DBSCAN Clustering')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

# Pairwise Distance Heatmap
ax = fig.add_subplot(232)
sns.heatmap(euclidean_distances, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
ax.set_title('Pairwise Euclidean Distance Heatmap')
ax.set_xlabel('Point Index')
ax.set_ylabel('Point Index')

# Z-score Distribution Plot
ax = fig.add_subplot(233)
sns.histplot(z_scores.flatten(), kde=True, ax=ax)
ax.set_title('Z-score Distribution')
ax.set_xlabel('Z-score')
ax.set_ylabel('Frequency')

# Mahalanobis Distance Plot
ax = fig.add_subplot(234)
sns.scatterplot(x=np.arange(len(mahalanobis_distances)), y=mahalanobis_distances, hue=(np.array(mahalanobis_distances) > 3).astype(int), palette=['blue', 'red'], ax=ax)
ax.axhline(y=3, color='r', linestyle='--')
ax.set_title('Mahalanobis Distances')
ax.set_xlabel('Point Index')
ax.set_ylabel('Mahalanobis Distance')
legend2 = ax.legend(['Threshold', 'Inliers', 'Outliers'], title="Legend")
ax.add_artist(legend2)

# 3D Scatter Plot Highlighting Outliers
ax = fig.add_subplot(235, projection='3d')
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.array(mahalanobis_distances) > 3, cmap='coolwarm', s=50)
ax.set_title('3D Scatter Plot Highlighting Outliers')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
legend3 = ax.legend(*scatter.legend_elements(), title="Outliers")
ax.add_artist(legend3)

# DBSCAN Clustering Plot in 2D (for illustrative purposes)
ax = fig.add_subplot(236)
scatter = ax.scatter(points_scaled[:, 0], points_scaled[:, 1], c=labels, cmap='viridis', s=50)
ax.set_title('2D Projection with DBSCAN Clustering')
ax.set_xlabel('Scaled X')
ax.set_ylabel('Scaled Y')
legend4 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend4)

plt.tight_layout()
plt.show()
