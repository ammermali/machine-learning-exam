import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('regression2.csv', header=None, names=['X1', 'X2', 'Y'])
X = data[['X1', 'X2']].values
Y = data['Y'].values


# Plotting of the histogram
plt.figure(figsize=(8, 5))
plt.hist(data['Y'], bins=100, color='skyblue', edgecolor='black')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('histogram_y.png', bbox_inches='tight')
plt.show()

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['X1'], data['X2'], data['Y'], c=data['Y'], cmap='tab10', s=10, alpha=0.8)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.savefig('scatter_3d.png', bbox_inches='tight')
plt.show()

# 2D Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data['X1'], data['X2'], c=data['Y'], cmap='tab10', s=10, alpha=0.9)
plt.xlabel('X1')
plt.ylabel('X2')
plt.savefig('original_data_2d.png', bbox_inches='tight')
plt.show()


# Converts floats to string to treat them as such
y_train_classes = Y.astype(str)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X, y_train_classes)

# Sets up the grid for results
step = 0.25
grid_range = np.arange(-45, 45 + step, step)
X1_grid, X2_grid = np.meshgrid(grid_range, grid_range)

grid_points = np.c_[X1_grid.ravel(), X2_grid.ravel()]
y_preds_str = classifier.predict(grid_points)

# Converts back to float
y_preds_float = y_preds_str.astype(float)


result_df = pd.DataFrame({
    0: grid_points[:, 0],
    1: grid_points[:, 1],
    2: y_preds_float
})
result_df.to_csv("result3.csv", index=False, header=False)

plt.figure(figsize=(10, 8))

plt.pcolormesh(X1_grid, X2_grid, y_preds_float.reshape(X1_grid.shape), cmap='tab10', shading='auto')
plt.xlabel('X1')
plt.ylabel('X2')
plt.savefig('predicted_grid_2d.png', bbox_inches='tight')
plt.show()