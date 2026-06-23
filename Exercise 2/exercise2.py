import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

# Loads the data and splits them
data = pd.read_csv('classification1.csv', header=None, names=['X1', 'X2', 'Class'])
X = data[['X1', 'X2']].values
y = data['Class'].values

# Plots the dataset
plt.figure(figsize=(10, 8))
colormap = ListedColormap(['#1f77b4', '#2ca02c', '#ff7f0e']) # blue, green, orange
scatter = plt.scatter(data['X1'], data['X2'], c=data['Class'], cmap=colormap, alpha=0.7, edgecolors='k')
plt.title('Visualization of classification1.csv')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('classification_plot.png')
plt.show()

# Applied k-NN
model = KNeighborsClassifier(n_neighbors=15, weights='distance')
model.fit(X, y)

# Sets up the grid for results
step = 0.25
grid_range = np.arange(-45, 45 + step, step)
xx, yy = np.meshgrid(grid_range, grid_range)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Creates predictions and probabilities for grid datapoints
predictions = model.predict(grid_points)
probabilities = model.predict_proba(grid_points)

# Generate the results and saves it
result_df = pd.DataFrame({
    'X1': grid_points[:, 0],
    'X2': grid_points[:, 1],
    'Class': predictions
})
result_df.to_csv('result2.csv', index=False, header=False)

colormaps = {
    0: 'Blues',
    1: 'Greens',
    2: 'Oranges'
}

# Plots the probability for every class
for CLASS in [0, 1, 2]:
    prob_target = probabilities[:, CLASS].reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, prob_target, cmap=colormaps[CLASS], shading='auto', vmin=0, vmax=1)
    plt.colorbar(label=f'Probability P(Class={CLASS})')

    plt.title(f'Probability Study: Class {CLASS}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig(f'class_{CLASS}_gradient.png')
    plt.show()


Z = predictions.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.pcolormesh(xx, yy, Z, cmap=colormap, shading='auto')
plt.title('Classification boundaries')
plt.xlabel('X1')
plt.ylabel('X2')
plt.savefig('decision_boundaries.png')
plt.show()