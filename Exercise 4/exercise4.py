import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

df = pd.read_csv('classification2.csv', header=None, names=['x', 'y', 'class'])

# Plotting of the original data

class_0 = df[df['class'] == 0.0]
class_1 = df[df['class'] == 1.0]
plt.figure(figsize=(10, 8))
plt.scatter(class_0['x'], class_0['y'], color='blue', label='Class 0', alpha=0.5, s=15)
plt.scatter(class_1['x'], class_1['y'], color='red', label='Class 1', alpha=0.5, s=15)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('original_plot.png', bbox_inches='tight')
plt.close()

# Training of SVM
X = df[['x', 'y']].values
y = df['class'].values
classifier = SVC(kernel='rbf', probability=True)
classifier.fit(X, y)


# Sets up the grid
x_range = np.arange(-45.0, 45.25, 0.25)
y_range = np.arange(-45.0, 45.25, 0.25)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.c_[xx.ravel(), yy.ravel()]
preds = classifier.predict(grid_points)
probs = classifier.predict_proba(grid_points)

print("Loading data...")
# 1. Load the dataset (assuming no headers, columns are x, y, class)
df = pd.read_csv('classification2.csv', header=None, names=['x', 'y', 'class'])
X = df[['x', 'y']].values
y = df['class'].values

# 2. Initialize and train the SVM model
# We use the RBF kernel which is great for overlapping, non-linear data.
# Setting probability=True applies Platt scaling so we can study class likelihoods.
clf = SVC(kernel='rbf', probability=True)
clf.fit(X, y)
x_range = np.arange(-45.0, 45.25, 0.25)
y_range = np.arange(-45.0, 45.25, 0.25)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.c_[xx.ravel(), yy.ravel()]
preds = clf.predict(grid_points)
probs = clf.predict_proba(grid_points)
result_df = pd.DataFrame(grid_points, columns=['x', 'y'])
result_df['class'] = preds
result_df.to_csv('result4.csv', index=False, header=False)


# Plot the probabilities
plt.figure(figsize=(10, 8))
prob_class_1 = probs[:, 1].reshape(xx.shape)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('svm_probabilities.png', bbox_inches='tight')
plt.close()