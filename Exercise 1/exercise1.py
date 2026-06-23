import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Load the data and split them into the X and the Y distribution

data = pd.read_csv('regression1.csv', header=None, names=['X', 'Y'])
X = data[['X']]
Y = data['Y']

# Plot and save the original visualization

plt.scatter(X, Y, alpha=0.5)
plt.title('Visualization of regression1.csv')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('original.png')
plt.show()

# Plot and save the scaled visualization

plt.scatter(X, Y, alpha=0.5)
plt.title('Visualization of regression1.csv')
plt.yscale('log')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('scaled.png')
plt.show()

# Implements the Linear Model Decision Tree
# Divides into segments the dataset and apply the linear regression to them

tree = DecisionTreeRegressor(max_depth=5) # max_depth is set to prevent overfitting
tree.fit(X, Y)
indices = tree.apply(X)
unique_indices = np.unique(indices)

y_pred = np.zeros_like(Y, dtype=float)

for index in unique_indices:
    mask = (indices == index)
    x_segment = X[mask]
    y_segment = Y[mask]

    regressor = LinearRegression()
    regressor.fit(x_segment, y_segment)
    y_pred[mask] = regressor.predict(x_segment)

results_df = pd.DataFrame(y_pred)
results_df.to_csv('result1.csv', index=False, header=False)

# Plots and saves the comparison between the dataset and the model
data = pd.read_csv('regression1.csv', header=None, names=['X', 'Y'])
results = pd.read_csv('result1.csv', header=None, names=['Y_pred'])

df = pd.concat([data, results], axis=1).sort_values(by='X')

x_plot = df['X'].values
y_actual = df['Y'].values
y_pred = df['Y_pred'].values

plt.figure(figsize=(10, 6))
plt.scatter(x_plot, y_actual, alpha=0.3, color='gray', label='Data')
plt.plot(x_plot, y_pred, color='red', lw=2, label='Model Prediction')
plt.yscale('log')
plt.title('Log-scaled comparison')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('results.png')
plt.show()