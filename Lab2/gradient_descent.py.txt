import pandas as pd
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

# uploaded = files.upload()

import pandas as pd

def compute_cost(x, y, theta_0, theta_1):
    m = len(x)
    x = np.array(x)
    predictions = theta_0 + theta_1 * x
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost
  
def gradient_descent(x,y, theta_0, theta_1):
  m = len(x)
  x = np.array(x)
  predictions = theta_0 + theta_1 * x
  d_0 = (1/m)*(np.sum(predictions - y))
  d_1 = (1/m)*(np.sum((predictions - y)*x))
  theta_0 -= learning_rate * d_0
  theta_1 -= learning_rate * d_1
  return theta_0, theta_1

# Read the dataset
data = pd.read_csv('linear_regression_dataset.csv')

# Clean and standardize column names
print(data.columns)
data.columns = data.columns.str.strip().str.lower()

# Extract 'hours_studied' and 'exam_score'
x = data['hours_studied'].tolist()
y = data['exam_score'].tolist()
theta_0 = 0.0
theta_1 = 0.0
learning_rate = 0.0007
iterations = 10000
  
for i in range(iterations):
  theta_0, theta_1 = gradient_descent(x,y,theta_0,theta_1)
  cost = compute_cost(x,y,theta_0,theta_1)
  minCost = float('inf')
  if(minCost>cost):
    minCost = cost
    minTheta_0 = theta_0
    minTheta_1 = theta_1
    
print("Minimum Cost: ",minCost)
print("Minimum Theta_0: ",minTheta_0)
print("Minimum Theta_1: ",minTheta_1)
x = np.array(x)
y_pred = minTheta_1 * x + minTheta_0

plt.scatter(x,y, color='blue', label='Actual data')
plt.plot(x,y_pred, color='black', label='Predicted line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Linear Regression')
plt.legend()
plt.show()


print("predicting the score, if studied for 6.32 hrs")
y = minTheta_1 * 6.32 + minTheta_0
print(y)