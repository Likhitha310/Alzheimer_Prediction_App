import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dummy data for example purposes (replace with your actual data)
X = np.array([[65, 0, 20], [70, 1, 30], [80, 0, 25], [75, 1, 35]])
y = np.array([0, 1, 0, 1])

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model to the 'model' folder
joblib.dump(model, 'model/alzheimer_model.pkl')
