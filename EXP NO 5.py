REGULARIZED LOGISTIC REGRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# LOAD DATA
data = load_breast_cancer()
X, y = data.data, data.target
features = data.feature_names

# SPLIT & SCALE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODELS
normal = LogisticRegression(penalty=None, max_iter=5000)
normal.fit(X_train, y_train)

ridge = LogisticRegression(penalty='l2', C=1.0, max_iter=5000)
ridge.fit(X_train, y_train)

# ACCURACY
print("Normal Logistic Accuracy:", accuracy_score(y_test, normal.predict(X_test)))
print("Ridge Accuracy:", accuracy_score(y_test, ridge.predict(X_test)))

# COEFFICIENTS
print("\nNormal Coefficients:")
print(pd.Series(np.round(normal.coef_[0], 3), index=features))

print("\nRidge Coefficients:")
print(pd.Series(np.round(ridge.coef_[0], 3), index=features))

# GRAPH
plt.plot(normal.coef_[0], label='Normal')
plt.plot(ridge.coef_[0], label='Ridge')
plt.legend()
plt.show()

OUTPUT
Normal Logistic Accuracy: 0.939
Ridge Accuracy: 0.974
