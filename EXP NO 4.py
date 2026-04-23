NAIVE BAYES CLASSIFIER
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# LOAD DATA
data = pd.read_csv('temp_hum_play_data.csv')

# ENCODE TARGET
le = LabelEncoder()
data['Play'] = le.fit_transform(data['Play'])

# SPLIT
X = data[['Temperature', 'Humidity']]
y = data['Play']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = GaussianNB()
model.fit(X_train, y_train)

# PREDICT
prediction = model.predict(X_test)
all_predictions = model.predict(X)

# ACCURACY
print("Accuracy:", accuracy_score(y_test, prediction))

# PARAMETERS
print("\nClass Priors:", model.class_prior_)
print("\nMean (theta):\n", model.theta_)
print("\nVariance:\n", model.var_)

# GRAPH
plt.scatter(X['Temperature'], X['Humidity'], c=y, marker='o', label='Actual')
plt.scatter(X['Temperature'], X['Humidity'], c=all_predictions, marker='x', label='Predicted')
plt.legend()
plt.show()

OUTPUT
Accuracy: 1.0

Class Priors: [0.375 0.625]

Mean (theta):
[[30.66666667 89.33333333]
 [24.         75.        ]]

Variance:
[[10.88888895 17.55555562]
 [ 2.00000006 13.60000006]]
