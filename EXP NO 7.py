DECISION TREE CLASSIFIER

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("/content/placement_data.csv")

# Features and Target
X = data[['cgpa','communication_skills','projects_completed', 'iq','internship_experience']]
y = data['placement']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Predict new candidate
new_candidate = pd.DataFrame([[8.1, 7, 3, 110, 1]], columns=X.columns)
prediction = model.predict(new_candidate)

if prediction[0] == 1:
    print("Candidate will likely be PLACED")
else:
    print("Candidate will likely NOT be placed")

# Visualize the Tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree for Placement Prediction")
plt.show()

OUTPUT
Model Accuracy: 1.0
Candidate will likely be PLACED

DECISION TREE REGRESSOR

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("/content/placemnet_salary.csv")

# Convert categorical variables to numerical
data['workex'] = data['workex'].map({'Yes': 1, 'No': 0})
data['specialisation'] = data['specialisation'].map({'Mkt&HR': 0, 'Mkt&Fin': 1})

# Features and Target
X = data[['ssc_p','hsc_p','degree_p','etest_p','mba_p','workex', 'specialisation']]
y = data['salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict salaries
y_pred = model.predict(X_test)

# Model performance
score = r2_score(y_test, y_pred)
print("Model R2 Score:", score)

# Predict salary for new candidate
new_candidate = pd.DataFrame([[85, 88, 82, 75, 80, 1, 1]], columns=X.columns)
predicted_salary = model.predict(new_candidate)
print("Predicted Salary:", predicted_salary[0])

# Visualize Results
plt.figure()
plt.scatter(X_test['mba_p'], y_test)
plt.xlabel("MBA Percentage")
plt.ylabel("Actual Salary")
plt.title("MBA % vs Salary")
plt.show

OUTPUT
Model R2 Score: 0.6721854304635762
Predicted Salary: 370000.0
