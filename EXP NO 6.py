ENSEMBLE TECHNIQUE
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression(max_iter=3000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

model = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('rf', rf)],
    voting='hard'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

plt.bar(["Voting"], [acc])
plt.show()

OUTPUT
Accuracy: 0.9777777777777777
