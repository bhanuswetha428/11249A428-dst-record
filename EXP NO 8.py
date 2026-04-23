SVM KERNELS

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear = SVC(kernel='linear')
poly = SVC(kernel='poly')
rbf = SVC(kernel='rbf')

linear.fit(X_train, y_train)
poly.fit(X_train, y_train)
rbf.fit(X_train, y_train)

print("Linear:", accuracy_score(y_test, linear.predict(X_test)))
print("Poly:", accuracy_score(y_test, poly.predict(X_test)))
print("RBF:", accuracy_score(y_test, rbf.predict(X_test)))

plt.bar(["Linear","Poly","RBF"], [
    accuracy_score(y_test, linear.predict(X_test)),
    accuracy_score(y_test, poly.predict(X_test)),
    accuracy_score(y_test, rbf.predict(X_test))
])
plt.show()

OUTPUT
Linear: 0.9666666666666667
Poly: 0.9333333333333333
RBF: 0.9333333333333333
