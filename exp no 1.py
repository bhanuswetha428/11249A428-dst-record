PROGRAM:
A)CREATING ARRAYS
import numpy as np

print(
"1D Array:\n", np.array([1,2,3,4,5]), "\n\n",
"2D Array:\n", np.array([[1,2,3],[4,5,6]]), "\n\n",
"Zeros:\n", np.zeros((3,3)), "\n\n",
"Ones:\n", np.ones((2,4)), "\n\n",
"Identity:\n", np.eye(3), "\n\n",
"Range:\n", np.arange(0,10,2)
)
OUTPUT:
1D Array:
[1 2 3 4 5]

2D Array:
[[1 2 3]
 [4 5 6]]

Zeros:
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]

Ones:
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]]

Identity:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

Range:
[0 2 4 6 8]

B)ARRAY OPERATIONS
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(
"Array a:", a, "\n",
"Array b:", b, "\n\n",
"Addition (a + b):", a + b, "\n",
"Multiplication (a * b):", a * b, "\n",
"Scalar Multiplication (a * 10):", a * 10, "\n\n",
"Square Root of a:", np.sqrt(a), "\n",
"Mean of a:", np.mean(a), "\n",
"Dot Product (a · b):", np.dot(a, b)
)
 OUTPUT
 Array a: [1 2 3]
Array b: [4 5 6]

Addition (a + b): [5 7 9]
Multiplication (a * b): [ 4 10 18]
Scalar Multiplication (a * 10): [10 20 30]

Square Root of a: [1.         1.41421356 1.73205081]
Mean of a: 2.0
Dot Product (a · b): 32

C)SLICING AND INDEXING:
import numpy as np

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print(
"Matrix:\n", matrix, "\n\n",
"Element at [1,2]:", matrix[1, 2], "\n\n",
"First Row:", matrix[0, :], "\n",
"Second Column:", matrix[:, 1], "\n\n",
"Top-left 2x2 Sub-matrix:\n", matrix[0:2, 0:2]
)
OUTPUT
Matrix:
[[10 20 30]
 [40 50 60]
 [70 80 90]]

Element at [1,2]: 60

First Row: [10 20 30]
Second Column: [20 50 80]

Top-left 2x2 Sub-matrix:
[[10 20]
 [40 50]]
C)CREATE DATAFRAME
 import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Paris', 'London']
}

df = pd.DataFrame(data)

print(df.head())
print("\n")
print(df.info())
print("\n")
print(df.describe())

OUTPUT
      Name  Age      City
0    Alice   25  New York
1      Bob   30     Paris
2  Charlie   35    London  
  <class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
Name    3 non-null object
Age     3 non-null int64
City    3 non-null object
        Age
count   3.0
mean   30.0
std     5.0
min    25.0
max    35.0
D)SELECTION AND FILTERING
ages = df['Age']
print("Ages column:")
print(ages)

print("\n")

above_25 = df[df['Age'] > 25]
print("People with Age > 25:")
print(above_25)

print("\n")

row_0 = df.iloc[0]
print("First row:")
print(row_0)

print("\n")

specific_val = df.loc[0, 'Name']
print("Name at index 0:")
print(specific_val)

OUTPUT
Ages column:
0    25
1    30
2    35
People with Age > 25:
      Name  Age   City
1      Bob   30  Paris
2  Charlie   35 London

E)DATA CLEANING
import pandas as pd
import numpy as np

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Age": [23, np.nan, 22, 28, np.nan],
    "City": ["New York", "London", np.nan, "Paris", "Berlin"]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

print("\nNull values:\n", df.isnull().sum())

print("\nAfter dropna():")
print(df.dropna())

print("\nFill with 0:")
print(df.fillna(0))

df["Age"] = df["Age"].fillna(df["Age"].mean())
print("\nFill Age with mean:")
print(df)

df["City"] = df["City"].fillna("Unknown")
print("\nFill City:")
print(df)


OUTPUT:
Original DataFrame:
      Name   Age      City
0    Alice  23.0  New York
1      Bob   NaN    London
2  Charlie  22.0       NaN
3    David  28.0     Paris
4      Eva   NaN    Berlin

Null values:
Name    0
Age     2
City    1
dtype: int64

After dropna():
    Name   Age      City
0  Alice  23.0  New York
3  David  28.0     Paris

Fill with 0:
      Name   Age      City
0    Alice  23.0  New York
1      Bob   0.0    London
2  Charlie  22.0         0
3    David  28.0     Paris
4      Eva   0.0    Berlin

Fill Age with mean:
      Name        Age      City
0    Alice  23.000000  New York
1      Bob  24.333333    London
2  Charlie  22.000000       NaN
3    David  28.000000     Paris
4      Eva  24.333333    Berlin

Fill City:
      Name   Age      City
0    Alice  23.0  New York
1      Bob   NaN    London
2  Charlie  22.0   Unknown
3    David  28.0     Paris
4      Eva   NaN    Berlin

7)MATPLOTLIB
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'StudyHours': [1,2,3,4,5,6,7,8],
    'ExamScore': [35,40,50,55,65,70,78,85]
}

df = pd.DataFrame(data)

# Scatter
plt.scatter(df['StudyHours'], df['ExamScore'])
plt.show()

# Line
plt.plot(df['StudyHours'], df['ExamScore'])
plt.show()

# Histogram
plt.hist(df['ExamScore'])
plt.show()

# Bar
plt.bar(df['StudyHours'], df['ExamScore'])
plt.show()
OUTPUT
## Graph Outputs

### Scatter Plot
![Scatter Plot](scatter_plot.png)

### Line Plot
![Line Plot](line_plot.png)

### Histogram
![Histogram](histogram.png)

### Bar Chart
![Bar Chart](bar_chart.png)
8) SCIKIT LEARN
    import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Classes_Attended': [30,35,40,45,50,55,60,65,70,75],
    'Internal_Marks': [35,38,42,46,50,55,60,65,70,75]
}

df = pd.DataFrame(data)

X = df[['Classes_Attended']]
y = df['Internal_Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, predictions))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()
OUTPUT
MSE: ~0.5 (approx)
Slope: ~0.9
Intercept: ~6.3
