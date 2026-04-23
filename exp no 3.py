REGULARIZED LINEAR REGRESSION
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt

np.random.seed(42)

# CREATE DATASET
X = np.random.randn(40, 15)

columns = [
    "House_Size","Bedrooms","Location_Rating","Age_of_House",
    "Wall_Color","Owner_Lucky_Number","Street_Length",
    "Nearby_Trees","Pet_Count","Car_Model_Code",
    "Random_Noise_1","Random_Noise_2","Random_Noise_3",
    "Random_Noise_4","Random_Noise_5"
]

df = pd.DataFrame(X, columns=columns)

# TARGET VARIABLE
df["House_Price"] = (
    15 * df["House_Size"] +
    8 * df["Bedrooms"] +
    12 * df["Location_Rating"] -
    5 * df["Age_of_House"] +
    np.random.randn(40) * 5
)

# PRINT DATA
print("===== DATASET SAMPLE =====")
print(df.head())

# SPLIT
X = df.drop("House_Price", axis=1)
y = df["House_Price"]

# TRAIN MODELS
linear_model = LinearRegression()
linear_model.fit(X, y)

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X, y)

# PRINT RESULTS
print("\n===== Linear Regression Coefficients =====")
print(pd.Series(np.round(linear_model.coef_, 2), index=X.columns))

print("\n===== Lasso Regression Coefficients =====")
print(pd.Series(np.round(lasso_model.coef_, 2), index=X.columns))

# PLOT
plt.figure(figsize=(14,6))
plt.bar(X.columns, linear_model.coef_, alpha=0.5, label='Linear')
plt.bar(X.columns, lasso_model.coef_, alpha=0.8, width=0.4, label='Lasso')
plt.xticks(rotation=90)
plt.axhline(0)
plt.legend()
plt.title("Real Features vs Noise Features")
plt.show()

OUTPUT
===== DATASET SAMPLE =====
   House_Size  Bedrooms  Location_Rating  Age_of_House  Wall_Color  ...  Random_Noise_5  House_Price
0    0.496714 -0.138264         0.647689       1.523030   -0.234153  ...       -1.724918     10.286654
1   -0.562288 -1.012831         0.314247      -0.908024   -1.412304  ...       -0.291694    -12.836700
2   -0.601707  1.852278        -0.013497      -1.057711    0.822545  ...       -1.478522     15.267244
3   -0.719844 -0.460639         1.057122       0.343618   -1.763040  ...        0.975545      3.262791
4   -0.479174 -0.185659        -1.106335      -1.196207    0.812526  ...       -2.619745    -13.900697
===== Linear Regression Coefficients =====
House_Size            16.55
Bedrooms               9.26
Location_Rating       11.33
Age_of_House          -5.22
Wall_Color            -0.49
Owner_Lucky_Number     1.54
Street_Length          0.47
Nearby_Trees          -0.69
Pet_Count              0.59
Car_Model_Code         1.40
Random_Noise_1         1.33
Random_Noise_2         0.88
Random_Noise_3         2.02
Random_Noise_4        -1.17
Random_Noise_5        -0.50
dtype: float64
===== Lasso Regression Coefficients =====
House_Size            15.00
Bedrooms               7.29
Location_Rating       10.52
Age_of_House          -3.60
Wall_Color             0.00
Owner_Lucky_Number     0.00
Street_Length          0.00
Nearby_Trees          -0.00
Pet_Count              0.00
Car_Model_Code         0.00
Random_Noise_1         0.50
Random_Noise_2         0.00
Random_Noise_3         0.02
Random_Noise_4        -0.00
Random_Noise_5        -0.00
dtype: float64
