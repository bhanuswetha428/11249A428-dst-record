DATA EXPLAORATION AND PREPROCESSING 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# STEP 1: CREATE DATASET
data = {
    'Country': ['India', 'USA', 'India', 'USA', 'UK', 'India'],
    'Age': [22, 25, np.nan, 30, 28, 35],
    'Salary': [40000, 60000, 50000, np.nan, 72000, 58000],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

print("--- ORIGINAL RAW DATA ---")
print(df)
print("\n")

# STEP 2: CHECK MISSING VALUES
print("--- MISSING VALUES COUNT ---")
print(df.isnull().sum())
print("\n")

# STEP 3: HANDLE MISSING VALUES
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

print("--- DATA AFTER CLEANING ---")
print(df)
print("\n")

# STEP 4: ENCODING
df_encoded = pd.get_dummies(df, columns=['Country'])
df_encoded['Purchased'] = df_encoded['Purchased'].map({'Yes': 1, 'No': 0})

print("--- DATA AFTER ENCODING ---")
print(df_encoded)
print("\n")

# STEP 5: FEATURE SCALING
X = df_encoded.drop('Purchased', axis=1)
y = df_encoded['Purchased']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("--- FINAL PROCESSED DATA ---")
print(pd.DataFrame(X_scaled, columns=X.columns).head())
OUTPUT
--- ORIGINAL RAW DATA ---
  Country   Age   Salary Purchased
0   India  22.0  40000.0        No
1     USA  25.0  60000.0       Yes
2   India   NaN  50000.0       Yes
3     USA  30.0      NaN        No
4      UK  28.0  72000.0       Yes
5   India  35.0  58000.0       Yes
--- MISSING VALUES COUNT ---
Country      0
Age          1
Salary       1
Purchased    0
dtype: int64
--- DATA AFTER CLEANING ---
  Country   Age   Salary Purchased
0   India  22.0  40000.0        No
1     USA  25.0  60000.0       Yes
2   India  28.0  50000.0       Yes
3     USA  30.0  56000.0        No
4      UK  28.0  72000.0       Yes
5   India  35.0  58000.0       Yes
--- DATA AFTER ENCODING ---
    Age   Salary  Purchased  Country_India  Country_UK  Country_USA
0  22.0  40000.0          0           True       False        False
1  25.0  60000.0          1          False       False         True
2  28.0  50000.0          1           True       False        False
3  30.0  56000.0          0          False       False         True
4  28.0  72000.0          1          False        True        False
5  35.0  58000.0          1           True       False        False
--- FINAL PROCESSED DATA ---
        Age    Salary  Country_India  Country_UK  Country_USA
0 -1.484615 -1.644453       1.0       -0.447214     -0.707107
1 -0.742307  0.411113      -1.0       -0.447214      1.414214
2  0.000000 -0.616670       1.0       -0.447214     -0.707107
3  0.494872  0.000000      -1.0       -0.447214      1.414214
4  0.000000  1.644453      -1.0        2.236068     -0.707107
