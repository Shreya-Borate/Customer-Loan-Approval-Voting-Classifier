import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# STEP 1 : LOAD THE DATASET
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 1 : LOAD THE DATASET")
print("=" * 80)

DataPath = "../data/customer_loan_approval.csv"

df = pd.read_csv(DataPath)

print("\nDataset loaded successfully.")

print("\nFirst 5 records : ")
print(df.head())

# For Logistic Regression and KNN, scaling is important.


# =============================================================================
# STEP 2 : UNDERSTAND THE DATASET
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 2 : UNDERSTAND THE DATASET")
print("=" * 80)

print("\nDataset Shape : ")
print(df.shape)

print("\nDataset columns : ")
print(list(df.columns))

print("\nDataset Information : ")
print(df.info())

print("\nDataset Description : ")
print(df.describe())


# =============================================================================
# STEP 3 : CHECK FOR MISSING VALUES
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 3 : CHECK FOR MISSING VALUES")
print("=" * 80)

print("Missing values in eacy columns : ")
print(df.isnull().sum())


# =============================================================================
# STEP 4 : SEPARATE INPUT AND OUTPUT VARIABLES
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 4 : SEPARATE INPUT AND OUTPUT VARIABLES")
print("=" * 80)

# X : Independent Variables / Features
# Y : Dependent Variable / Target

X = df.drop("LoanApproved",axis=1)
Y = df['LoanApproved']

print("\nFeature columns : ")
print(list(X.columns))

print("\nTarget Variable : ")
print("LoanApproved")

print("\nFeature Shape : ",X.shape)
print("\nTarget Shape : ",Y.shape)


# =============================================================================
# STEP 5 : TRAIN-TEST SPLIT
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 5 : TRAIN-TEST SPLIT")
print("=" * 80)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y 
    #stratify=Y helps maintain approximately the same class proportion in both training and testing datasets.                 
)

print("\nTraning Datashape : ")
print(x_train.shape)

print("\nTesting Datashape : ")
print(x_test.shape)

print("Traning Target shape : ")
print(y_train.shape)

print("Testing Target shape : ")
print(y_test.shape)


# =============================================================================
# STEP 6 : FEATURE SCALING
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 6 : FEATURE SCALING")
print("=" * 80)

Scaler = StandardScaler()

x_train_scaled = Scaler.fit_transform(x_train)
x_test_scaled = Scaler.transform(x_test)

print("\nFeature scaling completed successfully.")

print("\nTraining data shape after scaling :")
print(x_train_scaled.shape)

print("\nTesting data shape after scaling :")
print(x_test_scaled.shape)

