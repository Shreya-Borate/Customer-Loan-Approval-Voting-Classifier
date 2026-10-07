import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


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


# =============================================================================
# STEP 7 : BUILD LOGISTIC REGRESSION MODEL
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 7 : BUILD LOGISTIC REGRESSION MODEL")
print("=" * 80)

LogisticModel = LogisticRegression(random_state=42)

print("\nLogistic Regression model created successfully.")

# =============================================================================
# STEP 8 : TRAIN LOGISTIC REGRESSION MODEL
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 8 : TRAIN LOGISTIC REGRESSION MODEL")
print("=" * 80)

LogisticModel.fit(x_train_scaled, y_train)

print("\nLogistic Regression model trained successfully.")

# =============================================================================
# STEP 9 : MAKE PREDICTIONS
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 9 : MAKE PREDICTIONS")
print("=" * 80)

Y_pred = LogisticModel.predict(x_test_scaled)

print("\nPredictions made successfully.")

print("\nPredicted values :")
print(Y_pred)

print("\nActual values :")
print(y_test.values)

# =============================================================================
# STEP 10 : EVALUATE LOGISTIC REGRESSION MODEL
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 10 : EVALUATE LOGISTIC REGRESSION MODEL")
print("=" * 80)

Accuracy = accuracy_score(y_test, Y_pred)

print("\nLogistic Regression Accuracy :", Accuracy)
print("Logistic Regression Accuracy : {:.2f}%".format(Accuracy * 100))

# =============================================================================
# STEP 11 : CLASSIFICATION REPORT
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 11 : CLASSIFICATION REPORT")
print("=" * 80)

print("\nClassification Report :")
print(classification_report(y_test, Y_pred))

# =============================================================================
# STEP 12 : CONFUSION MATRIX
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 12 : CONFUSION MATRIX")
print("=" * 80)

ConfusionMatrix = pd.crosstab(
    y_test,
    Y_pred,
    rownames=["Actual"],
    colnames=["Predicted"]
)

print("\nConfusion Matrix :")
print(ConfusionMatrix)

# =============================================================================
# STEP 13 : BUILD DECISION TREE CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 13 : BUILD DECISION TREE CLASSIFIER")
print("=" * 80)

DecisionTreeModel = DecisionTreeClassifier(random_state=42)

print("\nDecision Tree Classifier created successfully.")
# =============================================================================
# STEP 14 : TRAIN DECISION TREE CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 14 : TRAIN DECISION TREE CLASSIFIER")
print("=" * 80)

DecisionTreeModel.fit(x_train, y_train)

print("\nDecision Tree Classifier trained successfully.")


# =============================================================================
# STEP 15 : MAKE DECISION TREE PREDICTIONS
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 15 : MAKE DECISION TREE PREDICTIONS")
print("=" * 80)

DecisionTreePred = DecisionTreeModel.predict(x_test)

print("\nDecision Tree predictions made successfully.")

print("\nPredicted values :")
print(DecisionTreePred)

print("\nActual values :")
print(y_test.values)

# =============================================================================
# STEP 16 : EVALUATE DECISION TREE CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 16 : EVALUATE DECISION TREE CLASSIFIER")
print("=" * 80)

DecisionTreeAccuracy = accuracy_score(y_test, DecisionTreePred)

print("\nDecision Tree Accuracy :", DecisionTreeAccuracy)
print("Decision Tree Accuracy : {:.2f}%".format(DecisionTreeAccuracy * 100))


# =============================================================================
# STEP 17 : BUILD K-NEAREST NEIGHBORS CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 17 : BUILD K-NEAREST NEIGHBORS CLASSIFIER")
print("=" * 80)

KNNModel = KNeighborsClassifier(n_neighbors=5)

print("\nK-Nearest Neighbors Classifier created successfully.")

# =============================================================================
# STEP 18 : TRAIN K-NEAREST NEIGHBORS CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 18 : TRAIN K-NEAREST NEIGHBORS CLASSIFIER")
print("=" * 80)

KNNModel.fit(x_train_scaled, y_train)

print("\nK-Nearest Neighbors Classifier trained successfully.")

# =============================================================================
# STEP 19 : MAKE KNN PREDICTIONS
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 19 : MAKE KNN PREDICTIONS")
print("=" * 80)

KNNPred = KNNModel.predict(x_test_scaled)

print("\nKNN predictions made successfully.")

print("\nPredicted values :")
print(KNNPred)

print("\nActual values :")
print(y_test.values)

# =============================================================================
# STEP 20 : EVALUATE KNN CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 20 : EVALUATE KNN CLASSIFIER")
print("=" * 80)

KNNAccuracy = accuracy_score(y_test, KNNPred)

print("\nKNN Accuracy :", KNNAccuracy)
print("KNN Accuracy : {:.2f}%".format(KNNAccuracy * 100))

# =============================================================================
# STEP 21 : CREATE HARD VOTING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 21 : CREATE HARD VOTING CLASSIFIER")
print("=" * 80)

HardVotingModel = VotingClassifier(
    estimators=[
        ("Logistic Regression", LogisticModel),
        ("Decision Tree", DecisionTreeModel),
        ("KNN", KNNModel)
    ],
    voting="hard"
)

print("\nHard Voting Classifier created successfully.")

# =============================================================================
# STEP 22 : TRAIN HARD VOTING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 22 : TRAIN HARD VOTING CLASSIFIER")
print("=" * 80)

HardVotingModel.fit(x_train_scaled, y_train)

print("\nHard Voting Classifier trained successfully.")


# =============================================================================
# STEP 23 : MAKE HARD VOTING PREDICTIONS
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 23 : MAKE HARD VOTING PREDICTIONS")
print("=" * 80)

HardVotingPred = HardVotingModel.predict(x_test_scaled)

print("\nHard Voting predictions made successfully.")

print("\nPredicted values :")
print(HardVotingPred)

print("\nActual values :")
print(y_test.values)


# =============================================================================
# STEP 24 : EVALUATE HARD VOTING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 24 : EVALUATE HARD VOTING CLASSIFIER")
print("=" * 80)

HardVotingAccuracy = accuracy_score(y_test, HardVotingPred)

print("\nHard Voting Accuracy :", HardVotingAccuracy)
print("Hard Voting Accuracy : {:.2f}%".format(HardVotingAccuracy * 100))


# =============================================================================
# STEP 25 : CREATE SOFT VOTING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 25 : CREATE SOFT VOTING CLASSIFIER")
print("=" * 80)

SoftVotingModel = VotingClassifier(
    estimators=[
        ("Logistic Regression", LogisticModel),
        ("Decision Tree", DecisionTreeModel),
        ("KNN", KNNModel)
    ],
    voting="soft"
)

print("\nSoft Voting Classifier created successfully.")

# =============================================================================
# STEP 26 : TRAIN SOFT VOTING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 26 : TRAIN SOFT VOTING CLASSIFIER")
print("=" * 80)

SoftVotingModel.fit(x_train_scaled, y_train)

print("\nSoft Voting Classifier trained successfully.")

# =============================================================================
# STEP 27 : MAKE SOFT VOTING PREDICTIONS
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 27 : MAKE SOFT VOTING PREDICTIONS")
print("=" * 80)

SoftVotingPred = SoftVotingModel.predict(x_test_scaled)

print("\nSoft Voting predictions made successfully.")

print("\nPredicted values :")
print(SoftVotingPred)

print("\nActual values :")
print(y_test.values)


# =============================================================================
# STEP 28 : EVALUATE SOFT VOTING CLASSIFIER
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 28 : EVALUATE SOFT VOTING CLASSIFIER")
print("=" * 80)

SoftVotingAccuracy = accuracy_score(y_test, SoftVotingPred)

print("\nSoft Voting Accuracy :", SoftVotingAccuracy)
print("Soft Voting Accuracy : {:.2f}%".format(SoftVotingAccuracy * 100))

# =============================================================================
# STEP 29 : MODEL ACCURACY COMPARISON
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 29 : MODEL ACCURACY COMPARISON")
print("=" * 80)

ModelResults = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Hard Voting",
        "Soft Voting"
    ],
    "Accuracy": [
        Accuracy,
        DecisionTreeAccuracy,
        KNNAccuracy,
        HardVotingAccuracy,
        SoftVotingAccuracy
    ]
})

ModelResults["Accuracy (%)"] = ModelResults["Accuracy"] * 100

print("\nModel Accuracy Comparison :")
print(ModelResults)

# =============================================================================
# STEP 30 : VISUALIZE MODEL ACCURACY
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 30 : VISUALIZE MODEL ACCURACY")
print("=" * 80)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=ModelResults,
    x="Model",
    y="Accuracy (%)"
)

plt.title("Customer Loan Approval - Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=20)
plt.ylim(0, 110)

plt.tight_layout()

plt.show()

# =============================================================================
# STEP 31 : PREDICT NEW LOAN APPLICATION
# =============================================================================

print("\n")
print("=" * 80)
print("STEP 31 : PREDICT NEW LOAN APPLICATION")
print("=" * 80)

NewApplicant = pd.DataFrame({
    "Age": [30],
    "Income": [55000],
    "CreditScore": [710],
    "ExistingLoan": [0],
    "EmploymentExperience": [7],
    "LoanAmount": [220000]
})

NewApplicantScaled = Scaler.transform(NewApplicant)

NewPrediction = SoftVotingModel.predict(NewApplicantScaled)

if NewPrediction[0] == 1:
    print("\nLoan Approval Prediction : APPROVED")
else:
    print("\nLoan Approval Prediction : REJECTED")