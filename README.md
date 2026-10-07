# Customer Loan Approval Prediction using Voting Classifier

## 📌 Project Overview

This project focuses on predicting whether a customer's loan application will be **approved or rejected** using Machine Learning classification algorithms.

Multiple classification models are trained and evaluated, and their predictions are combined using **Voting Classifiers** to improve the reliability of the final prediction.

The project compares:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Hard Voting Classifier
- Soft Voting Classifier

The final system can also predict the loan approval status of a new customer based on their financial and employment information.

---

## 🎯 Objective

The main objectives of this project are:

- To understand a customer loan approval dataset.
- To perform basic data preprocessing.
- To separate input features and target variable.
- To split the dataset into training and testing sets.
- To apply feature scaling where required.
- To build multiple classification models.
- To compare the performance of individual classifiers.
- To implement Hard Voting and Soft Voting Classifiers.
- To visualize model performance.
- To predict loan approval for a new customer.

---

## 📊 Dataset

The dataset contains **50 customer records** and **7 columns**.

### Features

| Feature | Description |
|---|---|
| Age | Age of the customer |
| Income | Customer's income |
| CreditScore | Customer's credit score |
| ExistingLoan | Indicates whether the customer already has a loan |
| EmploymentExperience | Years of employment experience |
| LoanAmount | Amount of loan requested |

### Target Variable

| Target | Meaning |
|---|---|
| `0` | Loan Rejected |
| `1` | Loan Approved |

---

## 🔄 Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Dataset Understanding
   ↓
Missing Value Check
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Machine Learning Models
   ↓
Model Predictions
   ↓
Model Evaluation
   ↓
Voting Classifiers
   ↓
Model Comparison
   ↓
New Customer Prediction