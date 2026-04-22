## Project Overview

This project focuses on predicting customer churn using machine learning techniques. The goal is to analyze telecom customer data, perform preprocessing, feature engineering, build predictive models, and deploy a modular ML pipeline. The project also includes version control using Git and GitHub for tracking and collaboration.

 ## Objectives
Preprocess telecom customer dataset
Handle missing values and encode categorical variables
Perform feature engineering to improve model performance
Build and evaluate machine learning models
Compare multiple models to select the best one
Implement a modular ML pipeline
Use Git and GitHub for version control
## Dataset
telecom Customer Churn Dataset

## Key features include:
Customer demographics (gender, senior citizen, partner, dependents)
Service details (internet service, phone service, streaming services)
Account information (contract type, payment method, tenure)
Target variable: Churn (Yes/No)
## Technologies Used
Python 🐍
Pandas
NumPy
Scikit-learn
XGBoost
Git & GitHub
## Project Workflow
1. Data Loading
Load dataset using Pandas
Inspect data structure
2. Data Preprocessing
Handle missing values in TotalCharges
Convert data types
Encode target variable (Churn → 0/1)
Drop irrelevant columns (customerID)
3. Feature Engineering
Create tenure-based groups:
0–1 Year
1–2 Years
2–4 Years
4+ Years
Convert categorical variables using one-hot encoding
4. Model Building
Models used:
Logistic Regression
Decision Tree Classifier
XGBoost Classifier
5. Model Evaluation
Accuracy Score
Classification Report
Confusion Matrix
Model Comparison
6. Best Model Selection
XGBoost / Logistic Regression selected based on performance
## Results Summary
Logistic Regression: Balanced performance with good recall
Decision Tree: Lower accuracy, prone to overfitting
XGBoost: Best overall accuracy and performance
## Modular ML Pipeline Structure
The project is implemented using modular functions:

load_data()
preprocess_data()
feature_engineering()
train_model()
evaluate_model()
run_pipeline()
## Version Control (Git & GitHub)
This project includes full version control:
. Repository initialized using Git
. Feature branching strategy used
. Code tracked with meaningful commits
. Merged development branch into main
. Project pushed to GitHub repository
## How to Run the Project
# Clone repository
git clone https://github.com/your-username/telecom-churn-ml-pipeline.git
# Navigate to project folder
cd telecom-churn-ml-pipeline
# Run pipeline
python pipeline.py
## Key Learnings
End-to-end ML pipeline development
Data preprocessing & feature engineering
Model comparison techniques
Modular programming in Python
Git workflow and version control
Real-world churn prediction use case
