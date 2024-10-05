# Bank Marketing Campaign - Machine Learning Project

## Project Overview

This project focuses on predicting the outcome of a bank marketing campaign using machine learning algorithms such as Logistic Regression and Random Forest. To address the class imbalance in the dataset, we apply SMOTE (Synthetic Minority Oversampling Technique). The target variable (`y`) represents whether a customer has subscribed to a term deposit (`yes` or `no`).

## Dataset

The dataset contains 45,211 observations and 17 features, which include information about the clients and the marketing campaign. The data was collected from direct marketing campaigns conducted by a Portuguese banking institution. The goal is to predict whether the client will subscribe to a term deposit based on the given attributes.

### Dataset Summary

| **Column Name** | **Description** |
|-----------------|-----------------|
| `age`           | Age of the client (numeric). |
| `job`           | Type of job (categorical: "admin.", "technician", "entrepreneur", etc.). |
| `marital`       | Marital status (categorical: "married", "single", "divorced"). |
| `education`     | Education level (categorical: "primary", "secondary", "tertiary", or "unknown"). |
| `default`       | Does the client have a credit in default? ("yes" or "no"). |
| `balance`       | Balance of the client's account in euros (numeric). |
| `housing`       | Does the client have a housing loan? ("yes" or "no"). |
| `loan`          | Does the client have a personal loan? ("yes" or "no"). |
| `contact`       | Contact communication type (categorical: "unknown", "telephone", "cellular"). |
| `day`           | Last contact day of the month (numeric). |
| `month`         | Last contact month of the year (categorical: "jan", "feb", "mar", etc.). |
| `duration`      | Duration of the last contact in seconds (numeric). |
| `campaign`      | Number of contacts performed during this campaign (numeric). |
| `pdays`         | Number of days since the client was last contacted from a previous campaign (numeric, -1 means the client was not previously contacted). |
| `previous`      | Number of contacts performed before this campaign (numeric). |
| `poutcome`      | Outcome of the previous marketing campaign (categorical: "success", "failure", "unknown"). |
| `y`             | Target variable, has the client subscribed to a term deposit? ("yes" or "no"). |

### Class Distribution
The dataset is imbalanced, with significantly more clients not subscribing to a term deposit (`no`) than those who did (`yes`). To address this imbalance, we use SMOTE to oversample the minority class.

## Machine Learning Approach

### 1. **Data Preprocessing**
   - Handle missing or unknown data in categorical columns.
   - Encode categorical variables using one-hot encoding.
   - Scale numerical features (age, balance, duration, etc.) to normalize the data.

### 2. **Modeling**
   - **Logistic Regression**: A baseline model for classification.
   - **Random Forest**: An ensemble model to improve accuracy and handle feature interactions.
   - **SMOTE**: Applied to balance the classes by generating synthetic data points for the minority class (`yes`).

### 3. **Evaluation Metrics**
   - **Accuracy**: Measures the overall correctness of the model.
   - **Precision, Recall, and F1-score**: Useful in evaluating the performance, especially given the class imbalance.
   - **ROC-AUC**: To visualize and compare the performance of the classifiers.

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/bank-marketing-ml.git
   ```

2. Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook bank_marketing_ml.ipynb
   ```

4. The notebook walks through the entire process from data preprocessing, model building, and evaluation.

## Files in the Repository

- `bank.csv`: The dataset used for the project.
- `bank_marketing_ml.ipynb`: Jupyter Notebook containing the full implementation of the project.
- `requirements.txt`: List of Python packages required to run the project.
- `README.md`: Overview of the project (this file).

## Results

After training and evaluating the models, the Random Forest classifier performed better than Logistic Regression, especially in handling the imbalanced classes, achieving higher F1-scores and better AUC-ROC curves.

## Conclusion

This project demonstrates the application of Logistic Regression and Random Forest on an imbalanced dataset using SMOTE to improve prediction performance. It shows the importance of handling class imbalance in classification tasks and how ensemble models can improve accuracy in such scenarios.

