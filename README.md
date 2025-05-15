# APS Failure Prediction

This project focuses on predicting APS (Air Pressure System) failures in Scania trucks using a dataset provided by Scania. The primary goal is to build a robust classification model that can accurately identify potential failures, thereby enabling proactive maintenance and reducing downtime.

## Dataset

The project utilizes two CSV files:
* `aps_failure_training_set.csv`: Used for training the prediction model.
* `aps_failure_test_set.csv`: Used for evaluating the performance of the trained model.

The dataset contains sensor readings and other operational data from Scania trucks. The target variable is binary, indicating whether an APS failure occurred ('neg' for negative, 'pos' for positive).

## Libraries and Dependencies

The project uses the following Python libraries:
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn (including `IterativeImputer`, `StandardScaler`, `PCA`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, and various metrics)
* imblearn (for `SMOTE`)

## Data Preprocessing

The data preprocessing involved several key steps:
1.  **Loading Data**: The training and test datasets were loaded using pandas.
2.  **Target Variable Encoding**: The 'class' column (target variable) was converted from categorical ('neg', 'pos') to numerical (0, 1).
3.  **Handling Missing Values**:
    * 'na' string values were replaced with `None` (NaN).
    * Features with more than 50% missing values were dropped.
    * For the remaining missing values, `IterativeImputer` (MICE technique) was used, assuming the data is Missing at Random (MAR).
4.  **Feature Scaling**: `StandardScaler` was applied to scale the features after imputation.
5.  **Dimensionality Reduction**: Principal Component Analysis (PCA) was used to reduce the number of features while retaining 90% of the explained variance.
6.  **Handling Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to address the class imbalance issue, as the positive class (failures) was a minority.

## Modeling

Several classification models were explored:
1.  **Logistic Regression**: A baseline logistic regression model was trained on the original imbalanced data and then on the SMOTE-resampled data.
2.  **Decision Tree Classifier**: A decision tree model was trained on the SMOTE-resampled data.
3.  **Random Forest Classifier**: A random forest model was trained on the SMOTE-resampled data. `GridSearchCV` was used to find the best hyperparameters (`max_depth` and `n_estimators`) for the Random Forest model.

## Results and Key Findings

The models were evaluated based on accuracy, recall, precision, and F1-score. The confusion matrix was also examined to understand the model's performance in terms of true positives, false positives, true negatives, and false negatives.

* **Logistic Regression (without SMOTE)**: Achieved an accuracy of approximately 98.75%.
* **Logistic Regression (with SMOTE)**: Achieved an accuracy of approximately 97.08%.
* **Decision Tree Classifier (with SMOTE)**: Achieved an accuracy of approximately 97.91%.
* **Random Forest Classifier (with SMOTE, default parameters)**: Achieved an accuracy of approximately 98.78%.
* **Random Forest Classifier (with SMOTE, tuned parameters - max\_depth: 16, n\_estimators: 256)**: Achieved an accuracy of approximately 98.59%.

The confusion matrix for the tuned Random Forest model showed:
* True Negatives: 15597
* False Positives: 28
* False Negatives: 136
* True Positives: 239

While accuracy is high across models, the class imbalance means that metrics like recall for the minority class (failures) and the F1-score are particularly important for evaluating practical utility. The use of SMOTE and hyperparameter tuning aimed to improve the model's ability to correctly identify failures.

## Instructions to Run

1.  Ensure all the libraries listed under "Libraries and Dependencies" are installed.
2.  Place the `aps_failure_training_set.csv` and `aps_failure_test_set.csv` files in the specified path within the notebook (currently `C:\Users\prady\Desktop\Projects\archive (1)\`) or update the path in the notebook accordingly.
3.  Run the Jupyter Notebook cells sequentially to execute the data preprocessing, model training, and evaluation steps.
