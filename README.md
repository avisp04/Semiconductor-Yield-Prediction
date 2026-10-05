# Semiconductor Manufacturing Process – Yield Prediction

### Overview

This project focuses on predicting the Pass/Fail yield of semiconductor production entities using sensor data.

The dataset contains a large number of sensor signals, but not all of them are useful for prediction. So, along with classification, the project also focuses on feature selection and dimensionality reduction.

### Dataset

`signal-data.csv`

* 1567 rows
* 592 columns
* Target: `Pass/Fail`
* `-1` = Pass
* `1` = Fail

The dataset is highly imbalanced, with much fewer Fail samples.

**Note:** When using Google Colab, the dataset needs to be manually uploaded when the notebook asks for it.

### Steps Performed

* Data exploration and cleaning
* Missing value and duplicate analysis
* Removal of constant and highly incomplete features
* Timestamp feature extraction
* Univariate, bivariate and multivariate analysis
* Data visualisation
* Train-test split
* Class imbalance handling using SMOTE
* Feature selection using SelectKBest
* Dimensionality reduction using PCA
* Model training and comparison
* Cross-validation and GridSearchCV
* Model evaluation using multiple metrics
* Feature importance analysis
* Final model selection and saving

### Models Used

* Logistic Regression
* SVM
* Random Forest
* Gaussian Naive Bayes
* PCA + SVM

The models are compared using Accuracy, F1-score, Recall, Balanced Accuracy, ROC-AUC and PR-AUC.

### Feature Analysis

SelectKBest, PCA, Random Forest importance and permutation importance are used to check which sensor signals are useful.

Different numbers of features are also tested to check whether all sensor signals are actually required.

### Files

```text
signal-data.csv
main.ipynb
README.md
```

After running the notebook:

```text
semiconductor_yield_model.pkl
semiconductor_preprocessing_info.pkl
```
The .pkl files contain the trained model and preprocessing information, which can be loaded later for prediction without retraining the model.

### How to Run

Open the notebook in Google Colab, run the first cell and upload `signal-data.csv` when prompted. Then run the remaining cells in order.

### Future Scope

* Try XGBoost or CatBoost
* Use time-based validation
* Try other feature selection methods
* Improve failure detection with threshold tuning
* Use more failure samples
* Deploy the model for real-time yield monitoring

### Conclusion

The project shows that machine learning can be used to predict semiconductor yield from sensor data. Feature selection and dimensionality reduction can also help reduce the number of signals without necessarily affecting model performance significantly.
