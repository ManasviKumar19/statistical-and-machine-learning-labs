# Statistical and Machine Learning Labs

This repository contains a collection of experiments and code implementations, that explore fundamental concepts in both **Statistical Learning** and **Machine Learning**.  
Each folder demonstrates a key idea through practical coding exercises and data-driven analysis.

---

## 📚 Table of Contents

1. [Bias-Variance Tradeoff](#polynomial_regression_experiment)
2. [LDA vs QDA Comparison](#lda_qda_comparison)
3. [LDA vs Logistic Regression](#lda_logistic_comparison)
4. [Breast Cancer Classification](#knn_rf_svm_comparison)
5. [Regularized Regression Masq](#lasso_ridge_elasticnet_comparison)
6. [Factor Clustering Methods](#clustering_analysis)
7. [Partial Least Squares regression](#psychometric_pls_analysis)
8. [Generalized Additive Models](#bike_sharing_gam_analysis)
9. [Quantitative Structure-Activity Relationship](#qsar_ensemble_models)

---

## Folder: `polynomial_regression_experiment`

### 📁 Description
This experiment demonstrates the **bias-variance tradeoff** in polynomial regression using synthetic data generated from a sine function.  
It compares models of different complexities (degrees 3 and 15) and evaluates their Mean Squared Error (MSE) on small and large datasets.

### ⚙️ File(s)
- `Source_Code.ipynb` — generates data, trains polynomial regression models, and computes MSE for various configurations.

### ▶️ How to Run
1. Open the notebook in Jupyter or VS Code.
2. Run all cells sequentially.
3. Observe printed MSE results for different model degrees and dataset sizes.

### 🧠 Key Concepts / Techniques
- Polynomial Regression  
- Bias-Variance Tradeoff  
- Mean Squared Error (MSE)  
- Synthetic Data Generation  
- Model Evaluation

---

## Folder: `lda_qda_comparison`

### 📁 Description
This experiment compares **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** on synthetically generated multivariate Gaussian data.  
It investigates how training sample size affects model accuracy and generalization performance.

### ⚙️ File(s)
- `Source_Code.ipynb` — generates synthetic data, trains LDA and QDA classifiers, and reports average accuracies across multiple runs.

### ▶️ How to Run
1. Open the notebook in Jupyter or VS Code.
2. Run all cells sequentially.
3. View printed results showing average LDA and QDA accuracies for small (50) and large (10,000) training sets.

### 🧠 Key Concepts / Techniques
- Linear Discriminant Analysis (LDA)  
- Quadratic Discriminant Analysis (QDA)  
- Classification with Multivariate Gaussian Data  
- Model Complexity and Generalization  
- Accuracy Evaluation


---

## Folder: `lda_logistic_comparison`

### 📁 Description
This experiment compares **Linear Discriminant Analysis (LDA)** and **Logistic Regression** on synthetically generated multivariate Gaussian data.  
It explores how **training sample size** impacts classification accuracy and model generalization. By repeating each experiment multiple times, it estimates the **average balanced accuracy** for both classifiers under small and large data conditions.

### ⚙️ File(s)
- `Source_Code.ipynb` — generates synthetic Gaussian data, trains LDA and Logistic Regression models, and reports their average accuracies across 100 runs for different training sizes.

### ▶️ How to Run
1. Open the notebook in Jupyter or VS Code.  
2. Run all cells sequentially.  
3. Observe printed outputs showing the average balanced accuracies for:
   - LDA (train = 50, train = 10,000)  
   - Logistic Regression (train = 50, train = 10,000)

### 🧠 Key Concepts / Techniques
- Linear Discriminant Analysis (LDA)  
- Logistic Regression  
- Multivariate Gaussian Data Generation  
- Classification and Model Generalization  
- Balanced Accuracy Evaluation  
- Effect of Sample Size on Model Performance  

### ⚙️ Requirements
This notebook uses standard Python libraries. Install dependencies using:
```bash pip install numpy pandas scikit-learn```

## Folder: `knn_rf_svm_comparison`

### 📁 Description
This experiment applies and compares three classification algorithms — **K-Nearest Neighbors (KNN)**, **Random Forest**, and **Support Vector Machine (SVM)** — on a **breast cancer diagnosis dataset**.  
The goal is to evaluate model performance using **cross-validation**, and assess accuracy, sensitivity, and specificity on test data.

### ⚙️ File(s)
- `Source_Code.ipynb` — loads the dataset, preprocesses features, trains and evaluates three classifiers, and reports key performance metrics.

### ▶️ How to Run
1. Place the dataset file (`data.csv`) in the specified directory or update the path in the notebook.  
2. Open the notebook in Jupyter or VS Code.  
3. Run all cells sequentially.  
4. Observe printed model performances, mean cross-validation accuracies, and confusion matrix–based metrics (sensitivity and specificity).

### 🧠 Key Concepts / Techniques
- Supervised Classification  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Cross-Validation Accuracy  
- Sensitivity and Specificity  
- Model Comparison  

---

## Folder: `lasso_ridge_elasticnet_comparison`

### 📁 Description
This experiment applies three regularized regression models — **Lasso**, **Ridge**, and **Elastic Net** — to predict depression diagnosis (`D_DEPDYS`) based on psychological questionnaire data (MASQ features).  
It focuses on **feature selection**, **multicollinearity handling**, and **model comparison** using cross-validation and test accuracy.

### ⚙️ File(s)
- `Source_Code.ipynb` — loads and cleans MASQ training and test datasets, performs feature scaling, fits regularized regression models with cross-validation, and evaluates predictive performance.

### ▶️ How to Run
1. Ensure that the training and test data files (`masq_train.feather` and `masq_test.feather`) are placed in the specified paths or update their locations in the notebook.  
2. Open the notebook in Jupyter or VS Code.  
3. Run all cells sequentially.  
4. Review printed outputs for:
   - Cross-validated best hyperparameters  
   - Test set accuracy for each model  
   - Feature coefficients for Lasso, Ridge, and Elastic Net

### 🧠 Key Concepts / Techniques
- Lasso Regression  
- Ridge Regression  
- Elastic Net Regularization  
- Cross-Validation (Grid Search)  
- Feature Scaling (Standardization)  
- Handling Multicollinearity  
- Model Comparison  

---

## Folder: `clustering_analysis`
### 📁 Description

This experiment explores Principal Component Analysis (PCA), Varimax rotation, and multiple clustering techniques on an alexithymia-related dataset to identify underlying psychological dimensions and cluster patterns.
The analysis demonstrates how dimensionality reduction and unsupervised learning can reveal structure in multivariate data.

### ⚙️ File(s)

`Source_Code.ipynb` — performs PCA with Varimax rotation, applies K-Means and Hierarchical Clustering, and visualizes results through scree plots, component matrices, and dendrograms.

### ▶️ How to Run

Ensure the dataset data_alexithymia_csv.csv is available in the same directory.

Open the notebook in Jupyter or VS Code.

Run all cells sequentially to generate visualizations and clustering outputs.

### 🧠 Key Concepts / Techniques

Principal Component Analysis (PCA)

- Varimax Rotation
- K-Means Clustering
- Hierarchical Clustering
- Dimensionality Reduction
- Scree Plot Analysis
- Dendrogram Visualization

---

## Folder: `psychometric_pls_analysis`

### 📁 Description
This experiment applies **Partial Least Squares (PLS) Regression** to an alexithymia dataset to explore relationships between psychological variables and outcome measures.  
It demonstrates how dimensionality reduction and regression can be combined to handle multicollinearity in high-dimensional data.

### ⚙️ File(s)
- `Source_Code.ipynb` — performs data loading, PLS regression model training and evaluation, and visualizes performance metrics such as Mean Squared Error (MSE).

### ▶️ How to Run
1. Ensure the dataset file `data_alexithymia_csv.csv` is available in the same directory.  
2. Open the notebook in Jupyter or VS Code.  
3. Run all cells to reproduce model training and output visualizations.

### 🧠 Key Concepts / Techniques
- Partial Least Squares (PLS) Regression  
- Dimensionality Reduction  
- Train-Test Split  
- Model Evaluation using Mean Squared Error (MSE)  
- Multicollinearity Handling  
- Component-based Modeling  

### ⚙️ Requirements
This notebook uses standard Python libraries. Install dependencies using:
pip install pandas scikit-learn matplotlib factor_analyzer scipy

---

## Folder: `bike_sharing_gam_analysis`

### 📁 Description
This experiment fits a Generalized Additive Model (GAM) to a bike demand dataset to analyze how temperature, precipitation, and seasonal trends influence bike rental counts.  
It demonstrates the flexibility of GAMs in modeling nonlinear relationships between predictors and the target variable.

### ⚙️ File(s)
- `Source_Code.ipynb` — fits a GAM using the mgcv package in R, visualizes smooth terms, and evaluates model performance using Mean Squared Error (MSE).

### ▶️ How to Run
1. Ensure the dataset file `bike_dat.Rdata` is available in the same directory.  
2. Open the notebook in Jupyter with an R kernel (or RStudio).  
3. Run all cells to train the GAM model, generate plots, and compute MSE.

### 🧠 Key Concepts / Techniques
- Generalized Additive Models (GAM)  
- Spline Smoothing  
- Model Evaluation using Mean Squared Error (MSE)  
- Temperature and Weather Effects on Demand  
- Seasonal Trend Modeling  

### ⚙️ Requirements
This notebook uses R

---
## Folder: `qsar_ensemble_models`

### 📁 Description
This notebook focuses on **QSAR (Quantitative Structure-Activity Relationship)** modeling using ensemble learning methods — Gradient Boosting Machines (GBM), Random Forests, Bagging, and Decision Trees — to classify compounds based on molecular descriptors.  
The workflow demonstrates model training, hyperparameter tuning, and evaluation using Brier Scores and Misclassification Rates.

### ⚙️ File(s)
- `Source_Code.py` — implements QSAR modeling with various ensemble methods, compares performance metrics, and discusses parameter effects (shrinkage, n.trees, interaction.depth).

### ▶️ How to Run
1. Ensure the dataset file `qsar.Rda` is in the same directory.  
2. Open the notebook in **Jupyter with R kernel** (or RStudio).  
3. Run all cells sequentially to:
   - Load the QSAR dataset.  
   - Train GBM and other ensemble models.  
   - Compare models via performance metrics (Brier Score, Misclassification Rate).  

### 🧠 Key Concepts / Techniques
- Ensemble Learning (Bagging, Random Forests, Boosting)  
- Gradient Boosting Machine (GBM) Tuning  
- Hyperparameter Effects:  
  - **Shrinkage:** controls learning rate — smaller values improve generalization but need more trees.  
  - **n.trees:** number of boosting iterations — more trees can enhance performance but risk overfitting.  
  - **interaction.depth:** captures variable interactions — higher depth models complex patterns.  
- Brier Score & Misclassification Rate  
- Model Comparison and Evaluation  

### ⚙️ Requirements
Install the following R packages before running:
install.packages(c("caret", "randomForest", "gbm", "mgcv"))

### 💡 Insights
- Lower **shrinkage** generally improves accuracy but increases computation.  
- Higher **interaction.depth** suggests presence of nonlinear interactions in the QSAR dataset.  
- **GBM** models can outperform single tree or bagged models when properly tuned.  
