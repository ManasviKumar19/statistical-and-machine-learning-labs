# Statistical and Machine Learning Labs

This repository contains a collection of experiments and code implementations, that explore fundamental concepts in both **Statistical Learning** and **Machine Learning**.  
Each folder demonstrates a key idea through practical coding exercises and data-driven analysis.

---

## 📚 Table of Contents

1. [Bias-Variance Tradeoff](#polynomial_regression_experiment)
2. [LDA vs QDA Comparison](#lda_qda_comparison)
3. [LDA vs Logistic Regression](#lda_logistic_comparison)
4. [Breast Cancer Classification](#knn_rf_svm_comparison)

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