# Statistical and Machine Learning Labs

This repository contains a collection of experiments and code implementations, that explore fundamental concepts in both **Statistical Learning** and **Machine Learning**.  
Each folder demonstrates a key idea through practical coding exercises and data-driven analysis.

---

## 📚 Table of Contents

1. [Bias-Variance Tradeoff](#polynomial_regression_experiment)

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

### ⚙️ Requirements
This notebook uses standard Python libraries. Install dependencies using:
```bash
pip install numpy pandas scikit-learn
