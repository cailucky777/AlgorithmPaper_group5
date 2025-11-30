# LightGBM vs. Classical Models: Porto Seguro Claim Prediction  
Course Project — Machine Learning / Applied Computing  

This project evaluates the performance of LightGBM on the Porto Seguro Safe Driver Prediction dataset and compares it with several classical machine-learning baselines, including Logistic Regression, Decision Tree, and Random Forest.  
The work is implemented following ACM publication standards, including a full LaTeX paper, related-work survey, experimental evaluation, and reproducible code.

---

## Project Overview

Real-world insurance datasets often contain high dimensionality, sparsity, missing values, and strong class imbalance.  
The goal of this project is to compare modern gradient-boosting models with classical models and analyze:

- Predictive performance (ROC AUC)
- Computational efficiency (training time)
- Strengths and limitations of different model classes
- Modern LightGBM developments (interpretability & hybrid models)

---

## Repository Structure

project/
│
├── paper/
│ ├── main.tex # ACM LaTeX paper
│ ├── references.bib # Bibliography file (ACM format)
│ └── figures/ # Images used in the paper
│
├── notebooks/
│ └── AAGroup5Project.ipynb # Colab notebook with full experiments
│ └── AAGroup5Project_Jacky.ipynb # Colab notebook with full experiments
│ └── AAGroup5Project_Nicky.ipynb # Colab notebook with full experiments
│
└── README.md # Project documentation


---

## Methods & Models

### Baseline Models
- **Logistic Regression** — linear, interpretable, poor with nonlinear features  
- **Decision Tree** — nonlinear but unstable and prone to overfitting  
- **Random Forest** — bagging-based ensemble, robust but less expressive than boosting  

### Modern Model
- **LightGBM**  
- Histogram-based decision trees  
- Gradient-based One-Side Sampling (GOSS)  
- Exclusive Feature Bundling (EFB)  
- Leaf-wise growth with depth constraints  

---

## Dataset

**Porto Seguro Safe Driver Prediction** (Kaggle, 2017)

- ~595,000 training rows  
- 59 features  
- High sparsity, missing values represented as `-1`  
- Imbalanced target (~3–4% positive class)  

---

## Key Results

| Model              | ROC AUC | Training Time |
|-------------------|---------|---------------|
| Logistic Regression | Low     | Very Fast     |
| Decision Tree       | ~0.51   | Fast          |
| Random Forest       | Moderate (~0.60) | Slow |
| **LightGBM**        | **0.6356** | **41.5s** |

LightGBM achieved the **best AUC** while requiring **much less computation** than Random Forest.

---

## 🧩 Recent Developments (Short Summary)
Recent work highlights:

- Use of **SHAP/LIME** for model interpretability  
- Monotonicity constraints for regulated industries  
- Hybrid systems combining **LSTM + LightGBM + KNN**, as seen in CN113344254A (2021)

---

## 📦 Requirements

- Python 3.10+
- lightgbm
- scikit-learn
- pandas
- numpy
- matplotlib / seaborn (optional)

Install all dependencies:

```bash
pip install -r requirements.txt

## How to Run
jupyter notebook notebooks/AAGroup5Project.ipynb

## Citation
If you use this project, please cite the main algorithms:
Ke et al. 2017 (LightGBM)
Friedman 2001 (GBM)
Breiman 2001 (Random Forest)
James et al. 2023 (Logistic Regression)
The ACM LaTeX template is used for the final paper.

