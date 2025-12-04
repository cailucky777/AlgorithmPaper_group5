# LightGBM for Imbalanced Auto Insurance Claim Prediction
**A Comparative Study on the Porto Seguro Dataset**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Porto%20Seguro-20BEFF.svg)](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)

*Course Project — Machine Learning / Applied Computing*  
*BCIT School of Computing and Academic Studies*

---

## Table of Contents
- [Overview](#overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [References](#references)

---

## Overview

This project addresses the challenge of predicting auto insurance claims using the **Porto Seguro Safe Driver Prediction** dataset from Kaggle. We evaluate the performance of **LightGBM** (Light Gradient Boosting Machine) against classical machine learning baselines on a highly imbalanced, sparse, and high-dimensional dataset.

### Research Questions
- How does LightGBM compare to classical models in predictive performance?
- What is the computational efficiency trade-off between accuracy and training time?
- How do modern gradient boosting techniques handle class imbalance and sparsity?

### Key Challenges
- **High dimensionality**: 59 features with mixed types (binary, categorical, continuous)
- **Severe class imbalance**: Only ~3-4% positive class (claims filed)
- **Sparsity**: Missing values encoded as `-1`
- **Real-world complexity**: Industrial-scale dataset with ~595,000 samples

---

## Key Findings

| Model | ROC AUC | Training Time | Kaggle Public | Kaggle Private |
|-------|---------|---------------|---------------|----------------|
| **LightGBM** | **0.6329** | **88.89s** | **0.26243** | **0.25679** |
| Random Forest | 0.6285 | 133.52s | 0.25690 | 0.25219 |
| Logistic Regression | 0.6202 | 93.23s | 0.24312 | 0.23871 |
| Decision Tree | 0.6010 | 5.23s | 0.20966 | 0.20429 |

**LightGBM achieved the highest AUC while maintaining competitive training time, demonstrating superior balance between accuracy and efficiency.**

---

## Repository Structure
```
project/
│
├── paper/
│   └── alg-group-02.pdf               # Final compiled paper
├── notebooks/
│   ├── AAGroup5Project.ipynb          # Main experimental notebook
│   ├── AAGroup5Project_Jacky.ipynb    # Individual experiments
│   └── AAGroup5Project_Nicky.ipynb    # Individual experiments
│
├── data/                               # (Not included - download from Kaggle)
│   ├── train.csv
│   └── test.csv
│
└── README.md                           # This file
```

---

## Dataset

**Porto Seguro Safe Driver Prediction** ([Kaggle Competition](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction))

### Characteristics
- **Samples**: ~595,000 training instances
- **Features**: 59 (+ 1 binary target)
  - 17 binary features (>95% zeros)
  - 14 categorical features
  - 26 continuous/ordinal features
- **Target**: `target = 1` if claim filed, `0` otherwise
- **Missing Values**: Encoded as `-1` (not `NaN`)
- **Class Distribution**: Highly imbalanced (~3-4% positive class)

### Feature Types
Features are tagged with postfixes indicating their type:
- `_bin`: Binary features
- `_cat`: Categorical features
- Others: Continuous or ordinal features

---

## Methods

### Baseline Models

#### 1. **Logistic Regression**
- Linear model assuming additive relationships
- Fast, interpretable, but limited expressiveness
- Poor performance on nonlinear patterns

#### 2. **Decision Tree**
- Single tree with recursive splitting
- Captures nonlinearity but prone to overfitting
- High variance and instability

#### 3. **Random Forest**
- Ensemble of decision trees (bagging)
- Reduces variance through averaging
- Cannot correct sequential errors

### Modern Algorithm: **LightGBM**

LightGBM combines several innovations for efficient gradient boosting:

#### Core Components

1. **Gradient Boosting Decision Trees (GBDT)**
```
   F_m(x) = F_{m-1}(x) + γ_m h_m(x)
```
   - Sequential error correction
   - Additive model building

2. **Histogram-based Splitting**
   - Bins continuous features into discrete bins
   - Reduces complexity from O(#data × #features) to O(#data × #bins)
   - Better handling of sparse features

3. **Gradient-based One-Side Sampling (GOSS)**
   - Keeps instances with large gradients
   - Randomly samples instances with small gradients
   - Maintains accuracy while reducing data size

4. **Exclusive Feature Bundling (EFB)**
   - Bundles mutually exclusive features
   - Reduces dimensionality from O(#features) to O(#bundles)
   - Constructs conflict graph for intelligent bundling

5. **Leaf-wise Growth**
   - Grows trees by splitting leaf with maximum delta loss
   - More accurate than level-wise growth
   - Controlled by max_depth to prevent overfitting

---

## Results

### Performance Metrics

**ROC AUC Comparison**
```
LightGBM:          0.6329  ████████████████████ (Best)
Random Forest:     0.6285  ███████████████████
Logistic Reg:      0.6202  ██████████████████
Decision Tree:     0.6010  █████████████████
```

**Training Time Comparison**
```
Decision Tree:     5.23s    █
LightGBM:         88.89s    █████████████████
Logistic Reg:     93.23s    ██████████████████
Random Forest:   133.52s    ███████████████████████████ (Slowest)
```

### Key Insights

1. **LightGBM Advantages**:
   - Highest predictive accuracy (ROC AUC: 0.6329)
   - Faster than Random Forest (88.89s vs 133.52s)
   - Handles sparsity and missing values efficiently
   - Captures complex nonlinear interactions

2. **Random Forest**:
   - Second-best accuracy but computationally expensive
   - More stable than single Decision Tree
   - Limited sequential error correction

3. **Logistic Regression**:
   - Competitive baseline for linear relationships
   - Fast training but limited expressiveness
   - Requires extensive feature engineering for nonlinear patterns

4. **Decision Tree**:
   - Fastest training but poorest performance
   - Highly interpretable but unstable
   - Severe overfitting on imbalanced data

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/lightgbm-insurance-prediction.git
cd lightgbm-insurance-prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


---

## Usage

### 1. Download Dataset

Download the Porto Seguro dataset from [Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data) and place files in `data/` directory:
```
data/
├── train.csv
└── test.csv
```

### 2. Run Experiments

#### Jupyter Notebook
```bash
jupyter notebook notebooks/AAGroup5Project.ipynb
```

### 3. Reproduce Paper Results

All experiments from the paper can be reproduced using the provided notebooks:
- `AAGroup5Project.ipynb`: Complete experimental pipeline
- Individual notebooks: Specific model implementations and ablation studies

---

## Authors

| Name | Email | Affiliation |
|------|-------|-------------|
| **Jacky Chen** | jchen574@my.bcit.ca | BCIT School of Computing |
| **Nicky Cheng** | nicky_cheng@outlook.com | BCIT School of Computing |
| **Luying Cai** | lcai25@my.bcit.ca | BCIT School of Computing |
| **Bryan Rachmat** | Rachmat.bryan@gmail.com | BCIT School of Computing |

*British Columbia Institute of Technology (BCIT)*  
*Vancouver/Burnaby, BC, Canada*

---

## References

### Core Papers

1. **Ke, G., et al. (2017)**  
   *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*  
   Advances in Neural Information Processing Systems (NIPS 2017)  
   [Paper Link](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

2. **Friedman, J. H. (2001)**  
   *Greedy Function Approximation: A Gradient Boosting Machine*  
   Annals of Statistics, 29(5):1189–1232  
   [DOI: 10.1214/aos/1013203451]

3. **Chen, T. & Guestrin, C. (2016)**  
   *XGBoost: A Scalable Tree Boosting System*  
   KDD '16: Proceedings of the 22nd ACM SIGKDD  
   [DOI: 10.1145/2939672.2939785]

4. **Breiman, L. (2001)**  
   *Random Forests*  
   Machine Learning, 45(1):5–32  
   [DOI: 10.1023/A:1010933404324]

### Dataset

**Porto Seguro Safe Driver Prediction** (2017)  
Kaggle Competition  
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

---

## Future Work

1. **Hyperparameter Optimization**
   - Grid search, random search, Bayesian optimization
   - Automated tuning frameworks (Optuna, Hyperopt)

2. **Advanced Feature Engineering**
   - Interaction terms and polynomial features
   - Embedding-based encodings for categorical variables
   - Domain-specific feature construction

3. **Extended Model Comparison**
   - CatBoost, XGBoost comparisons
   - Neural network baselines
   - Ensemble stacking approaches

4. **Evaluation Enhancement**
   - Precision-Recall AUC
   - Calibration curves
   - Cost-sensitive metrics
   - Fairness and bias analysis

5. **Generalization Studies**
   - Cross-domain evaluation (credit risk, healthcare)
   - Transfer learning experiments
   - Robustness to distribution shift

6. **Interpretability**
   - SHAP value analysis
   - LIME explanations
   - Feature importance visualization

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Porto Seguro** for providing the dataset via Kaggle
- **Microsoft Research** for developing LightGBM
- **BCIT School of Computing** for project support
- **Kaggle Community** for discussions and insights

---

## Contact

For questions or collaboration opportunities:
- **GitHub Issues**: [Open an issue](https://github.com/yourusername/lightgbm-insurance-prediction/issues)
- **Email**: lcai25@my.bcit.ca

