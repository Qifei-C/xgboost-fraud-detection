# XGBoost Vignette

A vignette on implementing XGBoost using credit card fraud data, created as a class project for PSTAT197A in Fall 2023.

**Contributors**: Mu Niu, Yixin Xue, Amber Wang, David Lin, Qifei Cui

## Abstract

This vignette provides a comprehensive exploration of XGBoost, covering its mathematical principles, programming language support, and practical application in classification problems. It utilizes a simulated "Credit Card Fraud" dataset obtained from Kaggle (https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/data). This dataset presents a binary classification problem, classifying transactions as either fraudulent or legitimate, with 8 variables and a total of 1,000,000 observations. Our model achieved an AUC score of 0.99988, indicating that the model is almost a perfect classifier. This unusually high result is likely due to the simulated nature of the dataset, which is designed for practice purposes.

## Repository Contents

```
XGBoostVignette/
├── data/                    # Contains raw and processed datasets
│   ├── raw/
│   └── processed/
├── scripts/                 # R scripts for the vignette
│   ├── drafts/              # Draft scripts
│   └── vignette-script.R    # Main R script for the vignette
├── img/                     # Images used in the vignette
├── vignette.qmd             # Quarto Markdown source for the vignette
├── vignette.html            # Rendered HTML output of the vignette
├── README.md                # This file
└── vignette-xgboost.Rproj   # RStudio project file
```

## Reference List

- Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016.

- He, Shen, et al. “An Effective Cost-Sensitive XGBoost Method for Malicious Urls Detection in Imbalanced Dataset.” IEEE Access, vol. 9, 2021, pp. 93089–93096, doi:10.1109/access.2021.3093094. 

- Narayanan R, Dhanush. “Credit Card Fraud”, https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/data. Accessed 06 Dec. 2023. 

- “XGBoost Documentation.” XGBoost Documentation - Xgboost 2.0.2 Documentation, xgboost.readthedocs.io/.