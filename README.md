# Nonlinear Classification with SVM and Radial Features

**Author:** Cameron Batts

## Business Problem
Compare an RBF-kernel SVM with logistic regression before and after engineering radial distance.

## Approach
Generate 4,000 synthetic observations with two balanced radial classes. Tune 20 gamma values using a StandardScaler/SVC pipeline and five-fold cross-validation with C=1. Compare logistic regression using x1/x2 with the same model after adding x3=sqrt(x1²+x2²).

## Results
Original in-sample accuracy was 58.925% for logistic regression using x1/x2. The tuned SVM and radial-feature logistic model each reached 100% accuracy, precision and recall on the fitting data. These scores illustrate representational fit; they are not independent test performance.

## Validation and Limitations
All displayed classification scores use the same observations used for model fitting. Grid-search cross-validation selects gamma, but the displayed 100% scores are resubstitution scores. Add a held-out test set and validate robustness under overlap/noise before drawing generalization conclusions.

Original numerical outputs are retained. The revised workflow was syntax-checked but not fully rerun. Dependencies are unpinned.

## Data
Synthetic data is generated in the notebook. The plotting helper is the existing Utils.DA.py file at the repository root.

## Run
```bash
python -m pip install -r requirements.txt
jupyter lab
```
Open the notebook linked below and run cells in order.

## Files
- [Analysis notebook](Notebooks/Support%20Vector%20Machine%20Classification%20%26%20Manual%20Feature%20Engineering.ipynb): code and original outputs.
- [Readable analysis](Reports/svm_feature_engineering.html): narrative report.
- [Existing PDF](Reports/Support%20Vector%20Machine%20Classification%20%26%20Manual%20Feature%20Engineering.pdf): original exported report; it may predate these interpretation corrections.
- [Dependencies](requirements.txt).

[Live portfolio report](https://cameron-batts.github.io/svm_feature_engineering.html) · [Portfolio](https://cameron-batts.github.io)
