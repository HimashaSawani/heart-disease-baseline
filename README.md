# Heart Disease Baseline Implementation and Evaluation

**Machine Learning Coursework — Phase 2, Option A**  
**Author:** P.V.H.S. Vitharama  
**Student ID:** KIC-BSCCSAI-251P-008

## Overview

This project implements a Logistic Regression baseline for binary heart-disease classification using the public UCI Cleveland dataset. The experiment was developed and executed entirely in Google Colab. The notebook covers data checks, preprocessing, hyperparameter tuning, held-out evaluation, uncertainty estimates and comparison plots.

The baseline is compared contextually with the LVQ results reported by Srinivasan et al. (2023). LVQ is not implemented or retrained in this project.

## Notebook and dataset

- The `.ipynb` notebook contains the complete implementation.
- `processed.cleveland.data` is the dataset file to upload when the notebook prompts for it.
- The upload cell also accepts the official UCI Heart Disease archive and extracts the Cleveland file.

**Source:** [UCI Heart Disease Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)  
**Dataset DOI:** [10.24432/C52P4X](https://doi.org/10.24432/C52P4X)

The dataset contains 303 records, 13 predictors and the diagnosis column `num`. The target is coded as `0` for `num = 0` and `1` for `num > 0`. This represents diagnosis in the recorded data, rather than future disease onset. There are 164 no-disease and 139 disease records. Missing values are marked `?`.

The notebook verifies this dataset snapshot using SHA-256:

```text
a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8
```

## View the saved results

Open the executed notebook on GitHub to view its saved tables and graphs. Viewing saved outputs does not require uploading the dataset or running the code. If GitHub cannot render the notebook, open it in Google Colab.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Open notebook → GitHub** and paste this repository's URL, or download the notebook and open it using the **Upload** tab.
3. Choose **Runtime → Run all** in a fresh runtime.
4. At the upload prompt, select `processed.cleveland.data` or the official UCI Heart Disease archive. If the dataset is included in this repository, download its raw file first.
5. Allow the remaining cells to finish. The notebook displays data checks, validation scores, test metrics, graphs and verification results.
6. Save or download the executed notebook to retain the new outputs.

The notebook installs its specified model-library versions. Dataset upload is interactive; code and saved outputs alone do not replace the dataset when rerunning the experiment. No separate Python scripts or project ZIP are required.

## Method

- Stratified 75:25 development/test split with seed 42: 227 development records and 76 test records.
- Numeric features: median imputation followed by standardization.
- Categorical features: most-frequent imputation followed by one-hot encoding (`drop="first"`, `handle_unknown="ignore"`).
- All fitted preprocessing remains inside the cross-validation pipeline.
- L2-regularized Logistic Regression with the L-BFGS solver and a 5,000-iteration limit.
- Five-fold stratified cross-validation searches `C = [0.01, 0.1, 1, 10, 100]`, using mean validation ROC-AUC.
- The selected pipeline is refitted on all development records and evaluated on the reserved test set at a fixed threshold of 0.5.

The recorded run selected **C = 0.1**, with mean validation ROC-AUC **0.9032**, fold standard deviation **0.0430**, and **18 transformed inputs**. The final fit converged in 11 iterations without convergence warnings.

## Recorded test results

| Metric | Result |
|---|---:|
| Accuracy | 84.21% |
| Precision | 82.86% |
| Recall / sensitivity | 82.86% |
| Specificity | 85.37% |
| F1 | 82.86% |
| Balanced accuracy | 84.11% |
| ROC-AUC | 0.9484 |
| Average precision | 0.9365 |

The confusion matrix contains **TN = 35, FP = 6, FN = 6 and TP = 29**. The majority-class reference achieves **53.95%** accuracy on the same test records.

The 95% Wilson accuracy interval is **74.40%–90.73%**, without continuity correction. The 95% percentile bootstrap ROC-AUC interval is **0.8944–0.9880**, based on 2,000 valid resamples. These estimates condition on the fitted model and test sample; the bootstrap does not retrain the model.

The notebook displays a confusion matrix, ROC and precision–recall curves, a baseline-versus-reported-LVQ comparison chart and a cross-validation tuning chart. It also saves JSON, CSV and PNG outputs in the Colab runtime under `phase2_colab/outputs`.

## Comparison with the selected paper

The paper reports LVQ accuracy of **98.78%**, compared with the baseline's **84.21%**: a descriptive gap of **14.57 percentage points**.

This is not a controlled comparison. The paper reports 503 records and ten attributes, whereas this implementation uses 303 Cleveland records and 13 predictors. Sample construction, preprocessing and evaluation protocols differ. The paper's reported sensitivity, recall, F1 and accuracy also contain internal inconsistencies. Its values are retained as published; the comparison does not establish that LVQ is superior on the Cleveland test set.

## Limitations

The results come from one split of a small historical dataset, with 35 positive test records and six false negatives. Cross-validation selects the hyperparameter; it does not provide external validation. Probability calibration and threshold optimization were not evaluated. This is an academic baseline, not a clinically validated screening system.

## References

1. A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, “Heart Disease,” UCI Machine Learning Repository, 1989. DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
2. S. Srinivasan et al., “An active learning machine technique based prediction of cardiovascular heart disease from UCI-repository database,” *Scientific Reports*, vol. 13, article 13588, 2023. DOI: [10.1038/s41598-023-40717-1](https://doi.org/10.1038/s41598-023-40717-1).
3. [Scikit-learn documentation](https://scikit-learn.org/stable/), including Pipeline, preprocessing, LogisticRegression and GridSearchCV.
