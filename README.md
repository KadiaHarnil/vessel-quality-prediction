# Steel Plate Surface Defect Classification

A multiclass machine learning portfolio project that classifies **seven steel plate surface fault types** from **27 numerical inspection features** using a tuned Random Forest classifier.

> **Important scope note:** This is a steel-plate **defect category** classifier, not a vessel PASS/FAIL, defect-free detection, or safety approval system. This repository name reflects an earlier project idea.

## Project summary

| Item | Details |
| --- | --- |
| Dataset | [Faulty Steel Plates on Kaggle](https://www.kaggle.com/datasets/uciml/faulty-steel-plates) ([UCI source](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)) |
| Samples | 1,941 |
| Input features | 27 numerical features |
| Target classes | Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults |
| Modeling approach | Random Forest + RandomizedSearchCV |
| Tuning metric | Macro F1 (to give rare fault classes equal weight) |

## Project workflow

1. Inspect the dataset and validate that each row has exactly one active defect indicator.
2. Convert the seven binary defect indicators to one categorical `Target` column.
3. Exclude those original target indicators from the feature matrix to prevent leakage.
4. Create a stratified 80/20 train/test split.
5. Tune a Random Forest with cross-validation using macro F1.
6. Evaluate with accuracy, per-class precision/recall/F1, and a confusion matrix.
7. Interpret built-in feature importance and validation-based permutation importance.
8. Save and reload the trained estimator with `joblib`.

## Results

In an **earlier exploratory test run**, the tuned Random Forest obtained approximately **79.95% test accuracy** and **0.79 test macro F1**. The formatted notebook includes a reproducible search setup and should be rerun to obtain its own verified scores; results may differ from the earlier run. These figures are not claims of production readiness.

## Run the notebook

**Kaggle (recommended):** Attach the [Faulty Steel Plates dataset](https://www.kaggle.com/datasets/uciml/faulty-steel-plates), import [`steel_plate_defect_classification.ipynb`](steel_plate_defect_classification.ipynb), then **Run All**. The notebook searches `/kaggle/input` for `faults.csv`.

**Local / Colab:** Install dependencies using `pip install -r requirements.txt` (locally), obtain `faults.csv` from the dataset source, place it in your current working directory (or `/content/` on Colab), then run all cells in order.

The dataset and generated model binaries are not committed to this repository. `steel_defect_model.pkl` is produced when the final notebook section runs.

## Limitations and future improvements

- Every example already belongs to a fault class, so the classifier does not detect fault-free plates.
- Some classes have few observations; metrics for those classes have higher uncertainty.
- Feature importance is descriptive of the fitted model, not proof of causation.
- Further validation on independent manufacturing data is required before real-world use.

**Possible next steps:** Try a second model family, examine class-specific mistakes, package a small prediction interface, and document model monitoring.

## Author and attribution

Learning and implementation project. Original data: [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults), mirrored on [Kaggle](https://www.kaggle.com/datasets/uciml/faulty-steel-plates).
