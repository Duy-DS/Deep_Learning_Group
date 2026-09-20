# Progress

## 2026-09-19

- Tao `README.md` theo quy uoc Kaggle/ML.
- Khoi tao `memory-bank/` de luu boi canh va tien do Lab02.
- Hoan thanh T01: tom tat Ames Housing paper va rut ra insight cho EDA, preprocessing, modeling va evaluation.
- Bat dau T03/T04 theo phuong an notebook + exported artifacts + findings handoff.
- Hoan thanh ban EDA da review: notebook chay 15 code cells, 9 figures, 7 metric tables va findings handoff; T03/T04 dang `Review`.
- Sua missing semantics theo row-level rules; giu cac anomaly basement, garage va pool la unknown missing.
- Bo sung `category_level_audit.csv`: 331 levels, 54 rare-train levels, 19 train-only va 15 test-only.
- Bo sung `missing_target_effect.csv`: 21 indicators de danh gia quan he giua missing/absence va log target.
- Cap nhat `eda-findings.md` lam handoff cho preprocessing.

## Chua thuc hien

- Nhom phe duyet EDA de chuyen T03/T04 sang `Done`.
- Xay dung preprocessing pipeline va leakage checks.
- Xay dung baseline.
- Danh gia cross-validation.
- Tao submission.

## 2026-09-19 - Preprocessing completed

- Hoan thien `Data Preprocessing.ipynb` voi cell markdown giai thich moi quyet dinh tu EDA.
- Xu ly semantic structural absence theo row-level rules; giu `MasVnrType=None` la category hop le.
- Fit imputer, OneHotEncoder va StandardScaler chi tren training split.
- Dung `log1p(SalePrice)`, split 80/20 voi seed 42 va Ridge baseline.
- Validation log-RMSE `0.1437`, log-MAE `0.0963`; output 314 features; khong con NaN sau transform.
- Tao `data/processed/` cho nguoi train model va `outputs/preprocessing/` cho artifacts.
- Cap nhat `memory-bank/preprocessing-handoff.md` lam contract giua EDA, preprocessing va modeling.

## Chua thuc hien sau preprocessing

- PyTorch Dataset/DataLoader.
- MLP, training loop va early stopping.
- Cross-validation, experiment tracking va submission Kaggle.

## 2026-09-19 - Baseline completed
- Hoan thanh baseline voi 2 model LinearRegression va RandomForestRegressor

