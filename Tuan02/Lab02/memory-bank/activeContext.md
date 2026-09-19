# Active Context

## Trang thai

T01 da hoan thanh. T03/T04 da co notebook, findings va exported artifacts; hien dang o trang thai `Review`.

EDA da chay top-to-bottom voi 15 code cells, audit du 79 features, xuat 9 figures va 7 metric tables. Missing semantics da duoc sua theo row-level rules de tach structural absence khoi unknown missing.

## Buoc tiep theo

1. Nhom review `01_eda.ipynb` va `eda-findings.md` de chuyen T03/T04 tu `Review` sang `Done`.
2. Ban giao `feature_audit.csv`, `missing_audit.csv`, `category_level_audit.csv` va `missing_target_effect.csv` cho TV3.
3. TV3 thiet ke preprocessing interface, fit tren training split va xu ly unseen categories.

## Luu y

- Target trong `train.csv` la `SalePrice`.
- `test.csv` khong co target.
- Dung `sample_submission.csv` de kiem tra thu tu va ten cot dau ra.
- Khong doc CSV bang NA defaults ma khong kiem tra literal `NA` va `None`.
- Khong map `NA -> None` theo toan bo cot basement/garage/pool; phai kiem tra feature dong hanh theo tung dong.
- `MSSubClass=150` chi xuat hien trong test; encoder phai co co che xu ly unseen level.
- 54 category levels co duoi 5 dong train; chua co quyet dinh gop rare categories.

## Cap nhat preprocessing 2026-09-19

Preprocessing da hoan thanh trong `notebooks/Data Preprocessing.ipynb` va chay top-to-bottom khong loi. Pipeline da chot semantic missing, numeric median imputation + missing indicators, categorical `Unknown`, one-hot encoding, `StandardScaler`, train/validation split 80/20 voi `random_state=42` va target `log1p(SalePrice)`.

Ket qua validation baseline: log-RMSE `0.1437`, log-MAE `0.0963`; ma tran sau encoding co 314 features va khong con NaN.

Data model-ready duoc luu tai `data_processed/`: `train_processed.csv`, `test_processed.csv`, `feature_names.csv`, kem `README.md`. Pipeline artifact va validation predictions nam tai `outputs/preprocessing/`.

T03/T04 da co findings va artifacts; T05/T06 da hoan thanh o muc pipeline baseline. Buoc tiep theo la PyTorch Dataset/DataLoader, baseline comparison va MLP.
