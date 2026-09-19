# System Patterns

## Quy trinh thi nghiem

1. Notebook dung de EDA va thu nghiem.
2. Logic on dinh co the chuyen vao `src/` khi can tai su dung.
3. Ket qua va submission duoc luu trong `outputs/`.

## Tai lap

Moi ket qua can ghi ro cach chia du lieu, metric, tham so va random seed neu co su dung.

## Data semantics

1. Doc raw CSV voi `keep_default_na=False` de bao toan token `NA` va category literal `None`.
2. Xac dinh structural absence theo tung dong bang feature dong hanh, khong map theo toan bo cot.
3. Giu rieng structural-absence indicator va unknown-missing indicator khi storage representation deu la `NaN`, nhu `GarageYrBlt`.
4. Chi fit imputer, scaler va encoder tren training split; test-only levels chi dung de audit.

## Preprocessing handoff pattern

1. Notebook preprocessing phai co markdown giai thich Observation -> Evidence -> Implication -> Action.
2. CSV model-ready va feature names duoc luu trong `data_processed/` de nguoi train model dung truc tiep.
3. Fitted pipeline va validation predictions duoc luu trong `outputs/preprocessing/`.
4. Nguoi train model khong duoc preprocess lai CSV model-ready; neu thay doi rule phai chay lai notebook va cap nhat memory bank.
5. Target regression dung log scale khi metric/project yeu cau log-RMSE; dung `expm1` khi tao prediction o scale gia goc.
