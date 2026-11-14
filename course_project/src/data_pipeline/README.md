# Pipeline UCI Beijing PM2.5 — T0101 đến T0106

## Cài đặt
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

## Dữ liệu
Đặt file CSV UCI Beijing PM2.5 vào `data/raw/PRSA_data.csv`.
Bộ dữ liệu UCI phổ biến có các cột `year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir`.
Nếu dataset của nhóm dùng tên cột khác, chỉnh `TARGET_COL`/`wind_col` hoặc tên cột ở notebook.

## Thứ tự chạy
1. Chạy `notebooks/01_eda.ipynb`.
2. Import các hàm từ `src/data_pipeline/preprocess.py`.
3. Làm sạch, mã hóa, chia dữ liệu theo thời gian.
4. Fit scaler trên `train` và transform các tập còn lại.
5. Tạo sliding windows riêng từng tập với L=96, H=24/48/96/192.

## Tên split
`train`, `train_dev`, `dev`, `test` theo tỷ lệ 60/10/10/20 và theo thứ tự thời gian.

## Lưu ý chống Data Leakage
- Không shuffle khi chia dữ liệu chuỗi thời gian.
- Scaler chỉ fit trên train.
- Tạo windows riêng bên trong từng split, không để cửa sổ X/Y đi qua ranh giới tập.
- Không đưa `datetime` hoặc cột target vào feature_cols nếu không có chủ đích.
