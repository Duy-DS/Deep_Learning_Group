# Lab02: House Prices Prediction Using PyTorch

## Muc tieu

Du bao `SalePrice` cua nha o Ames bang Multilayer Perceptron (MLP) trong PyTorch theo chuẩn thực hành Data Science / MLOps chuyên nghiệp.

Tai lieu dieu hanh, phan cong va tien do cua nhom nam tai
[`memory-bank/project-management.md`](memory-bank/project-management.md).

Tom tat Ames Housing paper va cac insight dung cho EDA/preprocessing nam tai
[`memory-bank/ames-housing-paper-summary.md`](memory-bank/ames-housing-paper-summary.md).

Ke hoach va ket qua EDA nam tai
[`memory-bank/eda-understanding-plan.md`](memory-bank/eda-understanding-plan.md) va
[`memory-bank/eda-findings.md`](memory-bank/eda-findings.md).

## Du lieu

Toàn bộ dữ liệu được quản lý tập trung trong thư mục `data/`:
- `data/raw/`: Dữ liệu gốc bất biến từ Kaggle (`train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt`).
- `data/processed/`: Dữ liệu đã làm sạch và model-ready (`train_processed.csv`, `test_processed.csv`, `feature_names.csv`).

## Cau truc du an (Project Architecture)

```text
Lab02/
├── .gitignore                      # Bo qua checkpoint mo hinh lon va file rac
├── README.md                       # Huong dan va tai lieu du an
│
├── data/                           # Tang du lieu tap trung
│   ├── raw/                        # Du lieu goc bat bien
│   └── processed/                  # Du lieu model-ready sau tien xu ly
│
├── notebooks/                      # Pipeline nghien cuu tuan tu (theo so thu tu)
│   ├── 01_eda.ipynb                # 1. Kham pha va phan tich phan phoi
│   ├── 02_data_preprocessing.ipynb # 2. Tien xu ly du lieu va pipeline artifacts
│   ├── 03_baseline.ipynb           # 3. Mo hinh co so Ridge Regression
│   ├── 04_train_mlp.ipynb          # 4. Huan luyen va danh gia mang PyTorch MLP
│   └── archive/                    # Cac notebook phu / nhap cu (mlp_model.ipynb)
│
├── src/                            # Ma nguon Python tai su dung (Modular Codebase)
│   ├── config.py                   # Quan ly duong dan dong va hyperparameters
│   ├── data/
│   │   └── dataset.py              # PyTorch HousePriceDataset
│   ├── models/
│   │   └── mlp.py                  # Kien truc mang MLPRegressor
│   └── utils/
│       └── metrics.py              # Ham tinh toan RMSE va evaluation helpers
│
├── models/                         # Luu tru checkpoint mo hinh (.pth)
├── outputs/                        # Artifacts sinh ra tu qua trinh thuc nghiem
│   ├── figures/                    # Bieu do EDA va huan luyen
│   ├── metrics/                    # Bang chi so va feature audit
│   └── preprocessing/              # Fitted pipeline pickle va validation predictions
├── docs/                           # Tai lieu nghien cuu va bao cao
│   └── reports/                    # Paper tham khao va project charters
└── memory-bank/                    # Nhat ky va boi canh lam viec cua Lab02
```

## Quy trinh thuc hien (Pipeline Workflow)

1. **`01_eda.ipynb`**: Khám phá `data/train.csv`, kiểm tra missing semantics và phân phối `SalePrice`.
2. **`02_data_preprocessing.ipynb`**: Thực hiện chuẩn hóa đặc trưng, fit pipeline và xuất dữ liệu model-ready ra `data/processed/`.
3. **`03_baseline.ipynb`**: Tạo baseline có thể lặp lại và đánh giá cross-validation.
4. **`04_train_mlp.ipynb`**: Huấn luyện kiến trúc PyTorch MLP, so sánh và đánh giá hiệu năng trên validation split.

## Quy uoc

- Khong chinh sua truc tiep cac file trong `data/raw/`.
- Su dung module `src/config.py` hoac `src/` khi can mo rong va tai su dung ma nguon.
- Ghi ro du lieu chia, metric va tham so cua moi thi nghiem.
- Chi commit code, notebook da duoc lam sach, tai lieu va submission; khong commit checkpoint binary lon len Git.
