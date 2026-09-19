# Lab02: House Prices Prediction Using PyTorch

## Muc tieu

Du bao `SalePrice` cua nha o Ames bang Multilayer Perceptron (MLP) trong PyTorch.

Tai lieu dieu hanh, phan cong va tien do cua nhom nam tai
[`memory-bank/project-management.md`](memory-bank/project-management.md).

Tom tat Ames Housing paper va cac insight dung cho EDA/preprocessing nam tai
[`memory-bank/ames-housing-paper-summary.md`](memory-bank/ames-housing-paper-summary.md).

Ke hoach va ket qua EDA nam tai
[`memory-bank/eda-understanding-plan.md`](memory-bank/eda-understanding-plan.md) va
[`memory-bank/eda-findings.md`](memory-bank/eda-findings.md).

## Du lieu

- `data/train.csv`: du lieu huan luyen, bao gom cot muc tieu `SalePrice`.
- `data/test.csv`: du lieu can du bao.
- `data/sample_submission.csv`: dinh dang nop bai tham chieu.
- `data/data_description.txt`: mo ta cac bien.

## Cau truc

```text
data/        Du lieu dau vao.
docs/        Tai lieu va tai lieu tham khao.
notebooks/   Kham pha du lieu va thi nghiem mo hinh.
outputs/     Bieu do, chi so va cac file submission.
src/         Ma nguon tai su dung khi pipeline duoc tach tu notebook.
memory-bank/ Nhat ky va boi canh lam viec cua Lab02.
```

## Quy trinh de xuat

1. Kham pha `train.csv`, xac dinh du lieu thieu va phan phoi `SalePrice`.
2. Tao baseline co the lap lai va danh gia bang cross-validation.
3. Thu nghiem xu ly dac trung va mo hinh; ghi ket qua vao `memory-bank/progress.md`.
4. Sinh `outputs/submission.csv` voi hai cot `Id` va `SalePrice` theo `sample_submission.csv`.

## Quy uoc

- Khong chinh sua truc tiep cac file trong `data/`.
- Ghi ro du lieu chia, metric va tham so cua moi thi nghiem.
- Chi commit code, notebook da duoc lam sach, tai lieu va submission can chia se; khong commit model/checkpoint lon.
