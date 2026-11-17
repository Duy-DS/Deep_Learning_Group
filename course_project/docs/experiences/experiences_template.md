# [EXPERIMENT] <Mã Thí Nghiệm>: <Tiêu đề tóm tắt thực nghiệm / ablation study>

- **Mã thực nghiệm (Experiment ID):** `EXP_[ID]` (Ví dụ: `EXP01`, `EXP02`)
- **Tên file lưu vết:** `docs/experiences/[ex ID]_[YYMMDD]_[HHMM].md`
- **Thời gian thực hiện:** YYYY-MM-DD HH:MM
- **Người thực hiện:** <Họ và tên thành viên phụ trách WP — chỉ ghi tên thành viên nhóm>
- **Work Package liên quan:** WP02 | WP03 | WP04 | WP06
- **Task ID:** `BPM25_<Task_ID>` (Ví dụ: `BPM25_T0305`, `BPM25_T0601`)
- **Nhánh Git (Branch):** `exp/BPM25_<Task_ID>_<ten_cong_viec>`
- **Trạng thái thực nghiệm:** [Success | Failed | Inconclusive]

---

## 1. Mục tiêu & Giả thuyết nghiên cứu (Objective & Research Hypothesis)
- **Mục tiêu thực nghiệm:** Nêu cụ thể mục tiêu kiểm chứng của lần chạy này (ví dụ: So sánh khả năng dự báo dài hạn giữa PatchTST và LSTM baseline; hoặc khảo sát tác động của cơ chế ProbSparse Attention).
- **Liên kết câu hỏi nghiên cứu (RQs):** Đóng góp trực tiếp câu trả lời cho [RQ1 | RQ2 | RQ3 | Ablation Study].
- **Giả thuyết khoa học (Hypothesis):** 
  - *Giả thuyết:* Việc sử dụng cơ chế Patching và Channel-independence của PatchTST sẽ giúp giảm đáng kể MSE trên horizon $H=96, 192$ so với baseline.
  - *Tiêu chí bác bỏ / chấp nhận:* MSE giảm tối thiểu 10% với độ lệch chuẩn nhỏ giữa các seeds.

---

## 2. Thiết lập thực nghiệm & Tính tái lập (Experimental Setup & Reproducibility)
- **Kiến trúc mô hình:** [LSTM | DLinear | Informer | Informer_NoProbSparse | PatchTST]
- **Chế độ dữ liệu:** `[multivariate | univariate]`
- **Phạm vi dữ liệu:** Trạm cụ thể hoặc toàn bộ 36 trạm quan trắc (`ALL_36_STATIONS`).
- **Phân chia chuỗi thời gian:** `60% Train / 10% Val / 10% Val-test / 20% Test` (Chia tuyến tính theo thời gian, tuyệt đối không shuffle).
- **Quy chuẩn chống rò rỉ (Data Leakage Guard):** StandardScaler chỉ fit trên tập Train, sau đó transform cho Val/Test.
- **Cấu hình trượt (Sliding Window):**
  - Chiều dài cửa sổ lịch sử (Lookback $L$): `96`
  - Khung thời gian dự báo (Horizon $H$): `[24 | 48 | 96 | 192]`
- **Danh sách Random Seeds:** `[42, 1024, 2026]` (Cố định seed cho `numpy`, `torch`, `torch.cuda`, Python `random`).
- **Siêu tham số chính (Đọc từ `configs/default.yaml`):**
  - Batch Size: `32`
  - Learning Rate: `0.0001`
  - Optimizer: `AdamW` (hoặc Adam)
  - Loss Function: `MSELoss`
  - Early Stopping Patience: `10`
  - Checkpoint Path: `checkpoints/EXP_[ID]_[model]_[station]_best.pth` (đã gitignored)

---

## 3. Lệnh thực thi tái hiện (Reproduction Command)
```bash
# Lệnh bash hoặc python command chính xác dùng để tái hiện kết quả
python training_loop.py --config configs/default.yaml --models informer --horizon 96 --seeds 42 1024 2026

# Hoặc thực thi qua script tự động 3 seeds:
bash scripts/train.sh --exp EXP01 --model patchtst --horizon 96
```

---

## 4. Bảng kết quả định lượng (Quantitative Results)
*Bắt buộc ghi nhận đầy đủ kết quả của từng seed và tổng hợp dưới dạng $\text{mean} \pm \text{std}$:*

| Mô hình / Biến thể | Horizon ($H$) | Seed 42 (MSE / MAE) | Seed 1024 (MSE / MAE) | Seed 2026 (MSE / MAE) | Kết quả tổng hợp ($\text{mean} \pm \text{std}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PatchTST (Multivariate)** | 24 | 0.0381 / 0.1412 | 0.0390 / 0.1435 | 0.0379 / 0.1408 | **MSE:** 0.0383 ± 0.0006<br>**MAE:** 0.1418 ± 0.0015 |
| **PatchTST (Multivariate)** | 96 | 0.0512 / 0.1680 | 0.0525 / 0.1701 | 0.0518 / 0.1690 | **MSE:** 0.0518 ± 0.0007<br>**MAE:** 0.1690 ± 0.0011 |
| **LSTM (Baseline)** | 96 | 0.0685 / 0.1982 | 0.0701 / 0.2014 | 0.0694 / 0.1998 | **MSE:** 0.0693 ± 0.0008<br>**MAE:** 0.1998 ± 0.0016 |

---

## 5. Bằng chứng thực thi & Terminal Logs (Execution Evidence)
```text
[SEED 42] Epoch 32/100 - Train Loss: 0.0321 - Val Loss: 0.0378 - Patience: 4/10
...
[EVALUATION FINISHED] Test MSE: 0.03833 ± 0.00058 | Test MAE: 0.14183 ± 0.00147
Trọng số tốt nhất đã được lưu tại: checkpoints/EXP01_patchtst_H96_best.pth
[VERIFIED] Không phát hiện rò rỉ dữ liệu giữa tập Train và Test.
```

---

## 6. Phân tích & Đánh giá kết quả (Discussion & Findings)
- **So sánh hiệu năng:** Mô hình có vượt trội hơn baseline không? Đạt tỷ lệ cải thiện bao nhiêu %?
- **Độ ổn định thống kê:** Giá trị $\text{std}$ giữa các lần chạy ngẫu nhiên có nhỏ không? Có seed nào bị bất thường hoặc rớt điểm tối ưu cục bộ không?
- **Độ phức tạp tính toán & VRAM:** Tốc độ huấn luyện (giây/epoch) và mức độ tiêu thụ bộ nhớ GPU ra sao khi nâng khung dự báo $H$ từ 24 lên 192?
- **Phân tích sai số tại các đỉnh ô nhiễm (Smog Spikes):** Mô hình có bắt kịp xu hướng biến động mạnh của $\text{PM}_{2.5}$ khi độ ẩm điểm sương (DEW) và hướng gió (WD) thay đổi không?

---

## 7. Kết luận & Hành động tiếp theo (Actionable Next Steps)
- [ ] Xác nhận chấp nhận / bác bỏ giả thuyết khoa học ban đầu.
- [ ] Bàn giao checkpoint tốt nhất cho WP04 để trích xuất giải thích SHAP / Integrated Gradients.
- [ ] Bàn giao metrics tổng hợp cho WP05 để đưa lên bảng so sánh trên Streamlit Dashboard.
- [ ] Cập nhật nhật ký phiên làm việc tại `docs/dev-logs/[YYMMDD]_[HHMM].md`.