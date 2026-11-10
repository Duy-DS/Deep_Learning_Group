# 📜 AGENTS.md — HIẾN PHÁP DỰ ÁN BTL01: FUNCTION APPROXIMATION (MLP)

Tài liệu này là **Hiến pháp tối cao (System Constitution)** điều chỉnh hành vi, kiến trúc mã nguồn, quy trình thực nghiệm và tiêu chuẩn báo cáo của mọi AI Agent cũng như lập trình viên tham gia vào dự án **Bài tập lớn 01 (BTL01) — Xấp xỉ hàm số dùng mạng Multi-Layer Perceptron (MLP)** thuộc môn học **Deep Learning**.

Mọi Agent khi bắt đầu phiên làm việc **BẮT BUỘC** phải đọc và tuân thủ tuyệt đối các điều khoản dưới đây.

---

## 🎯 Điều 1: Bối cảnh & Mục tiêu Dự án (Project Context)

1. **Bài toán cốt lõi:** Xấp xỉ hàm mật độ xác suất Gaussian 2 chiều (2-D Gaussian Function) bằng mạng nơ-ron truyền thẳng nhiều lớp (MLP):
   $$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}[\det(\mathbf{C}_{\mathbf{X}})]^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_{\mathbf{X}})^T \mathbf{C}_{\mathbf{X}}^{-1}(\mathbf{x} - \boldsymbol{\mu}_{\mathbf{X}})\right)$$
   * Số chiều đầu vào: $n = 2$, $\mathbf{x} = (x_1, x_2)^T$ với miền giá trị $x_1 \in [-2, 2]$, $x_2 \in [-4, 4]$.
   * Kỳ vọng: $\boldsymbol{\mu}_{\mathbf{X}} = [0, 0]^T$.
   * Ma trận hiệp phương sai: $\mathbf{C}_{\mathbf{X}} = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$ (hoặc theo cấu hình chuẩn của bài tập).
2. **Ba kịch bản Thực nghiệm Bắt buộc (Experiments A, B, C):**
   * **Thực nghiệm A (Khảo sát số lượng mẫu huấn luyện $A$):** Thay đổi $A \in \{200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000\}$ mẫu; tập test cố định $100$ mẫu sinh ngẫu nhiên cùng phân phối.
   * **Thực nghiệm B (Khảo sát số nơ-ron tầng ẩn $B$):** Mạng MLP 1 tầng ẩn (2 đầu vào + bias $\rightarrow$ $B$ nơ-ron ẩn $\rightarrow$ 1 đầu ra), khảo sát $B \in \{1, 2, \dots, 10\}$ (có thể mở rộng) để phân tích hiện tượng Underfitting và Overfitting.
   * **Thực nghiệm C (Khảo sát độ sâu mạng $C$):** Mạng MLP nhiều hơn 2 tầng ẩn ($C$ tầng ẩn), đánh giá sai số trên tập test, tốc độ hội tụ của hàm loss và thời gian huấn luyện.

---

## 🛠️ Điều 2: Ràng buộc Công nghệ & Nền tảng (Tech Stack Mandates)

1. **Framework Duy nhất:** Bắt buộc sử dụng **Python (3.10+)** và **PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)**.
2. **Công nghệ bị CẤM tuyệt đối:**
   * **KHÔNG** sử dụng TensorFlow, Keras, JAX, scikit-learn `MLPRegressor` hoặc bất kỳ thư viện AutoML/High-level wrapper nào (như PyTorch Lightning, FastAI) thay thế cho việc tự cài đặt vòng lặp huấn luyện.
3. **Cài đặt Vòng lặp Huấn luyện Thủ công:**
   * Phải tự viết rõ ràng vòng lặp `train` và `validation`: `forward` $\rightarrow$ `loss` $\rightarrow$ `optimizer.zero_grad()` $\rightarrow$ `loss.backward()` $\rightarrow$ `optimizer.step()`.
   * Bắt buộc chuyển đổi trạng thái `model.train()`, `model.eval()` và sử dụng ngữ cảnh `with torch.no_grad():` đúng quy chuẩn.
4. **Tính Tái lập (Reproducibility) & Thiết bị:**
   * Luôn cố định seed (`random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) trước mọi thực nghiệm.
   * Hỗ trợ chạy mượt mà trên **CPU** (mặc định) và tự động nhận diện **GPU (`cuda` / `mps`)** nếu khả dụng thông qua cấu hình.
   * **KHÔNG** bao giờ hardcode đường dẫn cá nhân tuyệt đối (ví dụ: `C:\Users\...`). Luôn dùng `pathlib.Path` hoặc đường dẫn tương đối tính từ thư mục gốc dự án.

---

## 🏛️ Điều 3: Tiêu chuẩn Kiến trúc Phần mềm & OOP (Coding Standards)

Toàn bộ mã nguồn trong dự án phải được thiết kế theo **Lập trình Hướng đối tượng (OOP)**, tách biệt rõ ràng các tầng trách nhiệm (Separation of Concerns):

1. **Thiết kế Class bắt buộc:**
   * `GaussianDataGenerator` & `GaussianDataset(torch.utils.data.Dataset)`: Quản lý công thức toán học Gaussian 2D, sinh mẫu ngẫu nhiên, chia tập train/val/test, chuẩn hóa (nếu có) và đóng gói `DataLoader`.
   * `FunctionApproximatorMLP(torch.nn.Module)`: Định nghĩa kiến trúc MLP linh hoạt theo tham số số tầng ẩn ($C$), danh sách số nơ-ron mỗi tầng ($B$), và hàm kích hoạt (Sigmoid, Tanh, ReLU...).
   * `Trainer`: Quản lý vòng lặp huấn luyện, theo dõi lịch sử `train_loss`, `val_loss` theo từng epoch, cơ chế Early Stopping và lưu checkpoint mô hình tốt nhất (`.pt` / `.pth`) dựa trên validation loss.
   * `Evaluator`: Tính toán các chỉ số đánh giá bắt buộc gồm **MSE** và ít nhất một chỉ số bổ sung (**MAE** hoặc **RMSE**), cùng thời gian huấn luyện.
   * `Visualizer`: Vẽ và xuất hình ảnh chất lượng cao (`.png`/`.pdf`, tối thiểu 300 DPI) bao gồm: bề mặt hàm Gaussian 3D thật vs. bề mặt xấp xỉ $\hat{y}$, đồ thị so sánh $y$ thật và $\hat{y}$ dự đoán, đường cong hội tụ Loss theo Epoch, và biểu đồ sai số theo $A, B, C$.
   * `ExperimentRunner`: Điều phối chạy tự động các kịch bản thực nghiệm A, B, C và xuất bảng kết quả tổng hợp ra file `.csv`.
2. **Tiêu chuẩn Docstring & Type Hints:**
   * **100% Modules, Classes và Methods/Functions** phải có **Docstring** theo chuẩn **Google Style** hoặc **NumPy Style** (mô tả mục đích, `Args`, `Returns`, `Raises`, kèm kích thước Tensor đầu vào/đầu ra như `Shape: (batch_size, 2)`).
   * Bắt buộc sử dụng **Type Hints** đầy đủ cho mọi tham số và giá trị trả về (ví dụ: `def forward(self, x: torch.Tensor) -> torch.Tensor:`).
   * Tuân thủ chuẩn định dạng **PEP 8**.
   * Ngôn ngữ ưu tiên: tiếng Việt, các thuật ngữ kĩ thuật có thể giữ nguyên mẫu hoặc tiếng Anh.

---

## 📂 Điều 4: Cấu trúc Thư mục Chuẩn (Project Directory Structure)

```text
BTL01_Function_Approximation/
├── AGENTS.md                  # Hiến pháp dự án (File này - Không tự ý xóa)
├── README.md                  # Hướng dẫn cài đặt, cấu trúc và lệnh tái lập thực nghiệm
├── requirements.txt           # Danh sách thư viện và phiên bản cố định
├── configs/                   # Cấu hình siêu tham số cho thực nghiệm A, B, C (YAML/JSON/Python)
├── data/                      # Lưu trữ dữ liệu sinh ra (raw, processed)
│   ├── train_full.csv              # Tập dữ liệu huấn luyện gốc (hoặc các file chia theo số mẫu A)
│   ├── val.csv                     # Tập validation riêng để chọn mô hình / siêu tham số
│   └── test.csv                    # Tập test độc lập (100 mẫu)
├── src/                       # Mã nguồn chính chuẩn OOP
│   ├── __init__.py
│   ├── data_generator.py      # Class sinh dữ liệu Gaussian 2D & Dataset
│   ├── models.py              # Class định nghĩa kiến trúc MLP
│   ├── trainer.py             # Class huấn luyện, validation và lưu checkpoint
│   ├── evaluator.py           # Class tính toán MSE, RMSE, MAE
│   ├── visualizer.py          # Class vẽ đồ thị 2D/3D và đường cong hội tụ
│   └── runner.py              # Class ExperimentRunner điều phối thực nghiệm A, B, C
├── experiments/               # Các script chạy thực nghiệm tự động từ đầu
│   ├── generate_data.py       # Script chạy sinh dữ liệu cố định và vẽ biểu đồ khảo sát phân bố dữ liệu
│   ├── run_exp_A.py           # Chạy Thực nghiệm A (khảo sát quy mô dữ liệu: A = 200 -> 2000)
│   ├── run_exp_B.py           # Chạy Thực nghiệm B (khảo sát số nơ-ron tầng ẩn: B = 1 -> 10+)
│   ├── run_exp_C.py           # Chạy Thực nghiệm C (khảo sát độ sâu / số tầng ẩn C và thời gian huấn luyện)
│   └── run_all.py             # (Tùy chọn) Script chạy toàn bộ quy trình từ tạo dữ liệu đến xuất kết quả A/B/C
├── notebooks/                      # (Tùy chọn) Jupyter Notebook phục vụ trình bày demo hoặc chạy thử nghiệm
│   └── demo_exploration.ipynb      # Lưu ý: Không được nộp duy nhất file HTML xuất từ notebook
├── scripts/                   # dùng để chứa các tập tin mã kịch bản (script) phục vụ cho việc tự động hóa, hỗ trợ phát triển, hoặc vận hành hệ thống
├── drafts/                    # Chứa các bản nháp, ý tưởng sơ khai, hoặc mã nguồn đang thử nghiệm.
├── outputs/                   # Kết quả sinh ra tự động từ thực nghiệm
│   ├── checkpoints/           # Trọng số mô hình tốt nhất (.pt / .pth)
│   ├── figures/               # Đồ thị, biểu đồ phục vụ báo cáo và slide (.png)
│   └── tables/                # Bảng kết quả thực nghiệm (.csv)
├── docs/                      # Tài liệu báo cáo (.docx, .pdf) và Slide (.pptx)
│   ├── bug-fixes/             # Báo cáo sau các phiên bản fix bug
│   ├── dev-logs/              # Báo cáo việc đã làm trong một ngày, Lưu dev-log cuối mỗi ngày làm việc
│   ├── project_charter/       # Các tài liệu của dự án
│   ├── reports/               # Lưu báo cáo sau mỗi khi hoàn thành một Task 
│   ├── slides/                # Các tài liệu liên quan nhằm mục đích thiết kế tài liệu trình chiếu
│   └── thesis/                # Tài liệu văn bản báo cáo tổng kết của dự án
└── main.py                    # Điểm vào (Entry point) chạy toàn bộ pipeline qua CLI
```

---

## 📝 Điều 5: Quy định Ghi chép Nhật ký & Báo cáo Task (Logging & Reporting Protocol)

Đây là quy định **BẮT BUỘC** về mặt kỷ luật quy trình. Mọi tiến độ công việc đều phải được ghi nhận minh bạch bằng văn bản trong thư mục `docs/`.

### 5.1. Báo cáo Hoàn thành Task (Post-Task Report)
* **Thời điểm kích hoạt:** Ngay sau khi hoàn thành bất kỳ một Task nào trong dự án.
* **Định dạng tên file:** `[task ID]_[YYMMDD]_[HHMM].md`
  * Ví dụ: `TASK-01_261025_1430.md` (lưu tại `docs/reports/`).
* **Nội dung bắt buộc trong file:**
  1. **Thông tin chung:** Task ID, Tên Task, Thời gian bắt đầu - kết thúc, Trạng thái.
  2. **Mục tiêu Task:** Tóm tắt yêu cầu kỹ thuật của task.
  3. **Các thay đổi đã thực hiện (Changes Made):** Liệt kê các file đã tạo mới/chỉnh sửa, các Class/Method OOP đã triển khai.
  4. **Kiểm thử & Xác minh (Verification & Metrics):** Lệnh đã chạy thử nghiệm, kết quả đầu ra (số liệu MSE/RMSE/MAE, hình ảnh sinh ra, hoặc kết quả test).
  5. **Vấn đề phát sinh & Cách giải quyết (Issues & Resolutions):** Ghi chú các lỗi gặp phải và hướng xử lý.

### 5.2. Nhật ký Làm việc Cuối ngày (Daily Dev-Log)
* **Thời điểm kích hoạt:** Sau mỗi ngày làm việc (hoặc khi kết thúc phiên làm việc trong ngày).
* **Định dạng tên file:** `[YYMMDD]_[HHMM]_dev-log.md`
  * Ví dụ: `261025_1800_dev-log.md` (lưu tại `docs/dev-logs/`).
* **Nội dung bắt buộc trong file:**
  1. **Tổng quan ngày làm việc:** Ngày tháng, danh sách các Task ID đã xử lý trong ngày.
  2. **Tiến độ chi tiết:** Tóm tắt những tính năng, thực nghiệm (A/B/C) hoặc tài liệu đã hoàn thành.
  3. **Trạng thái Codebase & Dữ liệu hiện tại:** Những module nào đã ổn định, những checkpoint/biểu đồ nào đã được xuất.
  4. **Tồn đọng & Rủi ro (Blockers):** Các vấn đề chưa giải quyết xong.
  5. **Kế hoạch ngày tiếp theo (Next Steps):** Danh sách các Task dự kiến thực hiện vào ngày làm việc kế tiếp.

---

## 🚫 Điều 6: Những điều Agent TUYỆT ĐỐI KHÔNG ĐƯỢC LÀM (Never-Dos)

1. **KHÔNG** viết code dạng script trải dài (spaghetti code) thiếu cấu trúc Class OOP hoặc bỏ qua Docstring.
2. **KHÔNG** đánh giá mô hình hoặc chọn siêu tham số dựa trên tập Test (Data Leakage). Tập Test ($100$ mẫu) chỉ được dùng ở bước đánh giá cuối cùng; việc chọn model checkpoint bắt buộc dựa trên Validation Loss.
3. **KHÔNG** kết thúc một Task hoặc kết thúc ngày làm việc mà quên khởi tạo file `[task ID]_[YYMMDD]_[HHMM].md` hoặc `[YYMMDD]_[HHMM]_dev-log.md`.
4. **KHÔNG** tạo đồ thị thiếu tiêu đề (Title), thiếu nhãn trục (Axis Labels), thiếu chú thích (Legend) hoặc độ phân giải thấp không đọc được chữ khi đưa vào báo cáo Word/PDF và Slide PowerPoint.
5. **KHÔNG** cài đặt thêm thư viện ngoài danh sách chuẩn mà không cập nhật ngay lập tức vào `requirements.txt`.

---

## 🔄 Điều 7: Quy trình Thực thi Chuẩn cho mỗi Task (Standard Agent Workflow)

Mỗi khi nhận một yêu cầu hoặc Task mới từ người dùng, Agent phải tuần tự thực hiện 5 bước:

1. **Bước 1 — Phân tích & Định danh Task:** Xác định rõ `Task ID` (ví dụ: `TASK-01`, `TASK-02`...), đối chiếu yêu cầu với các điều khoản trong `AGENTS.md`.
2. **Bước 2 — Thiết kế OOP:** Xác định các Class, thuộc tính, phương thức và chữ ký hàm (Type Hints + Docstring) trước khi viết logic chi tiết.
3. **Bước 3 — Triển khai & Kiểm thử:** Viết code PyTorch sạch, chạy kiểm thử để đảm bảo code chạy lại từ đầu không lỗi.
4. **Bước 4 — Xuất kết quả (Artifacts):** Lưu dữ liệu `.csv`, biểu đồ `.png` và trọng số `.pt` vào đúng thư mục quy định trong `outputs/`.
5. **Bước 5 — Viết Báo cáo Log:** Tạo ngay file báo cáo hoàn thành task `[task ID]_[YYMMDD]_[HHMM].md` (và nhắc nhở/tạo `[YYMMDD]_[HHMM]_dev-log.md` khi kết thúc ngày làm việc).