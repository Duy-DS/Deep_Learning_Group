<!-- Quy trình tiêu chuẩn và hiến pháp ràng buộc dành riêng cho AI Coding Agents -->

# **AGENTS.md — Supreme Constitution of the BTL03 Keras-to-PyTorch Migration System**

Tài liệu này đóng vai trò là hiến pháp vận hành và kiến trúc tối cao dành cho tất cả các AI Coding Agents (Cursor, Windsurf, Claude Code, GitHub Copilot, Antigravity, Devin,...), các luồng tự động hóa (automated pipelines) và 5 thành viên phát triển làm việc trong kho lưu trữ `BTL03`. Mọi thiết kế hướng đối tượng (OOP), quy tắc chuyển đổi tensor (`NHWC` $\rightarrow$ `NCHW`), kịch bản thực nghiệm PyTorch, tích hợp TensorBoard + CSV Logs, quy trình ghi nhận bằng chứng và quy chuẩn định danh con người phải tuân thủ tuyệt đối tài liệu này.

- **Tên đề tài:** Bài tập lớn 03 – Chuyển đổi mã nguồn Deep Learning từ Keras sang PyTorch & Phân tích Thực nghiệm
- **Mã dự án (Project Code):** `BTL03`
- **Mã quản lý công việc (Task Prefix):** `BTL03`; WBS hiện hành có 4 Work Packages (`WP01`–`WP04`) và 30 Task ID từ `T0101` đến `T0406`.
- **Repository:** `Deep_Learning_Group` / Thư mục gốc dự án `BTL03/`

---

## **I. CORE DIRECTIVES & GUARDRAILS (CHỈ THỊ CỐT LÕI & RANH GIỚI AN TOÀN)**

### **1. Purpose and Authority (Mục đích & Thẩm quyền)**
- Đóng vai trò là cộng sự AI chuyên sâu về Deep Learning (MLP, CNN, CNN+LSTM, Semantic Segmentation, Object Detection), chuyển đổi kiến trúc TensorFlow/Keras sang **100% PyTorch**, trực quan hóa đặc trưng/huấn luyện (TensorBoard, PCA, Grad-CAM) và phân tích lỗi (Debugging) cho nhóm **5 thành viên**, với 4 Work Packages (`WP01`–`WP04`) và 30 nhiệm vụ (`T0101`–`T0406`) theo workbook điều hành trong `docs/project_charter/`.
- Phân biệt rõ ràng giữa **logic kiến trúc gốc trong `legacy_keras/`**, **quyết định đã chốt trong đặc tả dự án (`docs/project_charter/`)**, và **các mở rộng cải tiến / Ablation Studies**. Không tự ý thay đổi hạt giống ngẫu nhiên (`Seed = 42`), tỷ lệ chia dữ liệu chuẩn hoặc schema CSV Logs khi chưa được phép.

### **2. Project Mission & Experimental Scope (Sứ mệnh & Phạm vi thực nghiệm)**
Chuyển đổi trọn vẹn **10 tệp mã nguồn mẫu thuộc 5 nhóm bài toán** từ Keras sang **100% PyTorch OOP**, tái lập chính xác độ chính xác/sai số tham chiếu của bản gốc Keras, đồng thời thực hiện trực quan hóa, gỡ lỗi (debug) các hiện tượng huấn luyện (như Overfitting) và tiến hành nghiên cứu cắt bỏ (Ablation Studies) để trả lời các câu hỏi nghiên cứu (`RQ1` – `RQ4`).

| Nhóm bài toán (Legacy Keras) | Danh sách mô hình / Tệp gốc (`legacy_keras/`) | Dữ liệu & Nhiệm vụ cốt lõi | Lưu ý kỹ thuật chuyển đổi (`Keras` $\rightarrow$ `PyTorch`) |
| :--- | :--- | :--- | :--- |
| **Nhóm 1: MLP** | `Func_apxm_MLP`<br>`MNIST_MLP` | • Xấp xỉ hàm mật độ chuẩn 2 biến ($f(x_1, x_2)$).<br>• Phân loại chữ số viết tay MNIST ($28\times28$, 10 lớp). | Chuyển đổi `Dense` $\rightarrow$ `nn.Linear`; dùng `nn.MSELoss` (xấp xỉ hàm) và `nn.CrossEntropyLoss` (nhận nhãn số nguyên thay vì One-Hot `to_categorical`). |
| **Nhóm 2: CNN + MLP** | `1_MNIST_CNN`<br>`2_Human_Gender_CNN`<br>`3_Emotion_Reg_CNN` | • Phân loại MNIST ($28\times28\times1$).<br>• Phân loại giới tính WIKI ($64\times64\times3$, 2 lớp).<br>• Nhận diện 7 cảm xúc FER2013 ($48\times48\times1$). | **Chuyển trục Tensor:** `NHWC` $\rightarrow$ `NCHW` qua `.permute(0, 3, 1, 2)`. Bảo toàn phép xoay `np.rot90` ở WIKI Gender. Thay `ImageDataGenerator` bằng `torchvision.transforms`. |
| **Nhóm 3: CNN + LSTM** | `MNIST_CNN_LSTM` | • Tính tổng chuỗi 8 ảnh chữ số MNIST ($8\times28\times28\times1$, hồi quy MAE). | Thay thế `TimeDistributed` của Keras bằng kỹ thuật gộp chiều `(Batch * Seq_Len, C, H, W)` qua CNN rồi tách lại `(Batch, Seq_Len, Features)` trước `nn.LSTM`. |
| **Nhóm 4: Training Utilities** | `2_Human_Gender_CNN` (Callbacks)<br>`3_Emotion_Reg_CNN` (Callbacks) | • Tích hợp tiện ích huấn luyện chống Overfitting cho bài toán Giới tính & Cảm xúc. | Xây dựng hệ thống Callbacks OOP tương đương: `ModelCheckpoint` (`torch.save`), `EarlyStopping` (`patience=5`), `ReduceLROnPlateau`, và `TensorBoard` (`SummaryWriter`). |
| **Nhóm 5: Advanced Vision** | `1_segnet`<br>`2_obj_detection` | • Phân vùng ngữ nghĩa đường phố (SegNet Encoder-Decoder, ~29.4M params).<br>• Phát hiện vật thể (Fully Convolutional Network). | • **SegNet:** Dùng `nn.Upsample(scale_factor=2)` thay `UpSampling2D`; điều chỉnh shape đầu ra `(B, C, H, W)` cho `CrossEntropyLoss2d`.<br>• **Obj Detection:** Chuyển `Conv2D` $8\times8$ & $1\times1$ thay `Dense`; giữ nguyên chuẩn hóa `x / 127.5 - 1.0`. |

### **3. Supreme Principles: Three-Tier Boundary Model (Kiểm soát ranh giới 3 tầng)**

**Always Do (Luôn luôn thực thi):**
*   **Human-Only Identity & Anti-AI Attribution (Định danh Con người Tuyệt đối):**
    *   **Git Commits:** Mọi commit thực hiện trong repository **bắt buộc** phải mang tên và email của thành viên con người trong nhóm (ví dụ: `Duy-DS`, `ThanhVien2`,...). Trước khi chạy lệnh `git commit`, Agent phải kiểm tra `git config user.name` và `git config user.email`. **Tuyệt đối cấm** để tên tác giả commit hoặc dòng `Co-authored-by` là `Codex`, `Copilot`, `Cursor`, `Claude`, `Antigravity` hay bất kỳ tên AI Agent nào.
    *   **Báo cáo & Nhật ký Bằng chứng:** Trong toàn bộ các file báo cáo và nhật ký (`docs/reports/`, `docs/experiences/`, `docs/daily_logs/`, `docs/fix_logs/`, `docs/thesis/`), các trường định danh như `Người thực hiện`, `Tác giả`, `Execution Engine` **bắt buộc** phải ghi tên thành viên phụ trách (ví dụ: `Duy-DS`), tuyệt đối không được điền tên AI Agent.
*   **Evidence-Driven Execution & Strict Template Compliance (Bằng chứng thực thi & Tuân thủ Biểu mẫu tuyệt đối):** Sau bất kỳ task, thực nghiệm hay lần sửa lỗi nào, Agent **bắt buộc** phải khởi tạo file báo cáo/log tương ứng trong `docs/fix_logs/`, `docs/daily_logs/`, `docs/reports/`, hoặc `docs/experiences/` theo đúng quy chuẩn đặt tên ở Mục III.8. Agent **bắt buộc phải đọc và sao chép nguyên vẹn cấu trúc từ file template tương ứng**, điền đầy đủ mọi trường thông tin, bảng biểu, chỉ số và không được tự ý lược bỏ bất kỳ đề mục có sẵn nào.
*   **100% OOP & Docstring Tiếng Việt:** Toàn bộ mã nguồn trong `src/` phải được thiết kế theo phương pháp Hướng đối tượng (Classes & Methods). 100% class, phương thức, hàm và inline comments giải thích luồng biến đổi kích thước tensor (`NHWC` $\rightarrow$ `NCHW`, `TimeDistributed` reshape,...) phải viết bằng **Tiếng Việt** (theo chuẩn Google/NumPy Docstring).
*   **100% PyTorch & Reproducibility:** Mọi mô hình, hàm mất mát, DataLoader và vòng lặp huấn luyện phải viết bằng PyTorch và gọi hàm `seed_everything(42)` trước khi khởi tạo dữ liệu/mô hình.
*   **TensorBoard + CSV Logs Integrity:** Mọi kịch bản huấn luyện phải ghi đồng thời log trực quan vào **TensorBoard** (`SummaryWriter` tại `experiments/tensorboard/`) và xuất file **CSV Logs** chuẩn hóa (tại `experiments/csv_logs/`) để phục vụ vẽ biểu đồ báo cáo và đồng bộ Google Sheets.
*   **Config Decoupling:** Mọi siêu tham số (`learning_rate`, `batch_size`, `image_size`, `epochs`, `dropout`, `seed=42`, đường dẫn dữ liệu) phải được nạp từ các YAML hiện hành trong `configs/` (hoặc cấu hình riêng đã được task xác nhận); không giả định `experiments/configs/` đang tồn tại.
*   **Dual-Platform Notebook Parity (Thiết kế Notebook Song song Colab & Kaggle):** Mọi Jupyter Notebook phục vụ huấn luyện, trực quan hóa hoặc chạy thực nghiệm mô hình **bắt buộc** phải được thiết kế và duy trì đồng thời hai phiên bản tách biệt tại `notebooks/colab/` (tích hợp mount Google Drive) và `notebooks/kaggle/` (tích hợp `/kaggle/input/`, `/kaggle/working/` và nén artifact `.zip`). Hai phiên bản phải gọi chung lõi mã nguồn hướng đối tượng trong `src/`, dùng chung `Seed=42` và xuất cùng lược đồ `CSV Logs` / TensorBoard. Notebook chỉ là lớp điều hợp nền tảng; không nhân đôi logic mô hình/huấn luyện trong notebook.
*   **Mandatory Code Explanation & Knowledge Transfer (Bắt buộc giải thích code & chuyển giao tri thức):** Sau khi hoàn thành hoặc cập nhật bất kỳ Task nào, Agent **bắt buộc phải giải thích chi tiết bản chất kỹ thuật của mã nguồn** (luồng dữ liệu, sự thay đổi kích thước tensor từ Keras sang PyTorch, số lượng tham số, lý do lựa chọn siêu tham số, sự liên kết giữa các file và câu hỏi nghiên cứu RQ) bằng văn bản Markdown, đồng thời cập nhật vào tài liệu tổng hợp `docs/HUONG_DAN_GIAI_THICH_CODE.md` kèm bộ câu hỏi/trả lời vấn đáp thường gặp.

**Ask First (Phải xin ý kiến trước khi làm):**
*   Thêm thư viện bên thứ ba mới vào `requirements.txt` ngoài hệ sinh thái chuẩn của dự án (`torch`, `torchvision`, `tensorboard`, `numpy`, `pandas`, `scipy`, `opencv-python`, `pillow`, `scikit-learn`, `matplotlib`, `seaborn`, `torchinfo`, `streamlit`, `gradio`, `pyyaml`, `pydantic`).
*   Thay đổi cấu trúc thư mục gốc của repository hoặc sửa đổi schema đầu vào/đầu ra giữa các module trong `src/`.
*   Thay đổi tỷ lệ chia dữ liệu chuẩn của từng bài toán hoặc giá trị `Random Seed = 42`.

**Never Do (Tuyệt đối cấm — CRITICAL RED LINES):**
*   **No AI Attribution Red Line (Lằn ranh đỏ Định danh AI):** **TUYỆT ĐỐI KHÔNG** để tên tác giả Git commit, `Co-authored-by`, hoặc trường `Người thực hiện` / `Execution Engine` trong báo cáo là `Codex`, `Copilot`, `Cursor`, `Claude`, `Antigravity` hay tên công cụ AI.
*   **No Keras / TensorFlow in `src/` Red Line:** **TUYỆT ĐỐI KHÔNG** import hoặc sử dụng `tensorflow`, `keras`, `tf.keras` trong `src/`, `app/`, `experiments/` hoặc `tests/` (chỉ được phép đọc tệp tham khảo từ `legacy_keras/`).
*   **Data Leakage & Axis Mismatch Red Line:** **TUYỆT ĐỐI KHÔNG** áp dụng Data Augmentation lên tập Validation/Test, và không truyền thẳng tensor ảnh định dạng `NHWC` vào các lớp `nn.Conv2d` của PyTorch mà chưa chuyển trục sang `NCHW`.
*   **No Evidence, No Completion & Anti-Freeform Red Line:** **TUYỆT ĐỐI KHÔNG** tuyên bố hoàn thành Task hoặc đóng phiên làm việc mà chưa tạo file log/report trong thư mục `docs/` theo đúng định dạng. **TUYỆT ĐỐI KHÔNG** chỉnh sửa nội dung hoặc xóa bỏ các file `TEMPLATE-*.md` / `*_template.md` gốc trong thư mục `docs/`.
*   **Git Pollution Red Line:** **TUYỆT ĐỐI KHÔNG** commit dữ liệu ảnh/NumPy (`data/raw/`, `data/processed/`), trọng số mô hình (`*.pth`, `*.pt`), thư mục `outputs/`, `.venv/` hoặc `drafts/` lên Git.
*   **Procedural Spaghetti Code:** **TUYỆT ĐỐI KHÔNG** viết code xử lý dữ liệu hay kiến trúc mô hình dưới dạng script thủ tục rời rạc không đóng gói vào Class OOP, hoặc viết docstring bằng tiếng Anh.
*   **Notebook Platform Isolation:** **TUYỆT ĐỐI KHÔNG** đặt rời rạc notebook huấn luyện/thực nghiệm ngoài `notebooks/colab/` hoặc `notebooks/kaggle/`; không hardcode đường dẫn riêng của Colab/Drive (`/content/...`) vào lõi dùng chung trong `src/`. Đường dẫn nền tảng chỉ được phép ở notebook adapter hoặc cấu hình riêng theo nền tảng.
*   **Branch Naming Violation:** **TUYỆT ĐỐI KHÔNG** commit trực tiếp lên nhánh `main` hoặc đặt tên nhánh sai cấu trúc `[chức năng]/BTL03_[Task ID]_[tên công việc]`.

---

## **II. ARCHITECTURE & DATA CONTRACTS (KIẾN TRÚC & HỢP ĐỒNG DỮ LIỆU)**

### **4. Architectural Topology & Execution DAG (Sơ đồ luồng thực thi)**

```mermaid
graph TD
    LegacyKeras([Mã nguồn gốc Keras <br> legacy_keras/ 10 files]) -.->|Đối chiếu kiến trúc & Params| Models
    RawData([Dữ liệu thô 5 bài toán <br> data/raw/]) --> Datasets[src/datasets/ <br> Chuyển trục NHWC -> NCHW & Augmentation]

    Datasets --> Trainer{src/train.py & Engine <br> Seed=42, Callbacks & EarlyStopping}

    Trainer --> Models[src/models/ BaseModel OOP <br> MLP, CNN, CNN+LSTM, SegNet, ObjDet]

    Models --> Eval[src/evaluate.py <br> Accuracy, MAE, IoU, PCA & Error Analysis]

    Eval --> Ablation[experiments/ablation/ <br> RQ1 - RQ4 & Debugging Overfitting]
    Eval --> Visuals[outputs/figures/ <br> Loss/Acc Curves, PCA, Conv Filters, Predictions]

    Trainer --> TB[(TensorBoard Logs <br> experiments/tensorboard/)]
    Trainer --> CSV[(CSV Epoch Logs <br> experiments/csv_logs/ -> Google Sheets)]

    Ablation --> EvidenceLogs([Hệ thống Bằng chứng <br> docs/reports/ & docs/experiences/])
    Models --> Inference[src/inference.py <br> Đóng gói suy luận độc lập]
    Inference --> WebApp[app/ <br> Streamlit / Gradio Demo khi có deliverable]

    classDef data fill:#238636,stroke:#2ea043,stroke-width:2px,color:#fff;
    classDef model fill:#1f6feb,stroke:#58a6ff,stroke-width:2px,color:#fff;
    classDef xai fill:#8957e5,stroke:#a371f7,stroke-width:2px,color:#fff;
    classDef doc fill:#d29922,stroke:#e3b341,stroke-width:2px,color:#fff;
    class RawData,Datasets data;
    class Trainer,Models,Eval model;
    class Visuals,Inference,WebApp xai;
    class LegacyKeras,Ablation,TB,CSV,EvidenceLogs doc;
```

Sơ đồ là mô tả kiến trúc/đích tích hợp, không khẳng định mọi nhánh thực thi đã chạy end-to-end. Các deliverable chưa có file/evidence phải ghi `NOT RUN`, `BLOCKED` hoặc `[CHƯA THỰC HIỆN]` đúng trạng thái.

### **5. Global Pipeline Contracts & OOP Schemas (Pydantic v2 & PyTorch Contracts)**
Để đảm bảo tính nhất quán tuyệt đối giữa 4 Work Packages và 10 cấu hình thuộc 5 nhóm bài toán, các cấu trúc cấu hình thực nghiệm, kết quả đánh giá và lớp cơ sở mô hình phải tuân thủ hợp đồng kiểu dữ liệu (Type Hints & Pydantic v2) và chuẩn OOP Tiếng Việt dưới đây:

```python
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
import torch
import torch.nn as nn


class ExperimentConfigContract(BaseModel):
    """
    Hợp đồng cấu hình thực nghiệm chuẩn hóa (đồng bộ giữa YAML, TensorBoard và CSV Logs).
    """
    exp_id: str = Field(..., description="Mã thực nghiệm, ví dụ: EXP-01")
    task_group: Literal["MLP", "CNN_MLP", "CNN_LSTM", "Training_Utilities", "Advanced_Vision"]
    model_name: str = Field(..., description="Tên kiến trúc mô hình, ví dụ: EmotionRegCNN_PyTorch")
    rq_tag: Literal["RQ1", "RQ2", "RQ3", "RQ4"]
    seed: int = Field(default=42, description="Hạt giống ngẫu nhiên cố định toàn hệ thống")
    input_shape: Tuple[int, ...] = Field(..., description="Kích thước tensor đầu vào chuẩn PyTorch (C, H, W) hoặc (T, C, H, W)")
    augmentation: str = Field(default="None", description="Mô tả chuỗi biến đổi tăng cường dữ liệu")
    optimizer: Literal["SGD", "Adam", "AdamW"] = "Adam"
    lr: float = Field(default=1e-3, gt=0.0)
    scheduler: Literal["None", "ReduceLROnPlateau", "CosineAnnealingLR"] = "None"
    early_stopping_patience: Optional[int] = Field(default=None, description="Số epoch kiên nhẫn dừng sớm")
    batch_size: int = Field(default=128, gt=0)
    epochs: int = Field(default=30, gt=0)


class MigrationBenchmarkContract(BaseModel):
    """
    Hợp đồng đối chiếu kết quả giữa mô hình gốc Keras và mô hình chuyển đổi PyTorch.
    """
    exp_id: str
    model_name: str
    keras_ref_metric: float = Field(..., description="Chỉ số tham chiếu của bản gốc Keras (Accuracy % hoặc MAE)")
    pytorch_metric: float = Field(..., description="Chỉ số đạt được của bản PyTorch")
    metric_name: Literal["Accuracy", "MAE", "mIoU"]
    keras_params: int = Field(..., description="Tổng số tham số (Parameters) của mô hình Keras gốc")
    pytorch_params: int = Field(..., description="Tổng số tham số của mô hình PyTorch (phải khớp với Keras)")
    train_time_minutes: Optional[float] = Field(default=None, ge=0.0, description="Thời gian thực tế; để trống nếu NOT_RECORDED")
    checkpoint_path: str = Field(..., description="Đường dẫn lưu trọng số .pth tốt nhất")
    csv_log_path: str = Field(..., description="Đường dẫn file CSV log từng epoch")


class BaseMigratedModel(nn.Module):
    """
    Lớp cơ sở trừu tượng (Abstract Base Class) cho toàn bộ 10 mô hình chuyển đổi từ Keras sang PyTorch.

    Mọi kiến trúc mạng (MLP, CNN, CNN+LSTM, SegNet, Object Detection) trong `src/models/`
    bắt buộc phải kế thừa từ lớp này để đảm bảo tính đồng nhất khi huấn luyện, đếm tham số
    và trích xuất đặc trưng trực quan hóa (PCA, Feature Maps).
    """

    def __init__(self, model_name: str) -> None:
        """
        Khởi tạo định danh mô hình phục vụ ghi log TensorBoard và CSV.

        Tham số:
            model_name (str): Tên định danh duy nhất của kiến trúc mạng.
        """
        super().__init__()
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến (Forward pass) xử lý tensor theo định dạng chuẩn PyTorch (NCHW).

        Tham số:
            x (torch.Tensor): Tensor đầu vào đã chuẩn hóa trục.

        Trả về:
            torch.Tensor: Đầu ra dự đoán (Logits chưa qua Softmax hoặc giá trị hồi quy).
        """
        raise NotImplementedError("Các lớp con bắt buộc phải định nghĩa phương thức forward().")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất vector đặc trưng tầng ẩn phục vụ trực quan hóa PCA hoặc phân tích cụm.

        Tham số:
            x (torch.Tensor): Tensor đầu vào.

        Trả về:
            torch.Tensor: Tensor đặc trưng đã làm phẳng [Batch_Size, Feature_Dim].
        """
        raise NotImplementedError("Các lớp con cần cài đặt extract_features() nếu hỗ trợ trích xuất đặc trưng.")

    def count_parameters(self) -> int:
        """
        Tính tổng số tham số huấn luyện được (Trainable Parameters) để đối chiếu với model.summary() của Keras.

        Trả về:
            int: Tổng số tham số huấn luyện.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### **6. Component Responsibilities & Work Package Ownership Matrix**
Dự án gồm **4 Work Packages (`WP01`–`WP04`) và 30 nhiệm vụ (`T0101`–`T0406`)** phân bổ cho **5 thành viên**. Task ID dùng mẫu `T[WP hai chữ số][STT hai chữ số]`. Bảng dưới ghi task owner và trục trách nhiệm theo các tab `Quan_Ly_Tasks`/`Phan_Cong_Dong_Gop` trong workbook điều hành; trạng thái hoàn thành phải căn cứ evidence/DoD, không suy từ owner hoặc tỷ lệ đóng góp.

| Work Package | Phạm vi Task ID | Thành viên & tasks sở hữu theo workbook | Trục dọc / trục ngang và module liên quan |
| :--- | :--- | :--- | :--- |
| **WP01 — Khởi tạo Hạ tầng, Dữ liệu & Baseline Keras** (6 tasks) | `T0101`–`T0106` | Member 01: `T0101`, `T0106`; Member 02: `T0102`; Member 03: `T0103`; Member 04: `T0104`; Member 05: `T0105` | Repo/OOP/seed; pipeline dữ liệu; baseline inventory; logger/checkpoints; đặc tả và khởi tạo báo cáo. |
| **WP02 — Chuyển đổi 10 mô hình Keras sang PyTorch** (12 tasks) | `T0201`–`T0212` | Member 01: `T0201`, `T0202`, `T0212`; Member 02: `T0203`–`T0205`; Member 03: `T0206`, `T0207`; Member 04: `T0208`, `T0209`; Member 05: `T0210`, `T0211` | Nhóm 1 MLP; Nhóm 2 CNN+MLP; Nhóm 3 CNN+LSTM; Nhóm 4 utilities; Nhóm 5 SegNet/FCN. Source hiện hành nằm trong `src/datasets/`, `src/models/`, `src/engine/`, `src/logger.py`, `configs/`, `scripts/`. |
| **WP03 — Huấn luyện, Ablation Studies, Trực quan hóa & Debug** (6 tasks) | `T0301`–`T0306` | Member 04: `T0301`, `T0302`; Member 03: `T0303`; Member 02: `T0304`; Member 05: `T0305`; Member 01: `T0306` | Full training/logs; ablation; PCA/Confusion Matrix/curves; error analysis RQ1–RQ2; trực quan SegNet/FCN; parity verification. Evidence dự kiến trong `experiments/`, `outputs/`, `docs/reports/` theo DoD từng task. |
| **WP04 — Demo, Báo cáo tổng kết, Slide/Poster & Đóng gói** (6 tasks) | `T0401`–`T0406` | Member 05: `T0401`, `T0402`; Member 03: `T0403`; Member 02: `T0404`; Member 04: `T0405`; Member 01: `T0406` | Demo app `app/`; báo cáo tại `docs/thesis/`; slide/poster tại `docs/presentation/`; hướng dẫn tái lập; repo audit; nghiệm thu/đóng gói. Một số deliverable là đường dẫn dự kiến trong workbook, chưa có trên cây hiện tại. |

### **7. Member Ownership & Contribution Matrix (Workbook Reference)**

Workbook `Phan_Cong_Dong_Gop` hiện phân bổ 30 tasks như sau. Tỷ lệ 20% là tỷ lệ đóng góp được ghi trong workbook; họ tên pháp lý/MSSV vẫn là placeholder cho tới khi nhóm xác nhận.

| Thành viên / alias | Vai trò và trục dọc | Trục ngang | Tasks sở hữu (workbook) | Tỷ lệ trong workbook |
| :--- | :--- | :--- | :--- | :---: |
| `MEMBER_01` — `Duy-DS` | Trưởng nhóm/Tech Lead; quản lý tiến độ, OOP PyTorch (`src/`, `configs/`, `AGENTS.md`), review PR/đóng gói | Nhóm 1 — MLP | `T0101`, `T0106`, `T0201`, `T0202`, `T0212`, `T0306`, `T0406` (7) | 20% |
| `MEMBER_02` — alias trong sheet `Member_02` | Pipeline dữ liệu, NCHW và augmentation | Nhóm 2 — CNN+MLP | `T0102`, `T0203`, `T0204`, `T0205`, `T0304`, `T0404` (6) | 20% |
| `MEMBER_03` — alias trong sheet `Member_03` | PCA/Confusion Matrix, visualization và error analysis RQ1–RQ2 | Nhóm 3 — CNN+LSTM | `T0103`, `T0206`, `T0207`, `T0303`, `T0403` (5) | 20% |
| `MEMBER_04` — alias trong sheet `Member_04` | GPU, TensorBoard/CSV, Drive checkpoints và ablation | Nhóm 4 — Training Utilities | `T0104`, `T0208`, `T0209`, `T0301`, `T0302`, `T0405` (6) | 20% |
| `MEMBER_05` — alias trong sheet `Member_05` | Demo app, chủ trì báo cáo, slide/poster | Nhóm 5 — SegNet/Object Detection | `T0105`, `T0210`, `T0211`, `T0305`, `T0401`, `T0402` (6) | 20% |

Tổng task ownership là 30; tổng tỷ lệ workbook là 100%. Các cột hoàn thành/trạng thái trong workbook là dữ liệu vận hành có thể thay đổi; không coi chúng là nghiệm thu nếu thiếu report/DoD hiện hành. `MEMBER_01_NAME`/`EMAIL` được đọc từ `BTL03/.env` khi cần author/committer; không chép email riêng vào tài liệu công khai.

---

## **III. WORKFLOW, GIT & EVIDENCE INTEGRITY (QUY TRÌNH & BẰNG CHỨNG NGHIỆM THU)**

### **8. Git Branching, Commit & Human Author Conventions (Quy chuẩn Nhánh, Commit & Tác giả)**
- Repository chấp nhận đồng thời hai kiểu branch đang dùng trong workbook và Git:
  1. **Theo Task ID:** `[feat|fix|exp|docs|refactor|test]/BTL03_Txxxx_<mo-ta-ngan>`.
     Ví dụ: `feat/BTL03_T0102_data-pipeline`, `test/BTL03_T0212_wp02-review-smoke`, `docs/BTL03_T0402_thesis-report-writing`.
  2. **Theo module/Work Package:** `feature/<ten-nhom-module>`.
     Ví dụ: `feature/data-pipeline`, `feature/group1-mlp`, `feature/group2-cnn`, `feature/group3-cnn-lstm`, `feature/group4-utilities`, `feature/group5-segnet-fcn`, `feature/experiments`, `feature/visualization`, `feature/demo-app`, `feature/docs-final`.
- Branch phải gắn với task/module thật và không dùng để ngụ ý task đã nghiệm thu.

- **Quy chuẩn Commit Message & Tác giả Con người (BẮT BUỘC):**
  - Định dạng Commit Message: `[BTL03_<Task ID>] <type>(<scope>): <mô tả ngắn gọn bằng tiếng Việt hoặc tiếng Anh>`.
    - *Ví dụ:* `[BTL03_T0301] feat(models): chuyển đổi kiến trúc MNIST_CNN_LSTM sang PyTorch thay thế TimeDistributed`
  - **Kiểm soát tác giả Git Commit:**
    - Tác giả commit (`Author` & `Committer`) **BẮT BUỘC** là tên thành viên thật (ví dụ: `Duy-DS`).
    - **CẤM TUYỆT ĐỐI** để tên tác giả commit hoặc chuỗi `Co-authored-by:` là `Codex`, `GitHub Copilot`, `Cursor`, `Claude`, `Antigravity` hay bất kỳ AI Agent nào. Nếu phát hiện `git config user.name` đang mang tên công cụ tự động, Agent phải dừng thao tác commit và yêu cầu người dùng cấu hình lại tên thành viên.

-  **Cơ chế Tra cứu Định danh Tự động qua `.env` (`.env` Identity Resolution):**
    -   Thông tin ánh xạ giữa mã thành viên (`MEMBER_01` đến `MEMBER_05`) với `NAME` (GitHub Username) và `EMAIL` (GitHub Email) được lưu trữ cục bộ tại tệp `.env` ở thư mục gốc dự án (tham chiếu cấu trúc từ `.env.example`).
    -   **Quy trình bắt buộc trước khi `git commit`:**
        1. Agent phải xác định thành viên phụ trách Task hiện tại dựa trên bảng phân công WBS (hoặc chỉ định trực tiếp trong prompt).
        2. Đọc tệp `.env` để trích xuất cặp biến `MEMBER_XX_NAME` và `MEMBER_XX_EMAIL` tương ứng.
        3. Thực thi lệnh commit với cờ `--author` tường minh để không làm sai lệch cấu hình global của máy:
           ```bash
           git commit --author="<MEMBER_XX_NAME> <<MEMBER_XX_EMAIL>>" -m "[BTL03_<Task ID>] <type>(<scope>): <mô tả>"
           ```
        4. Nếu tệp `.env` chưa tồn tại hoặc thiếu thông tin của thành viên phụ trách, Agent **bắt buộc phải dừng thao tác commit** và yêu cầu người dùng cập nhật tệp `.env` trước khi tiếp tục.

### **9. Mandatory Evidence Logging & File Naming Rules (Hệ thống Bằng chứng Bắt buộc)**
Dự án áp dụng cơ chế kiểm soát **"Evidence-First & Template-Locked"** nghiêm ngặt. Evidence thực thi, sửa lỗi, nhật ký và thực nghiệm được ghi trong bốn thư mục `docs/reports/`, `docs/fix_logs/`, `docs/daily_logs/`, `docs/experiences/`; charter, presentation và thesis có thư mục riêng bên dưới `docs/`. Khi tạo evidence file, Agent **bắt buộc phải giữ nguyên cấu trúc template gốc tương ứng**, ghi thành viên phụ trách thật (không ghi AI Agent) và tuân thủ quy tắc đặt tên:

| Thư mục lưu trữ | Định dạng tên file bắt buộc & Ví dụ | File Template gốc bắt buộc (Read-Only) | Quy tắc điền thông tin & Đề mục cốt lõi |
| :--- | :--- | :--- | :--- |
| **`docs/fix_logs/`** | **Định dạng:**<br>`[YYMMDD]_Txxxx_<ten_loi>.md` hoặc `[YYMMDD]_[HHMM]_<ten_van_de>.md`.<br>*Ví dụ:* `261124_T0204_wiki-rotation.md`. | `docs/fix_logs/TEMPLATE-bug-fix-report.md` | **Tiêu đề:** `# 🐛 Báo Cáo Sửa Lỗi: [Mã lỗi hoặc Tiêu đề lỗi ngắn]`<br>Ghi **Người xử lý** là thành viên thật; mô tả triệu chứng/nguyên nhân, phương án sửa và bằng chứng xác minh. Không ghi kiểm thử chưa chạy là PASS. |
| **`docs/daily_logs/`** | **Định dạng:**<br>`[YYMMDD]_[HHMM]_<alias>.md`.<br>*Ví dụ:* `261124_1800_Duy-DS.md`. | `docs/daily_logs/TEMPLATE-daily-log.md` | Ghi ngày, thành viên thực hiện, task Done, blocker/Todo và bối cảnh phiên sau; chỉ đánh dấu Done khi có evidence. |
| **`docs/reports/`** | **Ưu tiên theo Sheets:**<br>`[YYMMDD]_Txxxx_<ten_task>.md`.<br>**Tên cũ có timestamp:** `[Task ID]_[YYMMDD]_[HHMM].md` vẫn được chấp nhận khi template/record hiện hành dùng định dạng này.<br>*Ví dụ:* `261126_T0102_data_pipeline_report.md`; `261210_T0212_pr_review_smoke_test.md`. | `docs/reports/TEMPLATE-execution-report.md` | Giữ các mục metadata, acceptance matrix, guardrails, findings, metrics và issues của template; ghi author/executor là người thật. Đường dẫn trong cột evidence phải trỏ tới file thực tế hoặc được đánh dấu là deliverable dự kiến, không khai báo giả file chưa có. |
| **`docs/experiences/`** | **Định dạng:**<br>`[ex ID]_[YYMMDD]_[HHMM].md`<br><br>*Ví dụ:*<br>`docs/experiences/EXP01_261122_2100.md` | `docs/experiences/experiences_template.md` | **Tiêu đề:** `# [EXPERIMENT] <Mã Thí Nghiệm>: <Tiêu đề tóm tắt>`<br>**Metadata Header:** `Mã thực nghiệm (EXP_[ID])`, `Tên file lưu vết`, `Thời gian`, `Người thực hiện: [Tên thành viên, VD: Duy-DS]` *(Cấm ghi AI Agent)*, `Work Package`, `Task ID (BTL03_<Task_ID>)`, `Nhánh Git`, `Trạng thái`.<br>**1. Mục tiêu & Giả thuyết nghiên cứu:** Liên kết `RQ1`–`RQ4` / Đối chiếu Keras vs PyTorch / Ablation.<br>**2. Thiết lập thực nghiệm & Tính tái lập:** Kiến trúc, kích thước tensor, `Seed=42`, siêu tham số, đường dẫn TensorBoard & CSV Log.<br>**3. Lệnh thực thi tái hiện (Reproduction Command):** Lệnh CLI đầy đủ.<br>**4. Bảng kết quả định lượng (Quantitative Results):** So sánh Keras Baseline vs PyTorch & các biến thể.<br>**5. Bằng chứng thực thi & Terminal Logs:** Log terminal & đường dẫn biểu đồ TensorBoard/CSV.<br>**6. Phân tích & Đánh giá kết quả (Discussion & Findings):** Giải thích nguyên nhân thành công/thất bại, hiện tượng Overfitting, chi phí tính toán.<br>**7. Kết luận & Hành động tiếp theo:** Checkbox nghiệm thu. |
| **`docs/project_charter/`** | Workbook, đề bài, `README_BTL03.md`, `QUY_DINH_SAN_PHAM.txt`, `Yêu cầu.txt`, `examples/`; workbook dự kiến tham chiếu `AGENTS.md`/`Project_Charter_BTL03.md` | Tài liệu đặc tả và nguồn điều hành. Hai file Markdown nêu trên chưa có ở cây hiện tại; chỉ ghi như hiện hữu sau khi được tạo/đưa vào repo. |
| **`docs/presentation/`** | Slide, poster, kịch bản; workbook dự kiến có `Slide_Poster_BTL03.pdf` | Thư mục/file chưa có trên cây hiện tại; là deliverable WP04, không phải artifact đã nghiệm thu. |
| **`docs/thesis/`** | `AGENT_THESIS_RULES.md`, `templates/`, `source_materials/`, `md_drafts/`, `tex_source/`, `README.md`; workbook dự kiến `BaoCao_TongKet_BTL03.pdf` và `BienBan_NghiemThu_BTL03.md` | Tuân thủ quy tắc riêng tại `docs/thesis/AGENT_THESIS_RULES.md`; PDF/biên bản workbook chưa có ở cây hiện tại. |
| **`docs/` (root guide)** | `HUONG_DAN_GIAI_THICH_CODE.md` | Tài liệu chuyển giao kiến thức; cấu trúc thực tế hiện nằm tại `docs/HUONG_DAN_GIAI_THICH_CODE.md`. |

**Quy tắc riêng cho `docs/daily_logs/`:** mỗi phiên ghi một file `[YYMMDD]_[HHMM]_<alias>.md`; dùng alias thành viên thật, giữ nguyên đề mục trong template, và tách việc đã làm khỏi Todo/blocker. Không đặt file nhật ký phiên vào `reports/` hoặc `experiences/` để né quy tắc tên riêng.
### **10. Repository Structure, Access Permissions & `.gitignore` Policy**

**Cây repository tham chiếu theo cấu trúc hiện hành (chi tiết nội dung có thể tăng theo task):**

```text
BTL03/
├── app/                         # Vỏ thư mục demo; hiện có .gitkeep
├── configs/                     # YAML model configs và paths
├── data/                        # README + raw/processed .gitkeep; không commit dữ liệu nặng
├── docs/
│   ├── daily_logs/              # Nhật ký phiên + template
│   ├── experiences/             # Evidence thực nghiệm + template
│   ├── fix_logs/                # Evidence sửa lỗi + template
│   ├── project_charter/         # Workbook, tài liệu đặc tả và examples/
│   ├── reports/                 # Báo cáo task + template
│   ├── thesis/                  # AGENT_THESIS_RULES.md, templates/, source_materials/, md_drafts/, tex_source/
│   └── HUONG_DAN_GIAI_THICH_CODE.md
├── experiments/                 # Logs và helpers
│   ├── colab/
│   ├── csv_logs/                # Baseline, master và epoch CSV
│   └── tensorboard/             # Event files runtime, gitignored
├── legacy_keras/                # Notebook Keras tham chiếu, READ-ONLY
├── notebooks/                   # Adapter song song
│   ├── colab/
│   └── kaggle/
├── outputs/                     # Checkpoints/figures runtime
├── scripts/                     # Downloaders, runners và smoke harness
├── src/
│   ├── datasets/
│   ├── engine/                  # trainer.py và callbacks.py
│   ├── models/
│   ├── evaluate.py
│   ├── inference.py
│   ├── logger.py                # ExperimentLogger, CheckpointManager
│   ├── train.py
│   └── utils.py                 # seed_everything(42)
├── tests/
├── .env                         # Local identity; gitignored
├── AGENTS.md
├── README.md
└── requirements.txt
```

Workbook có một số deliverable/path dự kiến nhưng chưa có trong cây đã rà soát: `docs/presentation/`, `docs/project_charter/AGENTS.md`, `docs/project_charter/Project_Charter_BTL03.md`, PDF tổng kết và biên bản nghiệm thu. Không tạo placeholder rồi mô tả là đã nghiệm thu. Source thực tế có `src/engine/callbacks.py`; không giả định có `src/callbacks.py` hoặc `app/demo.py`.

**`.gitignore`:** `BTL03/.gitignore` hiện hành là nguồn có thẩm quyền. File đó bỏ qua `.env`/secrets, môi trường/cache, dữ liệu raw/processed, outputs, weights và TensorBoard events; có ngoại lệ cho `.gitkeep`, `data/README.md` và CSV logs được phép. Trước khi stage/commit đọc policy thực tế; không thay bằng snippet khác và không commit dữ liệu/artifacts nặng.
### **11. CSV Logs, Google Drive/Sheets & Parity Acceptance**
`experiments/csv_logs/master_experiment_runs.csv` phải giữ đúng schema 18 cột dưới đây, cùng thứ tự với hàng tiêu đề (row 2) của tab `Raw_TensorBoard_CSV_Logs`. Đồng bộ bằng **Paste Values Only** từ row 3; không sửa header hoặc tự điền metric thiếu.

```text
Timestamp, Run_ID, Model_Group, Model_Name, Exp_Type, Seed, Optimizer, LR, Batch_Size, Epochs, Best_Epoch, Train_Loss, Val_Loss, Train_Metric, Val_Metric, Test_Metric, Train_Time_Sec, Member_Alias
```

- CSV master, baseline CSV và epoch CSV gọn được lưu trong `experiments/csv_logs/`; chúng là evidence có thể track khi tuân thủ `.gitignore`. Cây hiện có 37 `*_epochs.csv`; số lượng có thể thay đổi khi có run mới, vì vậy “49+” trong yêu cầu/lộ trình không phải số hiện hữu cố định.
- Artifacts nặng hoặc tái sinh được — weights `.pt/.pth`, feature `.npz`, dữ liệu và TensorBoard event files — không commit lên Git. Lưu trong Drive chung theo `configs/paths.yaml`: `/content/drive/MyDrive/BTL03` khi Drive đã mount; các nhánh dự kiến gồm `data/raw/`, `checkpoints/{01_mlp,02_cnn_mlp,03_cnn_lstm,04_training_utilities,05_others}/`, TensorBoard runs và artifacts được nhóm duyệt. `CheckpointManager` không tự mount Drive; mirror chỉ xảy ra khi mount root tồn tại.
- `configs/kaggle_paths.yaml` quy định input read-only `/kaggle/input` và artifacts tạm ở `/kaggle/working/`; trước khi chia sẻ artifacts, chuyển/đóng gói theo workflow được nhóm duyệt sang Drive. Không ghi credential/token vào Git.
- Tab `Doi_Chieu_Keras_PyTorch` là bảng theo dõi parity/DoD, không phải bằng chứng tự thân. Tiêu chí kế hoạch của workbook là khớp kiến trúc và parameter count 100%, cùng delta metric trong ngưỡng cho phép (README nêu ±1.50% Accuracy hoặc ±0.02 MAE); phải ghi đúng metric/split, runtime, Run_ID và evidence. Chỉ đánh dấu `ĐẠT` khi có run/evidence đối chiếu tương ứng và DoD được owner/reviewer chấp nhận. Ngưỡng không phải kết quả đã đạt.
- Nếu workbook, README, baseline CSV, source và task report khác nhau, không tự chọn số thuận tiện: giữ provenance, ghi `CẦN ĐỐI CHIẾU`, ưu tiên execution report/CSV/source theo quy tắc evidence, và yêu cầu owner đồng bộ sheet. Parameter-count equality không chứng minh output/accuracy parity.

### **12. Standard Execution Commands & Completion Checklist**

**Lệnh tham khảo cho môi trường được nhóm phê duyệt (không chạy nếu task/guardrail cấm cài đặt hoặc thực thi):**
```bash
# 1. Kích hoạt môi trường ảo & Cài đặt thư viện
source .venv/bin/activate  # Linux/macOS (hoặc .venv\Scripts\activate trên Windows)
pip install -r requirements.txt

# 2. Kiểm tra định danh Git Author trước khi làm việc (Bắt buộc là tên người thật)
git config user.name

# 3. Chạy Unit Tests kiểm tra kiến trúc, chuyển trục NCHW & đếm số tham số khớp Keras
pytest tests/

# 4. Tra cứu CLI/config hiện hành; dùng đúng runner/config được report xác nhận
python -m src.train --help

# 5. Mở giao diện Web TensorBoard để theo dõi biểu đồ Loss/Accuracy trực quan
tensorboard --logdir experiments/tensorboard --port 6006

# 6. Tra cứu giao diện đánh giá trước khi dùng config/checkpoint thực tế
python -m src.evaluate --help

# 7. Chỉ chạy demo sau khi entry point được tạo và nghiệm thu; hiện app/ chỉ có .gitkeep.
```

**Checklist bắt buộc của Agent trước khi báo cáo hoàn thành bất kỳ yêu cầu nào:**
1. [ ] **Kiểm tra Định danh Con người (Anti-AI Attribution):** Đã xác định owner theo WBS/prompt, đọc cặp `MEMBER_XX_NAME`/`MEMBER_XX_EMAIL` từ `.env`, dùng identity người thật cho Author/Committer và không thêm AI `Co-authored-by` chưa? Git config global có thể khác; không dùng nó thay cho `.env` identity.
2. [ ] **Kiểm tra Branch:** Đang dùng branch theo task `[feat|fix|exp|docs|refactor|test]/BTL03_Txxxx_<mo-ta>` hoặc theo module `feature/<ten-nhom-module>`; không commit trực tiếp lên `main` chưa?
3. [ ] **Kiểm tra PyTorch, OOP & Tiếng Việt:** Mã nguồn trong `src/` viết 100% bằng PyTorch (không dùng Keras/TF), tuân thủ OOP kế thừa `BaseMigratedModel` và có đầy đủ Docstring + Comment bằng Tiếng Việt chưa?
4. [ ] **Kiểm tra Chuyển trục & Reproducibility:** Đã cố định `Seed = 42`, xử lý đúng định dạng tensor `NCHW` (hoặc `TimeDistributed` reshape) và đếm số tham số khớp với bản gốc Keras chưa?
5. [ ] **Kiểm tra TensorBoard + CSV Logs & `.gitignore`:** Đã cấu hình xuất đồng thời TensorBoard events và CSV Logs chưa? Đã đảm bảo không stage nhầm file `.pth`, `data/`, `outputs/` hoặc `experiments/tensorboard/` vào Git chưa?
6. [ ] **Khởi tạo Bằng chứng Chuẩn Template (`docs/`):** Đã sao chép nguyên vẹn cấu trúc từ 4 file template tương ứng trong `docs/`, điền đủ 100% các đề mục, bảng nghiệm thu, checkbox mà không tự ý lược bỏ, và đặt tên file đúng định dạng `[YYMMDD]_[HHMM]` chưa?
7. [ ] **Giải thích Code & Chuyển giao Tri thức (`HUONG_DAN_GIAI_THICH_CODE.md`):** Đã giải thích chi tiết bản chất kỹ thuật (luồng tensor shape, sự khác biệt Keras vs PyTorch, số tham số, siêu tham số) trong hội thoại và cập nhật bổ sung vào tài liệu `docs/HUONG_DAN_GIAI_THICH_CODE.md` kèm bộ câu hỏi/trả lời vấn đáp trước Giảng viên chưa?