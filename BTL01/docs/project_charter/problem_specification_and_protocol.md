# Đặc tả bài toán và giao thức thực nghiệm Gaussian 2D–MLP

**Task:** T0102 — **Phiên bản:** 1.0 — **Ngày:** 2026-10-25.

## 1. Phạm vi, nguồn chuẩn và trạng thái triển khai

Tài liệu kế thừa T0101, README và Điều 1–7 của `AGENTS.md`.
`configs/baseline.yaml` là nguồn chuẩn cho thông số dùng chung;
`configs/exp_specs.yaml` chứa lưới A/B/C và chỉ tham chiếu baseline, không sao chép nó.
Đây là hợp đồng thiết kế cho các task triển khai sau, **không phải báo cáo đã huấn luyện**.
Chưa sinh CSV dữ liệu, checkpoint, biểu đồ hay số liệu sai số mô hình tại T0102.
Mọi đường dẫn trong cấu hình được giải từ gốc dự án bằng `pathlib.Path`.

Hai YAML được đọc riêng bằng `yaml.safe_load`; không dùng tag `!include`.
Runner phải kiểm tra `schema_version`, liên kết hai chiều giữa hai file, các giá trị
được phép và tính nhất quán `hidden_depth == len(hidden_dims)` trước khi chạy.
Mỗi run tạo bản cấu hình đã resolve độc lập: chỉ thay đúng biến khảo sát; không sửa
baseline tại chỗ hay mang trạng thái optimizer/checkpoint từ run trước sang run sau.

## 2. Phát biểu bài toán và cơ sở toán học

### 2.1. Hàm đích

Với $n=2$, $\mathbf{x}=(x_1,x_2)^T$, mật độ Gaussian là

$$
f_{\mathbf X}(\mathbf x)=
\frac{1}{(2\pi)^{n/2}\sqrt{\det(\mathbf C)}}
\exp\left[-\frac12(\mathbf x-\boldsymbol\mu)^T
\mathbf C^{-1}(\mathbf x-\boldsymbol\mu)\right].
$$

$$
\boldsymbol\mu=\begin{bmatrix}0\\0\end{bmatrix},\qquad
\mathbf C=\begin{bmatrix}0.3&-0.5\\-0.5&2.0\end{bmatrix},\qquad
\mathcal D=[-2,2]\times[-4,4].
$$

$\boldsymbol\mu$ là tâm Gaussian; đường chéo $\mathbf C$ là phương sai từng
tọa độ, phần tử ngoài đường chéo là hiệp phương sai âm. Định thức đo hệ số
thể tích trong chuẩn hóa mật độ; nghịch đảo xác định khoảng cách Mahalanobis.

$$
\det(\mathbf C)=0.3\cdot2.0-(-0.5)^2=0.35,\qquad
\mathbf C^{-1}=\frac1{0.35}\begin{bmatrix}2.0&0.5\\0.5&0.3\end{bmatrix}.
$$

Ma trận đối xứng có hai định thức con chính đầu $0.3>0$ và $0.35>0$,
nên xác định dương theo tiêu chuẩn Sylvester, do đó khả nghịch. **Chỉ riêng
điều kiện định thức dương không đủ để kết luận xác định dương tổng quát.**
Hằng số chuẩn hóa $\alpha=1/(2\pi\sqrt{0.35})\approx0.269021$.
Vì $\boldsymbol\mu=0$, có thể kiểm tra nhãn bằng biểu thức tương đương:

$$
y=\alpha\exp\left[-\frac{2x_1^2+x_1x_2+0.3x_2^2}{0.7}\right].
$$

Gaussian được chuẩn hóa trên $\mathbb R^2$; $\mathcal D$ chỉ là miền khảo sát.
Không tái chuẩn hóa thành mật độ Gaussian bị cắt trên hình chữ nhật.
**Đầu vào được lấy mẫu đều, không lấy mẫu từ chính phân phối Gaussian đích.**

### 2.2. Hồi quy bằng MLP và liên hệ UAT

Học ánh xạ $\hat y=f_{\boldsymbol\theta}(x_1,x_2)$ từ nhãn giải tích không nhiễu.
Với $C$ tầng ẩn, $\mathbf h^{(0)}=\mathbf x$:

$$
\mathbf h^{(l)}=\tanh(\mathbf W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)}),
\quad l=1,\ldots,C,\qquad
\hat y=\mathbf W^{(C+1)}\mathbf h^{(C)}+b^{(C+1)}.
$$

Baseline $2\to10\to1$ có bias, tổng cộng $2\cdot10+10+10+1=41$ tham số.
Đầu ra tuyến tính không bảo đảm không âm hay tích phân bằng 1; đây là hồi quy
giá trị mật độ, không phải mô hình mật độ có ràng buộc. Không clip dự đoán trước
khi tính metric, vì clipping sẽ thay đổi giao thức đánh giá.

Theo kết quả UAT cho mạng một tầng ẩn với kích hoạt sigmoid liên tục [1],
có thể xấp xỉ đều hàm liên tục trên compact với độ rộng đủ lớn. Tanh liên hệ
affine với logistic sigmoid nên có cùng cơ sở biểu diễn. Gaussian liên tục trên
$\mathcal D$ thỏa điều kiện hàm đích. Định lý chỉ bảo đảm **tồn tại** biểu diễn;
không bảo đảm 10 nơ-ron đủ, Adam tìm được nghiệm tối ưu, hay độ sâu tăng luôn tốt hơn.

## 3. Dữ liệu và giao thức chống rò rỉ

### 3.1. Sinh và lưu trữ

Mỗi điểm có $x_1\sim\mathcal U(-2,2)$ và $x_2\sim\mathcal U(-4,4)$ độc lập;
$y=f_{\mathbf X}(\mathbf x)$. Tính nhãn bằng `float64`, lưu CSV UTF-8 có header
`x1,x2,y`, không có cột index, định dạng `%.17g`. Dataset chuyển sang `float32`:
đầu vào shape `(N, 2)`, nhãn và dự đoán shape `(N, 1)` để tránh broadcasting sai.
Baseline không chuẩn hóa input/target và không thêm nhiễu.

| Tập | Số mẫu | File | Quyền sử dụng |
| :--- | ---: | :--- | :--- |
| Train đầy đủ | 2000 | `data/train_full.csv` | Gradient và tập con A/B/C |
| Validation | 200 | `data/val.csv` | Early Stopping, checkpoint, lựa chọn cấu hình |
| Test | 100 | `data/test.csv` | Chỉ đánh giá cuối của suite đã khóa |

Tạo `SeedSequence(42).spawn(3)` theo thứ tự train/val/test và dùng
`Generator(PCG64(child_sequence))` cho mỗi tập. Ba luồng không dùng lại mẫu
hay khởi tạo lại cùng một seed riêng lẻ cho cả ba tập. Kiểm tra số hàng, hữu hạn,
miền tọa độ, nhãn giải tích và giao rỗng tọa độ giữa các tập; nếu có trùng thì
tiếp tục lấy từ luồng tương ứng, không sao chép hay dịch điểm.
CSV được sinh một lần, ghi SHA-256 trong manifest và dùng lại xuyên suốt suite.
Kiểm tra tính toàn vẹn dữ liệu không phải đánh giá mô hình trên test.

Tạo một hoán vị train bằng `Generator(PCG64(42)).permutation(2000)`.
Tập $A$ là prefix $A$ phần tử của hoán vị này, nên các tập A lồng nhau;
B/C dùng đúng prefix 1000. Hoán vị chọn mẫu tách biệt với RNG shuffle mini-batch.
Không lấy validation/test từ 2000 mẫu train và không trộn lại split theo training seed.

### 3.2. Hai pha đánh giá bất di bất dịch

1. **Pha phát triển:** khóa lưới A/B/C và các seed trước khi chạy; huấn luyện chỉ
   trên train, giám sát validation. Trainer không nhận test loader. Nếu cần chọn
   cấu hình đại diện, dùng mean validation MSE qua ba seed; hòa thì ưu tiên ít
   tham số hơn, rồi thứ tự cấu hình trong YAML. Không chọn seed tốt nhất.
2. **Pha cuối:** sau khi hoàn tất mọi quyết định dựa trên validation, khóa toàn bộ
   checkpoint và cấu hình; Evaluator đọc test để đánh giá mỗi checkpoint đã đăng ký
   đúng một lượt. Có thể nạp CSV một lần và đánh giá cả suite; không hiểu “một lần”
   là chỉ được đánh giá một mô hình trong A/B/C. Không sửa grid hay chọn checkpoint
   theo test MSE. Mọi thay đổi sau khi xem test phải gắn nhãn thăm dò, không coi test
   cũ là holdout chưa quan sát cho một kết luận xác nhận mới.

Không đưa test loss vào lịch sử epoch, Early Stopping hoặc quyết định optimizer.
Nếu sau này thêm normalization, chỉ fit trên train của run, rồi áp dụng nguyên trạng
cho val/test; đó là thay đổi protocol có version mới, không phải baseline hiện tại.

## 4. Baseline khóa và điều kiện so sánh

### 4.1. Bảng cấu hình và cơ sở lựa chọn

| Nhóm / khóa | Giá trị | Lý giải hoặc quy tắc |
| :--- | :--- | :--- |
| `reproducibility.seed`, `deterministic` | `42`, `true` | Khóa ngẫu nhiên và yêu cầu thuật toán xác định |
| RNG dữ liệu / subset | PCG64, seed 42; ba split streams | Không làm biến đổi holdout khi đổi training seed |
| `hardware.device` | `auto`: CUDA → MPS → CPU | Hỗ trợ accelerator và CPU; toàn suite dùng cùng thiết bị |
| `dtype`, `num_workers` | float32, 0 | Đơn giản hóa luồng dữ liệu và tái lập |
| Miền, mean, covariance | $[-2,2]\times[-4,4]$, $[0,0]$, $[[0.3,-0.5],[-0.5,2.0]]$ | Đúng bài toán Gaussian 2D |
| Input/output | 2 / 1 | Hồi quy một mật độ từ hai tọa độ |
| Train đầy đủ / val / test | 2000 / 200 / 100 | Validation tách biệt, test độc lập |
| Train mặc định B/C | 1000 | Kế thừa thiết kế T0101/README |
| `hidden_depth`, `hidden_dims` | 1, `[10]` | Mạng nhỏ làm mốc kiểm soát; không khẳng định tối ưu |
| `activation` | tanh | Trơn và zero-centered, phù hợp giả thuyết xấp xỉ mặt cong trơn; không bảo đảm hơn ReLU |
| Các activation được hỗ trợ | tanh, relu, sigmoid, leaky_relu | Chỉ tanh thuộc suite baseline; LeakyReLU slope 0.01 |
| `weight_init` | xavier_normal | Khởi tạo theo fan-in/fan-out nhằm kiểm soát độ lớn tín hiệu [2] |
| Gain hidden / output; bias | $5/3$ / 1; bias bằng 0 | Cố định đầy đủ quy tắc khởi tạo tanh và đầu ra tuyến tính |
| `batch_size`, `drop_last` | 32, false | Mini-batch nhỏ; không bỏ các mẫu cuối |
| `epochs` | tối đa 1000 | Cùng giới hạn epoch, không bảo đảm mọi run hội tụ |
| `optimizer` | adam | Moment bậc một/hai thích nghi [3]; không tuyên bố luôn nhanh hơn SGD |
| `learning_rate`, `weight_decay` | 0.005, 0.0 | Giá trị tiên nghiệm đã chốt; không được suy diễn là tối ưu |
| Adam betas / eps / amsgrad | `[0.9,0.999]` / `1.0e-8` / false | Khóa hành vi optimizer |
| SGD tùy chọn | momentum 0, nesterov false | Ngoài suite baseline, cùng LR nếu không có protocol riêng |
| `loss_function`, reduction | mse, mean | Tối ưu sai số bình phương trên mật độ gốc |
| Scheduler / clipping | none / false | Không thêm biến can thiệp trong suite |
| Shuffle train / validation | true / false | Shuffle train theo seed run, validation ổn định |
| Early Stopping | enabled, patience 50, min_delta $10^{-6}$, val_loss, min | Hạn chế kéo dài khi validation không tiến bộ đáng kể |
| Checkpoint | minimum val_loss, restore_best | Không chọn epoch cuối hay dùng test để chọn mô hình |
| Multi-seed | `[42,100,2026]` | Định lượng biến thiên do khởi tạo và shuffle trên dữ liệu cố định |
| Metrics / aggregate | MSE, RMSE, MAE; mean ± sample std | Giữ kết quả từng run, không chỉ báo cáo seed tốt nhất |
| Timing | perf_counter, đồng bộ accelerator | Đo thời gian thực gồm train/val/checkpoint writes |
| Outputs | `data/`, `outputs/checkpoints/`, `outputs/tables/`, `outputs/figures/` | Đường dẫn tương đối từ project root |
| Hình ảnh | PNG 300 DPI; grid $101\times101$ | Grid chỉ để vẽ, không thay test hoặc chọn model |

Với activation ngoài tanh, gain khởi tạo phải được resolve rõ theo activation trong
manifest (đầu ra vẫn gain 1); các run này không thuộc so sánh A/B/C đã khóa.

### 4.2. Seed, thiết bị và giới hạn tái lập

Trước mỗi run, seed `random`, `numpy`, `torch.manual_seed` và
`torch.cuda.manual_seed_all` bằng seed run. DataLoader dùng `torch.Generator`
riêng cùng seed để RNG khởi tạo mạng không làm thay đổi thứ tự mini-batch.
RNG dữ liệu/subset vẫn cố định ở 42. Thiết lập `CUBLAS_WORKSPACE_CONFIG=:4096:8`
trước khởi tạo CUDA; tắt cuDNN benchmark, bật cuDNN deterministic và
`torch.use_deterministic_algorithms(True, warn_only=False)`.
Nếu backend không hỗ trợ phép toán xác định, dừng và ghi lỗi; có thể chạy lại
**toàn suite** trên CPU, không âm thầm bỏ deterministic hoặc đổi thiết bị giữa run.
Seed không bảo đảm kết quả bitwise giữa phiên bản, hệ điều hành và phần cứng khác nhau.
Ghi phiên bản Python/thư viện, thiết bị, cấu hình đã resolve và hash dữ liệu trong manifest.

### 4.3. Hợp đồng Trainer và chọn checkpoint

Vòng lặp PyTorch thủ công: `model.train()`; forward → loss →
`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`.
Validation: `model.eval()` và `with torch.no_grad():`, không cập nhật gradient.
Loss epoch là tổng sai số bình phương chia số mẫu thực tế, không lấy trung bình
không trọng số của các batch mean khi batch cuối nhỏ hơn.

Checkpoint được lưu mỗi khi `val_loss` hữu hạn **nhỏ hơn tuyệt đối** minimum
trước đó; hòa giữ epoch sớm hơn. Early Stopping dùng một mốc tiến bộ riêng:
reset counter nếu `val_loss < reference_loss - min_delta`, ngược lại tăng counter;
dừng khi counter đạt 50. Epoch đầu hữu hạn khởi tạo cả hai mốc. Tách hai tiêu chí
để checkpoint luôn là minimum thực sự dù cải thiện nhỏ hơn $10^{-6}$.
NaN/Inf làm run thất bại và không được đưa vào thống kê như một run hợp lệ.
Nạp lại checkpoint tốt nhất trước khi tính train/val metrics cuối và test pha cuối.

## 5. Thiết kế A/B/C

### 5.1. A — Quy mô mẫu huấn luyện

- **Câu hỏi:** dữ liệu bổ sung cải thiện xấp xỉ và độ ổn định đến mức nào?
- **Biến thay đổi:** $A\in\{200,400,600,800,1000,1200,1400,1600,1800,2000\}$.
- **Giữ cố định:** MLP `[10]`, tanh, optimizer/hyperparameters baseline, val/test,
  hoán vị subset và ba seed. Có $10\times3=30$ run.
- **Giả thuyết:** sai số có thể giảm rồi bão hòa khi dữ liệu tăng; không bắt buộc
  đơn điệu ở từng seed. Cùng số epoch nhưng A lớn có nhiều optimizer step hơn;
  phải ghi số step và thời gian, không diễn giải đây là so sánh cùng compute budget.
- **Đầu ra:** `summary_exp_A.csv`, `runs_exp_A.csv`, hình
  `exp_A_error_vs_samples.png`, lịch sử loss và checkpoint riêng từng A/seed.

### 5.2. B — Độ rộng của một tầng ẩn

- **Câu hỏi:** độ rộng ảnh hưởng sai số và khoảng cách train–validation thế nào?
- **Biến thay đổi:** $B=1,\ldots,10$; mở rộng đăng ký trước $B\in\{16,32,64\}$,
  bật mặc định qua `include_extensions: true`. `hidden_dims=[B]`, $C=1$.
- **Giữ cố định:** đúng prefix 1000 train, val/test, ba seed, toàn bộ training baseline.
  Tổng $13\times3=39$ run; nếu tắt extension phải ghi rõ suite khác, chỉ có 30 run.
- **Giả thuyết:** mạng quá hẹp có thể underfit; mạng rộng tăng năng lực biểu diễn
  nhưng không nhất thiết overfit. Không gán trước ngưỡng B=3 hay B=10 là kết luận.
  Underfitting/overfitting được phân tích từ train/validation và loss curves;
  test chỉ bổ sung mô tả cuối, không dùng xác định ngưỡng để chọn B.
- **Đầu ra:** `summary_exp_B.csv`, `runs_exp_B.csv`, hình
  `exp_B_error_vs_neurons.png`, gap train–validation, loss history, checkpoint B/seed.

### 5.3. C — Độ sâu với độ rộng cố định

- **Câu hỏi:** thêm tầng thay đổi chất lượng xấp xỉ, hội tụ và chi phí ra sao?
- **Biến thay đổi:** $C\in\{1,2,3,4,5\}$, `hidden_dims=[10]*C`.
  C=1,2 là đối chứng; C=3,4,5 đáp ứng yêu cầu nhiều hơn hai tầng ẩn.
- **Giữ cố định:** B=10 mỗi tầng, prefix 1000 train, val/test, seed, activation,
  khởi tạo theo cùng quy tắc và training baseline. Tổng $5\times3=15$ run.
- **Công bằng:** đây là **fixed-width, không parameter-matched**. Số tham số
  $P(C)=41+110(C-1)$ lần lượt là 41, 151, 261, 371, 481. Chỉ thay độ sâu
  ở cấp cấu hình, nhưng năng lực và compute cũng tăng; không quy toàn bộ hiệu ứng
  cho độ sâu thuần túy. So sánh cùng ngân sách tham số phải là suite phụ riêng.
- **Giả thuyết:** mạng sâu có thể giảm sai số hoặc gặp khó tối ưu, không kỳ vọng
  thứ hạng đơn điệu. So sánh best epoch, epochs thực chạy, optimizer steps,
  thời gian tổng và thời gian/epoch, không chỉ final loss.
- **Đầu ra:** `summary_exp_C.csv`, `runs_exp_C.csv`, hình
  `exp_C_depth_convergence_time.png`, loss curves, số tham số, timing, checkpoint C/seed.

Toàn suite gồm **84 run đăng ký trước**. Những baseline trùng giữa A/B/C vẫn có
run ID riêng; không warm-start hay chọn lại seed theo kết quả.

## 6. Chỉ số và phép đo thời gian

Với $e_i=y_i-\hat y_i$ trên một split gồm N mẫu:

$$
\operatorname{MSE}=\frac1N\sum_{i=1}^{N}e_i^2,\qquad
\operatorname{RMSE}=\sqrt{\operatorname{MSE}},\qquad
\operatorname{MAE}=\frac1N\sum_{i=1}^{N}|e_i|.
$$

MSE phạt mạnh sai số lớn và là loss tối ưu. RMSE cùng đơn vị với mật độ đích;
MAE mô tả độ lệch tuyệt đối trung bình, ít nhấn mạnh các sai số lớn hơn MSE.
Tính ba metric riêng cho train/validation/test tại checkpoint tốt nhất;
RMSE được tính từ MSE **từng run**, rồi mới tổng hợp, không lấy căn mean MSE
để thay thế mean RMSE. Mọi metric trên nhãn và dự đoán nguyên gốc, không clipping.

Với $K=3$ seed và một metric $m$:

$$
\bar m=\frac1K\sum_{k=1}^{K}m_k,\qquad
s_m=\sqrt{\frac1{K-1}\sum_{k=1}^{K}(m_k-\bar m)^2}.
$$

Báo cáo $\bar m\pm s_m$ với `ddof=1`. Đây là độ biến thiên do training RNG
trên cùng dữ liệu, **không phải** khoảng tin cậy hay độ bất định do lấy mẫu test.
Không loại run xấu; run thất bại phải lưu trạng thái và nguyên nhân, chỉ tổng hợp
khi đủ ba run hợp lệ hoặc đánh dấu kết quả chưa đầy đủ.

Đồng bộ accelerator nếu có; lấy `t0=time.perf_counter()` ngay trước epoch đầu;
sau epoch cuối đồng bộ và lấy `t1`. $T_{\mathrm{train}}=t_1-t_0$ (giây).
Bao gồm forward/backward, validation, Early Stopping và checkpoint writes;
không gồm sinh dữ liệu, dựng model, nạp checkpoint cuối, evaluation hay vẽ hình.
Ghi số epoch thực và $T_{\mathrm{epoch}}=T_{\mathrm{train}}/E_{\mathrm{actual}}$.
Cùng máy, backend, dtype và điều kiện tải; ghi lần chạy đầu có thể chịu overhead
backend. Thời gian có I/O nên không được coi là benchmark kernel thuần.

## 7. Artifacts và truy vết

Mọi tên dưới đây được giải từ các template trong baseline:

| Artifact | Nơi lưu và nội dung tối thiểu |
| :--- | :--- |
| Checkpoint | `outputs/checkpoints/best_model_{experiment}_{factor_name}{factor_value}_seed{seed}.pt`; state_dict, best epoch, val_loss và resolved config |
| History | `outputs/tables/history_{experiment}_{factor_name}{factor_value}_seed{seed}.csv`; epoch, train_loss, val_loss, elapsed_seconds; tuyệt đối không có test_loss theo epoch |
| Per-run | `outputs/tables/runs_exp_A.csv` và B/C tương ứng; run_id, factor, seed, train_size, hidden_dims, parameter_count, best_epoch, epochs_run, optimizer_steps, train/val/test MSE/RMSE/MAE, training_time_seconds, status |
| Summary | `outputs/tables/summary_exp_A.csv` và B/C tương ứng; factor, n_runs, mean/std từng metric, timing và epochs |
| Manifest | `outputs/tables/manifest_{experiment}_{factor_name}{factor_value}_seed{seed}.json`; hash hai YAML, resolved config, hash CSV, phiên bản, thiết bị, seed, thời gian và checkpoint path |
| Figures | `outputs/figures/`; true vs predicted 3D, y vs y_hat, train/val loss, error theo A/B/C và timing C; PNG tối thiểu 300 DPI |

Ví dụ checkpoint A: `best_model_exp_A_train_size200_seed42.pt`.
Đồ thị tổng hợp có error bars mean ± std, title, nhãn trục (kèm đơn vị), legend;
3D ghi rõ x1, x2 và mật độ y. Đường cong chưa hội tụ không kéo dài giả sau Early
Stopping; hình đại diện chọn seed 42 và cấu hình bằng validation, không chọn hình
đẹp nhất theo test. Grid vẽ cố định $101\times101$ không được dùng đánh giá chọn model.

## 8. Phân công OOP cho các task kế tiếp

| Thành phần | Hợp đồng cần triển khai (chưa triển khai trong T0102) |
| :--- | :--- |
| GaussianDataGenerator — T0201 | Giữ mu/cov/domain; `compute_density(x: np.ndarray) -> np.ndarray`, shape `(N,2)` → `(N,1)`; kiểm tra SPD, sinh uniform bằng RNG độc lập |
| GaussianDataset — T0301 | Giữ features/targets float32; `__len__(self) -> int`, `__getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]`; đọc split và subset không rò rỉ |
| FunctionApproximatorMLP — T0302 | Giữ hidden_dims/activation/init; `forward(self, x: torch.Tensor) -> torch.Tensor`, shape `(batch_size,2)` → `(batch_size,1)`; nn.Linear có bias và output tuyến tính |
| Trainer — T0303 | Giữ model/optimizer/history/checkpoint; `train_epoch`, `validate`, `fit` nhận train/val, không nhận test; checkpoint theo val_loss |
| Evaluator — T0304 | `evaluate` nhận checkpoint và split cuối; MSE/RMSE/MAE, không cập nhật optimizer |
| Visualizer | Vẽ trên metrics/history đã lưu, bảo đảm title/labels/legend/300 DPI |
| ExperimentRunner — T0304 | Resolve cấu hình mỗi run, freeze data, train từ đầu, khóa pha validation rồi mới mở pha test và tổng hợp CSV |

Các chữ ký chi tiết cho Trainer/Evaluator/Runner được hoàn thiện cùng thiết kế
kiểu trả về ở task triển khai; mọi module/class/method thực tế phải có type hints,
Google/NumPy docstring và tensor shapes. Không dùng high-level training wrapper.

## 9. Tiêu chí nghiệm thu và giới hạn

- Hai YAML parse bằng `yaml.safe_load`; literal khoa học là số thực, seed và lưới đúng.
- Ma trận đối xứng, Sylvester dương, eigenvalues dương, Cholesky thành công,
  nghịch đảo nhân lại cho identity; định thức gần 0.35 theo dung sai số thực.
- README đồng bộ validation 200, B/C train 1000, tanh, MPS và fixed-width C.
- Đường dẫn cấu hình tương đối; không tạo dữ liệu/checkpoint/metric giả cho T0102.
- Báo cáo task và dev-log ghi đúng kết quả kiểm tra thực tế, nêu rõ PyTorch và
  pipeline chưa được kiểm thử ở task cấu hình này.
- `requirements.txt` hiện dùng cận dưới `>=`, **không phải lockfile**. T0102 không
  tùy ý đổi toàn bộ dependency; manifest phải ghi phiên bản thực, cần lock môi trường
  trước suite chính thức để củng cố tái lập xuyên máy.

## 10. Tài liệu tham khảo khoa học

[1] Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function*.
Mathematics of Control, Signals, and Systems, 2, 303–314. DOI: 10.1007/BF02551274.

[2] Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep
feedforward neural networks*. AISTATS, Proceedings of Machine Learning Research,
9, 249–256. Xavier normal trong cấu hình là lựa chọn triển khai cùng họ fan-scaled;
không đồng nhất mọi chi tiết với mọi thí nghiệm trong bài gốc.

[3] Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*.
ICLR; bản tiền ấn phẩm arXiv:1412.6980 (2014).