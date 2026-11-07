# KẾ HOẠCH TASK CHO AGENT LẬP TRÌNH

*Đi kèm `FINAL_De_xuat_nghien_cuu.md`. Mỗi task giao cho **một phiên agent**: dán **Ngữ cảnh chung** (Mục A) cộng **khối task** tương ứng. Làm theo thứ tự phụ thuộc ở Mục C; dừng ở các cổng G0–G3.*

---

## A. Ngữ cảnh chung (dán vào đầu MỌI phiên agent)

```
Dự án: nghiên cứu Block Relocation Problem (BRP), mở rộng L2BR (Shin et al., EJOR 2026).
Đề xuất đầy đủ: FINAL_De_xuat_nghien_cuu.md (đọc Mục 4 trước khi code).
Thư mục dự án: C:\Users\X1\Project\UnifiedLearingnFWforBRP (Windows, git repo).

Mã gốc L2BR: https://github.com/operagang/L2BR, CỐ ĐỊNH commit 941ce86afc09cf4e1c62400b9c5d17407a82d297.
- Đặt tại third_party/L2BR. KHÔNG sửa trực tiếp; mọi thay đổi lưu thành patch trong patches/.
- File chính: env.py (Env, step_rBRP, step_uBRP, create_mask_*), decoder.py (class L2BR, forward(x) -> logits),
  encoder.py, ess.py (ESS.validation), benchmark_loader.py (data_CVS, data_ZQLZ, data_LL),
  instance_generator.py, sampler.py, test.py, utils.py (load_model dùng weights_only=True), params.py.
- Trạng thái là tensor x kích thước (batch, S, T); Env.find_target_stack lấy container có GIÁ TRỊ LỚN NHẤT
  (giá trị lớn = lấy sớm hơn); Env.clear() tự lấy container và chuẩn hóa lại giá trị. Kiểm tra lại trong mã.
- uBRP: hành động là chỉ số trong S*S, nguồn = idx // S, đích = idx % S. rBRP: hành động là cột đích.
- Với CVS: tham số -t (max_tiers) = H + 2.
- Gán cứng 'cuda' ở: decoder.py (đối số mặc định device='cuda:0' tại dòng ~70 và ~83),
  encoder.py dòng ~116 (.to('cuda:0')), ess.py dòng ~308–333 (data_.to('cuda'), Env(device='cuda'), model.to('cuda')).
- ESS.validation chạy các rollout đồng bộ, dừng khi rollout đầu tiên kết thúc; traj_limit 500 (CVS/ZQLZ), 2000 (LL).
- Mô hình công khai: IL/final_models/{CVS-rBRP,CVS-uBRP}/H=5xS=5.pt (= L2BR-Base, cần xác minh),
  SIL/final_models/{CVS,ZQLZ,LL}-{rBRP,uBRP}/*.pt.

Mã của ta: package Python brp/ (không phụ thuộc chỉnh sửa trong third_party). Bố cục:
  brp/sim.py, brp/bounds.py, brp/exact.py, brp/data.py, brp/l2br_adapter.py,
  brp/search/pg_beam.py, brp/graph/build.py, brp/value/{gvalue.py,lstm_value.py},
  experiments/configs/*.yaml, experiments/run.py, brp/stats.py, tests/, results/ (gitignore), reports/.

Quy tắc bắt buộc:
1. Viết test (pytest) trước hoặc cùng mã; mọi test phải chạy qua trước khi báo xong.
2. Mọi chạy thực nghiệm ghi JSONL mỗi bài: instance_id, group, method, config_hash, seed, obj, time_s,
   hardware, git_commit, solution (dãy hành động). Không ghi đè kết quả cũ.
3. Mọi lời giải phải được KIỂM TRA LẠI bằng brp/sim.py (phát lại dãy hành động, đếm số lần dời).
4. KHÔNG chọn siêu tham số trên benchmark (CVS/ZQLZ/LL); chỉ trên bài sinh ngẫu nhiên.
5. Đặt seed cho mọi phần ngẫu nhiên. Ghi phiên bản Python, torch, phần cứng.
6. Máy local CHỈ có CPU: dùng cho test, debug, bài nhỏ. Mọi chạy cần GPU đi qua Kaggle CLI theo Mục E
   (tài khoản minhthuong123, lệnh `python -m kaggle ...`). Agent tự push, theo dõi, tải kết quả về
   kaggle_out/<slug>/ và đọc log; luôn kiểm tra dòng "cuda available: True" trong log
   (Kaggle có thể lặng lẽ chạy CPU). Không dùng SSH/tunnel vào Kaggle.
7. Khi xong: báo cáo con số thật (kể cả kết quả xấu), danh sách file đã tạo, lệnh để chạy lại.
```

---

## B. Danh sách task

### GIAI ĐOẠN 0: Hạ tầng (tuần 1)

#### T0.1 Khung dự án và chạy L2BR trên CPU
- **Phụ thuộc:** không.
- **Mục tiêu:** clone L2BR đúng commit vào `third_party/L2BR`, tạo patch cho thiết bị (tham số hóa `'cuda'` thành `device` lấy từ `torch.cuda.is_available()`), tạo `brp/l2br_adapter.py` để nạp mô hình và chạy best-of-M từ mã của ta.
- **Đầu ra:** `third_party/L2BR` (commit cố định), `patches/l2br_device.patch`, `scripts/setup_l2br.(sh|ps1)` (clone + áp patch), `brp/l2br_adapter.py` với hàm `best_of_m(model, instance, M, temperature, seed) -> (obj, actions, time)`, `requirements.txt`, `.gitignore`, `tests/test_l2br_adapter.py`.
- **Nghiệm thu:**
  - Chạy được trên CPU, không còn lỗi CUDA.
  - **Tái lập Bảng 2 của bài (cột L2BR Base, uBRP):** với mô hình IL CVS-uBRP H=5xS=5, M=2560, trung bình trên CVS H=3, S=3..8 lần lượt là 4,975; 6,025; 6,850; 8,275; 9,100; 10,300 (sai lệch ≤ 0,5%). Agent phải tự đối chiếu các số này với PDF `1-s2.0-S0377221726007551-main.pdf` trang 9.
  - Nếu không khớp: báo cáo sai lệch, không tự sửa số.

#### T0.2 Bộ mô phỏng tham chiếu và nạp dữ liệu
- **Phụ thuộc:** T0.1.
- **Mục tiêu:** `brp/sim.py` mô phỏng BRP một bài (thuần Python/numpy, trạng thái bất biến), cả rBRP và uBRP: hành động hợp lệ, áp dụng hành động, tự lấy container, `is_done`, `canonical_key` (sắp xếp các cột thành bộ chuẩn). `brp/data.py` đọc CVS, ZQLZ, LL (tái dùng hoặc bọc `benchmark_loader.py`), chuyển qua lại giữa `Instance` và tensor `x` của L2BR, và sinh bài ngẫu nhiên (bọc `instance_generator.py`).
- **Đầu ra:** `brp/sim.py`, `brp/data.py`, `tests/test_sim.py`, `tests/test_data.py`.
- **Nghiệm thu:**
  - Trên 1.000 bài ngẫu nhiên mỗi biến thể: phát lại các dãy hành động hợp lệ ngẫu nhiên trên cả `brp/sim.py` và `Env` của L2BR cho **cùng số lần dời** và cùng trạng thái kết thúc.
  - Chuyển `Instance → x → Instance` giữ nguyên.
  - Số bài nạp được: CVS 21 nhóm × 40, ZQLZ 125 nhóm × 100, LL 50 R + 20 U.
  - `canonical_key` bất biến khi hoán đổi các cột.

#### T0.3 Cận dưới và bộ giải chính xác cho bài nhỏ
- **Phụ thuộc:** T0.2.
- **Mục tiêu:** `brp/bounds.py`: `lb1(state)` (Mục 4.2 của đề xuất). `brp/exact.py`: IDA* hoặc nhánh cận dùng `lb1`, loại trạng thái trùng bằng `canonical_key`, cho bài tới khoảng 12 container.
- **Đầu ra:** hai file trên, `tests/test_bounds.py`, `tests/test_exact.py`.
- **Nghiệm thu:**
  - `lb1 ≤ OPT` trên 2.000 bài nhỏ ngẫu nhiên (rBRP và uBRP).
  - Trung bình tối ưu: CVS uBRP H=3, S=3 bằng **4,975** (Bảng 2, Algo-TS); CVS rBRP H=3, S=3 bằng **5,000** (Bảng 5, IP). Đối chiếu PDF.
  - Có test trên ví dụ tay (Hình 1 của bài: 3 cột, 9 container; lời giải minh họa trong bài dùng 4 lần dời, nên tối ưu phải ≤ 4; ghi lại giá trị tối ưu tìm được).

#### T0.4 Khung chạy thực nghiệm và thống kê
- **Phụ thuộc:** T0.2.
- **Mục tiêu:** `experiments/run.py` đọc YAML (phương pháp, bộ dữ liệu, nhóm, ngân sách thời gian, seed), chạy song song nhiều bài, ghi JSONL theo quy tắc ở Mục A, kiểm tra lại lời giải bằng `sim`. `brp/stats.py`: tổng hợp theo nhóm (định dạng như bảng L2BR: obj, t_avg), Wilcoxon cặp, bootstrap CI 95%, đánh dấu tốt nhất. `scripts/make_table.py` xuất Markdown và LaTeX.
- **Đầu ra:** các file trên, `tests/test_stats.py`, một cấu hình mẫu chạy được.
- **Nghiệm thu:** chạy thử end-to-end với phương pháp L2BR best-of-M trên CVS H=3; test thống kê khớp `scipy.stats` trên dữ liệu giả.

#### T0.5 Hạ tầng chạy trên Kaggle GPU
- **Phụ thuộc:** T0.1, T0.4.
- **Mục tiêu:** để mọi thực nghiệm chạy được trên Kaggle bằng một lệnh, theo quy trình ở Mục E.
- **Đầu ra:**
  - `scripts/kaggle_push_code.py`: đóng gói `brp/`, `experiments/`, `patches/` vào `kaggle/code_pkg/`, rồi gọi `push_dataset.py` để tạo hoặc cập nhật dataset riêng tư `minhthuong123/brp-code` (ghi git commit vào file `VERSION`).
  - `scripts/kaggle_job.py <config.yaml> --slug brp-<task>-<mô-tả>`: sinh `kaggle/jobs/<slug>/` gồm `kernel-metadata.json` (riêng tư, GPU, internet, `dataset_sources: [minhthuong123/brp-code]`) và `main.py`. `main.py` làm các việc:
    - in báo cáo GPU, dừng với lỗi nếu không có GPU;
    - tìm dataset code bằng cách duyệt `/kaggle/input`, thêm vào `sys.path`;
    - clone L2BR đúng commit vào `/kaggle/temp` rồi áp patch;
    - chạy `experiments/run.py` với config;
    - ghi JSONL và `summary.json` vào `/kaggle/working`, lưu checkpoint định kỳ để chịu được giới hạn 12 giờ.
  - `scripts/kaggle_run_task.(ps1|sh)`: gọi theo thứ tự push code, tạo job, rồi chạy `kaggle_run.py`; kết quả tải về `kaggle_out/<slug>/` rồi chép vào `results/<task>/`.
  - `kaggle/RUNS.md`: sổ ghi mỗi lần chạy (slug, version, config hash, GPU, thời gian, chỉ số chính, đường dẫn kết quả).
- **Nghiệm thu:**
  - Job `brp-smoke-l2br` (đã có sẵn ở `kaggle/jobs/smoke_l2br/`) chạy trên GPU và tái lập Bảng 2 (Base, H=3: 4,975; 6,025; 6,850).
  - Một job sinh bởi `kaggle_job.py` chạy end-to-end với config mẫu của T0.4 và trả JSONL hợp lệ về `results/`.

---

### GIAI ĐOẠN 1: Bước 0 và cổng G0 (tuần 1–2)

#### T1.1 Đo dư địa khi tăng số mẫu của L2BR
- **Phụ thuộc:** T0.1, T0.4.
- **Mục tiêu:** chạy L2BR best-of-M với M ∈ {256, 2.560, 25.600} và nhiệt độ ∈ {0,5; 1; 1,5} trên: CVS uBRP H=10, S=6 và S=10 (mỗi nhóm 40 bài, mô hình `SIL/final_models/CVS-uBRP/H=10xS=6.pt`, `H=10xS=10.pt`); ZQLZ uBRP T=7, S=10 (100 bài, `SIL/final_models/ZQLZ-uBRP/T=7xS=10.pt`). Ghi thời gian thật.
- **Đầu ra:** `results/T1_1/*.jsonl`, `reports/T1_1_headroom.md` có bảng và hình obj theo thời gian.
- **Nghiệm thu:** với M=2.560, kết quả khớp cột L2BR-Large/Targeted tương ứng của Bảng 2 và 3 trong khoảng ±1% (agent đối chiếu số trong PDF); báo cáo mức giảm khi M tăng từ 2.560 lên 25.600.
- **Chạy ở đâu:** **Kaggle GPU** qua T0.5 (slug `brp-t11-headroom`). Thử 5 bài trên CPU local trước để bắt lỗi.

#### T1.2 PG-Beam tối thiểu (V = LB1)
- **Phụ thuộc:** T0.2, T0.3, T0.1.
- **Mục tiêu:** cài `brp/search/pg_beam.py` theo giả mã Mục 4.3 của đề xuất, phiên bản một bài: đánh giá chính sách theo lô cho cả beam (một lần gọi `model(x)` cho B trạng thái), top-k hành động hợp lệ, điểm `g + LB1 − λ·log p`, loại trùng bằng `canonical_key`, cắt nhánh bằng UB, giới hạn thời gian, trả về lời giải tốt nhất (anytime).
- **Đầu ra:** `brp/search/pg_beam.py`, `tests/test_pg_beam.py`, cấu hình `experiments/configs/T1_2.yaml`.
- **Nghiệm thu:**
  - Mọi lời giải hợp lệ khi phát lại bằng `sim`, obj bằng số lần dời phát lại.
  - Trên bài nhỏ (CVS H=3, H=4), PG-Beam đạt đúng tối ưu từ T0.3 với B đủ lớn.
  - Kết quả lặp lại được với cùng seed.
  - Có test chứng tỏ không có vòng lặp (không trạng thái nào được mở rộng hai lần).

#### T1.3 So sánh cùng thời gian và báo cáo cổng G0
- **Phụ thuộc:** T1.1, T1.2.
- **Mục tiêu:** trên cùng các nhóm của T1.1 (bắt đầu 20 bài mỗi nhóm), so PG-Beam với L2BR best-of-M ở cùng thời gian (đặt bằng thời gian L2BR M=2.560 trên cùng máy). Chọn B, k, λ chỉ trên **bài sinh ngẫu nhiên** cùng kích thước, không trên benchmark.
- **Chạy ở đâu:** Kaggle GPU (slug `brp-t13-g0`); local CPU chỉ để thử 2–3 bài.
- **Đầu ra:** `reports/G0_decision.md`: bảng, kiểm định Wilcoxon, khoảng tin cậy, và **kết luận theo quy tắc**: đi tiếp nếu cải thiện ≥ 0,3% trên ít nhất một nhóm lớn (PG-Beam hoặc M lớn), ngược lại DỪNG.
- **Nghiệm thu:** báo cáo có đủ số liệu thật, không diễn giải vượt dữ liệu.

> **CỔNG G0:** người dùng đọc `reports/G0_decision.md`. Không đạt thì chuyển sang phương án dự phòng (đề xuất, Mục 11), không làm tiếp các giai đoạn dưới.

---

### GIAI ĐOẠN 2: PG-Beam đầy đủ và cổng G1 (tuần 3–6)

#### T2.1 PG-Beam theo lô, hỗ trợ GPU và rBRP
- **Phụ thuộc:** G0 đạt.
- **Mục tiêu:** vector hóa PG-Beam để giải nhiều bài cùng lúc (ghép beam của nhiều bài vào một lô), tính `LB1` và mặt nạ hành động bằng tensor, chạy trên GPU nếu có, hỗ trợ rBRP (hành động là cột đích), điều khiển thời gian chính xác.
- **Đầu ra:** cập nhật `brp/search/pg_beam.py` (hoặc `pg_beam_batched.py`), test so sánh với bản T1.2.
- **Nghiệm thu:** cùng seed và cấu hình cho **cùng obj** như bản T1.2 trên 100 bài; báo cáo tốc độ tăng.

#### T2.2 Chọn siêu tham số trên bài sinh ngẫu nhiên
- **Phụ thuộc:** T2.1.
- **Mục tiêu:** tìm B, k, λ, nhiệt độ cho từng ngân sách thời gian (1, 5, 30 giây/bài) trên tập kiểm định sinh ngẫu nhiên đúng kích thước từng nhóm (tối thiểu 100 bài/kích thước). Có thể dùng Optuna.
- **Đầu ra:** `reports/T2_2_tuning.md`, cấu hình cố định `experiments/configs/pgbeam_final.yaml`.
- **Nghiệm thu:** chứng minh bằng hash rằng không bài benchmark nào nằm trong tập chọn tham số.

#### T2.3 Chạy benchmark đầy đủ
- **Phụ thuộc:** T2.2.
- **Mục tiêu:** chạy PG-Beam và L2BR (cùng máy, cùng thời gian) trên toàn bộ CVS và ZQLZ uBRP ở 3 mức thời gian; sau đó rBRP (CVS, ZQLZ) và LL.
- **Chạy ở đâu:** Kaggle GPU, tách mỗi bộ dữ liệu và mỗi mức thời gian thành một job riêng (slug `brp-t23-<bộ>-<thời gian>`) để mỗi job dưới 9 giờ.
- **Đầu ra:** `results/T2_3/`, `reports/T2_3_benchmark.md`, bảng theo định dạng L2BR (đậm = tốt nhất), danh sách nhóm phá kỷ lục so với số liệu công bố, và **file lời giải** của mọi kỷ lục mới.
- **Nghiệm thu:** mọi kỷ lục mới đều được phát lại hợp lệ bằng `sim`; bảng ghi rõ nguồn số liệu baseline (chạy lại hay lấy từ bài).

#### T2.4 Đối chứng và báo cáo cổng G1
- **Phụ thuộc:** T2.3.
- **Mục tiêu:** chạy đối chứng: không LB (V = 0), không chính sách (phân phối đều), không loại trùng, quét B và k. Viết báo cáo G1.
- **Chạy ở đâu:** Kaggle GPU (slug `brp-t24-ablation-<biến thể>`).
- **Đầu ra:** `reports/G1_decision.md`, kết luận theo quy tắc: H1 đạt nếu cải thiện trung bình ≥ 0,5% trên CVS uBRP H≥6 với p < 0,05.
- **Nghiệm thu:** như T1.3.

> **CỔNG G1:** không đạt thì viết bài ngắn hoặc hội nghị với kết quả hiện có; không làm Giai đoạn 3.

---

### GIAI ĐOẠN 3: G-Value trên đồ thị và cổng G2 (tuần 7–11)

#### T3.1 Dựng đồ thị chắn
- **Phụ thuộc:** T0.2, T0.3.
- **Mục tiêu:** `brp/graph/build.py`: từ trạng thái (hoặc lô tensor `x`) dựng đồ thị dị loại theo Mục 4.4 của đề xuất: nút container và nút cột; cạnh "nằm trên", "chắn", "thuộc cột"; đặc trưng nút. Hỗ trợ ghép lô. Ưu tiên PyTorch Geometric; nếu không cài được trên Windows, cài bằng tensor PyTorch thuần.
- **Đầu ra:** `brp/graph/build.py`, `tests/test_graph.py`.
- **Nghiệm thu:**
  - Số nút container có cạnh "chắn" đi ra **bằng đúng** `lb1(state)` trên 2.000 trạng thái ngẫu nhiên.
  - Đúng trên ví dụ tay (Hình 1 của bài).
  - Ghép lô và tách lô khớp nhau.
  - Báo cáo tốc độ dựng (trạng thái/giây).

#### T3.2 Sinh dữ liệu huấn luyện hàm giá trị
- **Phụ thuộc:** T2.1 (hoặc T1.2), T0.3.
- **Mục tiêu:** sinh bài ngẫu nhiên kích thước H×S từ 6×6 tới 10×10 (uBRP), giải bằng PG-Beam cấu hình mạnh; lưu mọi trạng thái trên lời giải tốt nhất kèm nhãn "số lần dời còn lại". Thêm tập bài nhỏ có nhãn tối ưu (T0.3) để đo độ lệch nhãn. Chia train/val/test **theo bài**.
- **Chạy ở đâu:** Kaggle GPU (slug `brp-t32-valuedata-<kích thước>`). Kết quả đưa lên dataset riêng `minhthuong123/brp-value-data` bằng `push_dataset.py`, KHÔNG gộp vào dataset code.
- **Đầu ra:** `data/value/{train,val,test}.pt` (hoặc parquet), `reports/T3_2_dataset_card.md` (số lượng, phân phối nhãn, độ lệch nhãn trên bài nhỏ).
- **Nghiệm thu:** không trạng thái nào xuất phát từ bài benchmark (kiểm tra hash); dataset card đầy đủ.

#### T3.3 Mô hình G-Value
- **Phụ thuộc:** T3.1, T3.2.
- **Mục tiêu:** `brp/value/gvalue.py`: GNN dị loại 3–4 lớp, gộp toàn đồ thị, đầu ra softplus; mất mát Huber; huấn luyện với early stopping trên val; 5 seed.
- **Chạy ở đâu:** Kaggle GPU (slug `brp-t33-gvalue-seed<k>`), input là `brp-code` và `brp-value-data`; mô hình đưa lên `minhthuong123/brp-models`.
- **Đầu ra:** mô hình đã huấn luyện, `reports/T3_3_gvalue.md` có MAE, Spearman so với nhãn, so với `lb1`.
- **Nghiệm thu:** chạy được trên CPU với tập nhỏ; báo cáo thời gian suy luận cho mỗi trạng thái.

#### T3.4 Baseline: đầu giá trị trên bộ mã hóa L2BR
- **Phụ thuộc:** T3.2.
- **Mục tiêu:** `brp/value/lstm_value.py`: thêm đầu hồi quy trên embedding của bộ mã hóa L2BR (trung bình embedding các cột), hai biến thể: đóng băng bộ mã hóa và tinh chỉnh. **Cùng dữ liệu, cùng ngân sách huấn luyện** với T3.3.
- **Chạy ở đâu:** Kaggle GPU, cùng cấu hình máy với T3.3.
- **Đầu ra:** mô hình, `reports/T3_4_lstm_value.md`.
- **Nghiệm thu:** như T3.3.

#### T3.5 Tích hợp vào PG-Beam và báo cáo cổng G2
- **Phụ thuộc:** T3.3, T3.4, T2.2.
- **Mục tiêu:** điểm `g + max(LB1, V) − λ·log p`; chọn lại λ trên bài sinh ngẫu nhiên; so sánh 4 biến thể (LB1, giá trị LSTM, G-Value bỏ cạnh "chắn", G-Value đầy đủ) trên CVS và ZQLZ uBRP lớn ở cùng thời gian, tính cả thời gian suy luận của mạng giá trị.
- **Chạy ở đâu:** Kaggle GPU (input thêm `brp-models`).
- **Đầu ra:** `reports/G2_decision.md`, kết luận theo quy tắc: G-Value phải có Spearman cao hơn rõ rệt so với LB1 và giá trị LSTM, **và** cải thiện obj ở cùng thời gian.
- **Nghiệm thu:** nếu giá trị LSTM ngang G-Value, báo cáo phải nói thẳng "không có bằng chứng lợi ích từ đồ thị".

> **CỔNG G2:** không đạt thì giữ PG-Beam (cộng giá trị LSTM nếu nó có ích), bỏ tuyên bố về đồ thị.

---

### GIAI ĐOẠN 4: Thực nghiệm cuối và cổng G3 (tuần 12–14)

#### T4.1 Benchmark cuối cùng
- **Phụ thuộc:** G2 (đạt hoặc không, dùng cấu hình tốt nhất).
- **Mục tiêu:** chạy cấu hình cuối trên toàn bộ CVS, ZQLZ, LL (rBRP và uBRP), 3 mức thời gian, 5 seed; thống kê đầy đủ; cập nhật danh sách kỷ lục mới cùng file lời giải.
- **Chạy ở đâu:** Kaggle GPU, một job cho mỗi (bộ dữ liệu, biến thể, mức thời gian, seed); có script gom kết quả.
- **Đầu ra:** `results/final/`, `reports/G3_decision.md` (quy tắc: cải thiện ≥ 2% ở các nhóm lớn nhất và H4 đạt thì nhắm Q1, ngược lại Q2).

#### T4.2 (Tùy chọn, cần GPU) Chưng cất PG-Beam vào chính sách
- **Phụ thuộc:** T4.1.
- **Mục tiêu:** sửa quy trình SIL (`train_SIL.py`, `ess.py` qua patch hoặc bản sao trong `brp/`) để nhãn lấy từ PG-Beam thay vì best-of-M; so chính sách mới và cũ ở cùng ngân sách suy luận (best-of-M và PG-Beam).
- **Chạy ở đâu:** Kaggle GPU; chia nhỏ theo epoch và lưu checkpoint vào `/kaggle/working` vì giới hạn 12 giờ mỗi phiên, phiên sau đọc checkpoint từ output của phiên trước (thêm vào `kernel_sources`).
- **Đầu ra:** mô hình, `reports/T4_2_distill.md`.

#### T4.3 Kiểm tra tổng quát hóa trên bộ dữ liệu bổ sung
- **Phụ thuộc:** T4.1.
- **Mục tiêu:** tải và đọc bộ bài lớn của Tricoire et al. (https://plis.univie.ac.at/research/blockrelocating) và Bacci et al. (https://www.iasi.cnr.it/~tbacci/); chạy cấu hình cuối. Nếu không tải được hoặc giấy phép không rõ, ghi lại và bỏ qua.
- **Đầu ra:** bộ đọc mới trong `brp/data.py`, `reports/T4_3_generalization.md`.

---

### GIAI ĐOẠN 5: Sản phẩm cho bài báo (tuần 15–20)

#### T5.1 Hình và bảng
- **Phụ thuộc:** T4.1.
- **Mục tiêu:** hình chất lượng theo thời gian, theo quy mô, cột đối chứng; bảng LaTeX định dạng giống L2BR; tất cả sinh từ JSONL bằng script.
- **Đầu ra:** `paper/figures/*.pdf`, `paper/tables/*.tex`, `scripts/make_figures.py`.
- **Nghiệm thu:** chạy lại script từ JSONL ra đúng hình và bảng (không sửa tay).

#### T5.2 Gói tái lập
- **Phụ thuộc:** T5.1.
- **Mục tiêu:** README tái lập (cài đặt, tải dữ liệu từ nguồn gốc, lệnh cho từng bảng), file môi trường, script `reproduce_table_X`, mô hình G-Value đã huấn luyện, file lời giải kỷ lục kèm bộ kiểm tra, ghi chú giấy phép (mã của ta, mã L2BR MIT, dữ liệu thuộc tác giả gốc).
- **Đầu ra:** `README.md`, `environment.yml` hoặc `requirements.txt` cố định phiên bản, `scripts/reproduce_*`.
- **Nghiệm thu:** trên một bản clone sạch, làm theo README tái lập được ít nhất một bảng nhỏ.

**Việc con người phải làm (không giao agent):** chứng minh các mệnh đề (Mục 4.3 của đề xuất), đọc toàn văn tài liệu (Mục 14 của đề xuất), kiểm tra phân hạng tạp chí, viết bài.

---

## C. Thứ tự và phụ thuộc

```
T0.1 ─┬─ T0.2 ─┬─ T0.3 ─┐
      │        └─ T0.4 ─┴─ T0.5 (Kaggle)
      └──────────────── T1.1 ─┐
             T0.3 + T0.2 ── T1.2 ─┴─ T1.3 ══ [G0]
                                            │
                       T2.1 ─ T2.2 ─ T2.3 ─ T2.4 ══ [G1]
                                            │
        T3.1 ─┐   T3.2 ─┬─ T3.3 ─┐
              └─────────┴─ T3.4 ─┴─ T3.5 ══ [G2]
                                            │
                                 T4.1 ══ [G3] ─┬─ T4.2 (tùy chọn)
                                               └─ T4.3
                                            │
                                     T5.1 ─ T5.2
```

Có thể chạy song song: T0.3 và T0.4; T1.1 và T1.2; T3.1 và T3.2 (khi đã có PG-Beam); T3.3 và T3.4.

---

## D. Mẫu giao việc cho agent

```
[Dán toàn bộ Mục A: Ngữ cảnh chung]

TASK: <dán nguyên khối task, ví dụ T1.2>

Yêu cầu thêm:
- Làm trên nhánh git riêng: task/<mã-task>.
- Trước khi code, đọc các file liên quan và nêu ngắn gọn kế hoạch.
- Kết thúc bằng: (1) danh sách file đã tạo/sửa, (2) lệnh chạy test và kết quả,
  (3) số liệu nghiệm thu thực tế so với tiêu chí, (4) điều gì chưa làm được và vì sao.
```

**Mẹo khi dùng agent:**
- Mỗi phiên chỉ một task; nếu task lớn (T2.3, T4.1), tách theo bộ dữ liệu.
- Kiểm tra tiêu chí nghiệm thu bằng chính lệnh agent đưa ra trước khi chuyển task sau.
- Ở mỗi cổng (G0–G3), tự đọc báo cáo và quyết định; đừng để agent tự quyết đi tiếp.
- Chạy nặng đều qua Kaggle (Mục E); agent tự push, theo dõi và tải kết quả. Bạn chỉ cần giữ tài khoản Kaggle đăng nhập và còn quota GPU.

---

## E. Quy trình chạy trên Kaggle GPU

**Đã kiểm tra trên máy này (07/11/2026):** Kaggle CLI 2.2.5, tài khoản `minhthuong123`, gọi qua `python -m kaggle ...` (trên Windows `kaggle.exe` thường không có trong PATH).

**Xác thực:** dùng `KAGGLE_API_TOKEN` trong file `.env` ở gốc repo (đã có trong `.gitignore`; **không bao giờ commit, in ra log hay đưa vào dataset**). Thứ tự ưu tiên: biến môi trường của tiến trình, rồi `.env`, rồi `~/.kaggle/`. Kiểm tra bằng `python kaggle/kaggle_cli.py auth` (chỉ in nguồn xác thực, không in khóa).

**Công cụ chính: `kaggle/kaggle_cli.py`** (đã viết và chạy thử `auth`, `status`):
```
python kaggle/kaggle_cli.py auth                      # nguồn xác thực + kiểm tra kết nối
python kaggle/kaggle_cli.py run kaggle/jobs/<job> [--accelerator NvidiaTeslaT4|NvidiaTeslaP100|none]
python kaggle/kaggle_cli.py status <slug>             # queued / running / complete / error
python kaggle/kaggle_cli.py logs <slug> [-f]          # in log một lần, hoặc theo dõi trực tiếp
python kaggle/kaggle_cli.py fetch <slug>              # tải output của lần chạy mới nhất
```

**Xem log ở đâu:**

| Nơi | Nội dung |
|---|---|
| Màn hình khi chạy `kaggle_cli.py run` hoặc `logs -f` | Log trực tiếp, mỗi dòng có giờ local |
| `kaggle_out/<slug>/live.log` | Bản lưu của log trực tiếp (mở bằng VS Code, tự cập nhật) |
| `kaggle_out/<slug>/status.log` | Các lần đổi trạng thái (queued, running, complete, error) kèm thời gian |
| `kaggle_out/<slug>/<slug>.log` | Log đầy đủ do Kaggle tạo, tải về khi job kết thúc |
| `kaggle_out/<slug>/run.log`, `summary.json` | Log và số liệu do chính job ghi vào `/kaggle/working` (task T0.5 phải đưa bộ ghi này vào mẫu `main.py`) |
| Trình duyệt: `https://www.kaggle.com/code/minhthuong123/<slug>` | Trang job, tab Logs của phiên đang chạy |

**Trạng thái `queued` lâu:** Kaggle chưa cấp GPU. Không phải lỗi kết nối. Nguyên nhân thường gặp: các phiên GPU khác của tài khoản đang mở trong trình duyệt giữ suất, hoặc hàng chờ đông. Kaggle CLI **không có lệnh hủy phiên** (chỉ có `delete`, xóa hẳn notebook); muốn dừng phiên khác phải vào kaggle.com/code, bấm **Active Events**, rồi **Stop**. Khi log stream báo lỗi `500` lúc job còn `queued` thì là bình thường, công cụ tự thử lại.

Skill `kaggle-remote` có thêm 3 script dùng chung tại `C:\Users\X1\.claude\skills\kaggle-remote\scripts\`: `new_job.py` (tạo job mẫu), `kaggle_run.py`, `push_dataset.py` (tạo hoặc cập nhật dataset riêng tư).

**Bố cục trong repo:**
```
kaggle/
  jobs/<slug>/{kernel-metadata.json, main.py}   # mỗi job một thư mục (được commit)
  code_pkg/                                     # bản đóng gói tạm để push (gitignore)
  RUNS.md                                       # sổ ghi các lần chạy
kaggle_out/<slug>/                              # kết quả tải về (gitignore)
```

**Dataset riêng tư trên Kaggle (tách riêng để đổi code không phải tải lại dữ liệu lớn):**

| Dataset | Nội dung | Cập nhật khi |
|---|---|---|
| `minhthuong123/brp-code` | `brp/`, `experiments/`, `patches/`, file `VERSION` (git commit) | Mỗi lần code thay đổi, TRƯỚC khi push job |
| `minhthuong123/brp-value-data` | Dữ liệu huấn luyện G-Value (T3.2) | Khi sinh dữ liệu mới |
| `minhthuong123/brp-models` | Mô hình G-Value và mô hình chưng cất | Khi có mô hình mới |

Mã L2BR và dữ liệu benchmark **không** đưa lên dataset: job tự `git clone` repo L2BR đúng commit `941ce86…` (cần bật internet), rồi áp `patches/`.

**Các bước cho một lần chạy:**
```
# 1. Đẩy code mới nhất
python scripts/kaggle_push_code.py -m "<ghi chú>"
# 2. Sinh job từ config
python scripts/kaggle_job.py experiments/configs/<cfg>.yaml --slug brp-<task>-<mô-tả>
# 3. Chạy, chờ, tải kết quả (chạy nền với job dài)
python kaggle/kaggle_cli.py run kaggle/jobs/<slug> --timeout 32400   # log trực tiếp vào kaggle_out/<slug>/live.log
# 4. Theo dõi lại từ terminal khác (tùy chọn)
python kaggle/kaggle_cli.py logs <slug> -f
# 5. Kết quả nằm ở kaggle_out/<slug>/ -> chép vào results/<task>/, ghi một dòng vào kaggle/RUNS.md
```

**Quy tắc cho mọi job:**
- Mọi dòng in ra phải đồng thời được ghi vào `/kaggle/working/run.log` (bộ ghi kiểu tee, do T0.5 cung cấp), mỗi dòng có tiền tố rõ: `[PROGRESS]`, `[RESULT]`, `[SUMMARY]`, `[ERROR]`.
- Đầu `main.py` in `torch`, `cuda available` và tên GPU, rồi **dừng với lỗi nếu không có GPU**. Kaggle có thể lặng lẽ chạy CPU khi tài khoản chưa xác minh số điện thoại hoặc hết quota.
- Chỉ file trong `/kaggle/working` được tải về (khoảng 20 GB). In thêm `SUMMARY {json}` ra stdout để vẫn có số liệu nếu tải file lỗi.
- Không hard-code đường dẫn input; duyệt `/kaggle/input` để tìm file đánh dấu (ví dụ `VERSION`).
- Mỗi phiên tối đa khoảng 12 giờ: đặt `--timeout` khoảng 9 giờ và lưu checkpoint hoặc JSONL theo từng bài để chạy tiếp được.
- In một dòng ngắn mỗi bài hoặc mỗi epoch với `flush=True` (tiến độ, obj, thời gian, ETA) để theo dõi được bằng `kernels logs -f`.
- Không đưa token hay khóa API vào code; cần thì dùng Kaggle Secrets.

**Quota GPU (khoảng 30 giờ/tuần, cần kiểm tra trên kaggle.com/settings):** ưu tiên theo thứ tự cổng. Ước lượng thô cần kiểm chứng ở job đầu tiên:
- T1.1 + T1.3 (cổng G0): vài giờ GPU.
- T2.3, T2.4: chia nhiều job, có thể vượt một tuần quota; chạy dần theo bộ dữ liệu.
- T3.2–T3.5: phụ thuộc kích thước dữ liệu; bắt đầu nhỏ.
- Không dùng GPU cho test hay smoke test thuần CPU (`--accelerator none`).
