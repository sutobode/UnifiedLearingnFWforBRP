# ĐỀ XUẤT NGHIÊN CỨU (BẢN CHỐT)

## Bound-aware search with graph-based value estimation over learned policies for the Block Relocation Problem

*Tên tiếng Việt: Tìm kiếm có cận dưới và hàm giá trị đồ thị trên chính sách học được cho bài toán dời container*

*Bản chốt ngày 07/11/2026, thay thế mọi bản đề xuất trước. Kế hoạch task cho agent lập trình: `TASKS_Agent_Coding.md`.*

**Quy ước:** **[ĐÃ KIỂM CHỨNG]** = đã tải, đọc hoặc chạy trong phiên làm việc. **[CHƯA]** = phải kiểm tra trước khi đưa vào bài báo.

---

## 1. Tóm tắt

L2BR (Shin et al., EJOR 2026) huấn luyện một chính sách mạng nơron cho bài toán dời container (BRP) bằng bắt chước (IL) rồi tự cải thiện (SIL), đạt kết quả tốt nhất hiện có. Khi giải, L2BR chỉ **lấy mẫu độc lập nhiều lần rồi chọn lời giải tốt nhất**. Nghiên cứu này giữ nguyên chính sách đã huấn luyện (công khai), và thay cách giải bằng một **beam search có cận dưới** (PG-Beam). Beam được dẫn hướng bởi chính sách, và chấm điểm bằng cận dưới chấp nhận được cộng một **hàm giá trị học trên đồ thị chắn giữa các container** (G-Value).

**Mục tiêu đo được:** giảm số lần dời so với L2BR ở cùng thời gian chạy trên các bài uBRP lớn chưa biết nghiệm tối ưu, và phá kỷ lục đã công bố ở một phần các nhóm bài.

**Điều kiện thực hiện:** chỉ dùng dữ liệu và mô hình công khai; không thu thập dữ liệu.

---

## 2. Khoảng trống nghiên cứu

| # | Khoảng trống | Bằng chứng |
|---|---|---|
| G1 | Chính sách học được cho BRP chỉ được dùng bằng lấy mẫu (best-of-M), chưa kết hợp với tìm kiếm có cận như BBS | L2BR Mục 3.3.2 và mã `ess.py` **[ĐÃ KIỂM CHỨNG]**. Tìm kiếm tài liệu chưa thấy beam search dẫn bằng chính sách học được cho uBRP **[CHƯA đọc toàn văn các bài gần]** |
| G2 | Trên bài uBRP lớn, chưa có cận dưới hay nghiệm tối ưu, nên kỷ lục hiện tại có thể còn xa tối ưu | Bảng 2 và 3 của L2BR: CVS H=10 và ZQLZ T=7 không có kết quả chính xác. L2BR-Base so với L2BR-Large ở CVS 10×10: 112,08 so với 93,20, chênh 17%. Hình 5b: đường SIL vẫn giảm ở 7×10⁴ giây **[ĐÃ KIỂM CHỨNG trên PDF]** |
| G3 | GNN chưa dùng được cho BRP: L2BR thử GNN làm bộ mã hóa chính sách và thất bại ở bài lớn | L2BR Mục 4.9: "failed to generate terminated solutions during SIL on larger instances" **[ĐÃ KIỂM CHỨNG]**. Không tìm thấy bài GNN nào cho BRP/CRP **[CHƯA đủ để nói "đầu tiên"]** |

**Không nhắm vào:** bài nhỏ, bài vừa và bộ LL (L2BR đã tối ưu hoặc trong khoảng 0,7% cận dưới).

---

## 3. Đóng góp dự kiến

1. **PG-Beam:** beam search dẫn bằng chính sách L2BR, chấm điểm bằng số lần dời đã dùng cộng cận dưới chấp nhận được, có loại trạng thái trùng (kể cả hoán vị cột) và cắt nhánh bằng cận trên. Chứng minh: không đi vòng, luôn kết thúc, và cắt nhánh bằng cận dưới chấp nhận được không loại nghiệm tốt hơn cận trên hiện có.
2. **G-Value:** GNN trên đồ thị chắn ước lượng số lần dời còn lại, dùng làm hàm giá trị cho PG-Beam. Giải thích vì sao GNN thất bại khi làm chính sách (L2BR) nhưng hữu ích khi làm hàm giá trị: hàm giá trị chỉ cần xếp hạng trạng thái, không phải tự sinh quỹ đạo kết thúc được.
3. **Kết quả mới nhất** trên các bài uBRP lớn công khai, kèm mã và nghiệm công bố để tái lập.
4. (Tùy chọn, nếu có GPU) **Chưng cất:** huấn luyện lại chính sách bằng nghiệm của PG-Beam thay cho best-of-M.

---

## 4. Phương pháp

### 4.1 Bài toán
Giữ đúng định nghĩa của L2BR: S cột, chiều cao tối đa T, N container, độ ưu tiên biết trước, mục tiêu tối thiểu số lần dời; hai biến thể rBRP và uBRP. Container ở đỉnh và đúng lượt thì tự động được lấy.

### 4.2 Cận dưới LB1
`LB1(s)` = số container nằm trên ít nhất một container có độ ưu tiên lấy trước nó trong cùng cột. Mỗi container như vậy phải được dời ít nhất một lần, mỗi lần dời chỉ chuyển một container, nên `LB1` chấp nhận được (không vượt số lần dời còn lại tối ưu) cho cả rBRP và uBRP. **[CHƯA: đối chiếu cách đặt tên và các cận mạnh hơn trong Jin & Tanaka (2023), Tanaka & Mizuno (2018)]**

### 4.3 PG-Beam
```
Đầu vào: trạng thái s0, chính sách π (L2BR), bề rộng B, nhánh k, trọng số λ, ngân sách thời gian τ
UB ← kết quả best-of-M của L2BR (hoặc +∞)
Beam ← {s0}; Đã_thăm ← {khóa_chuẩn(s0)}
while Beam khác rỗng và chưa hết τ:
    Ứng_viên ← ∅
    for s in Beam:                       # chạy theo lô trên GPU/CPU
        for a in top-k hành động hợp lệ theo π(·|s):
            s' ← áp dụng a, tự động lấy container
            if s' rỗng: UB ← min(UB, g(s')); continue
            if khóa_chuẩn(s') ∈ Đã_thăm: continue
            if g(s') + LB1(s') ≥ UB: continue      # không thể tốt hơn
            điểm(s') ← g(s') + max(LB1(s'), V(s')) − λ·log π(a|s)
            thêm s' vào Ứng_viên và Đã_thăm
    Beam ← B ứng viên có điểm thấp nhất
return UB và lời giải tương ứng
```
- `g(s)`: số lần dời đã dùng. `V(s)`: G-Value (bản không có G-Value dùng `V = LB1`).
- `khóa_chuẩn`: sắp xếp các cột thành bộ chuẩn, vì với mục tiêu số lần dời các cột hoán đổi được cho nhau.
- Mệnh đề cần chứng minh trong bài: (i) tập trạng thái hữu hạn và không thăm lại, nên thuật toán kết thúc; (ii) nhánh bị cắt có `g + LB1 ≥ UB` nên không chứa lời giải tốt hơn UB.

### 4.4 Đồ thị chắn và G-Value
- **Nút container:** độ ưu tiên chuẩn hóa (như công thức (1) của L2BR), tầng chuẩn hóa, cờ "đang chắn".
- **Nút cột (ảo):** chiều cao, số chỗ trống, độ ưu tiên nhỏ nhất trong cột.
- **Cạnh:** "nằm trên" (container với container ngay dưới); "chắn" (b nằm trên c, c phải lấy trước b); "thuộc cột" (container với nút cột của nó).
- **Mô hình:** GNN dị loại 3–4 lớp (ví dụ GATv2 hoặc GINE), gộp toàn đồ thị, đầu ra `V(s) ≥ 0` (dùng softplus). Hàm mất mát: Huber trên số lần dời còn lại.
- **Nhãn:** số lần dời còn lại của lời giải tốt nhất tìm được (PG-Beam hoặc best-of-M lớn) trên bài sinh ngẫu nhiên. Bài nhỏ có thể dùng lời giải chính xác để kiểm tra độ lệch nhãn.

---

## 5. Giả thuyết

- **H1:** Cùng thời gian, PG-Beam (V = LB1) cho số lần dời trung bình thấp hơn best-of-M của L2BR trên CVS uBRP H≥6 và ZQLZ uBRP T≥6.
- **H2:** Lợi thế tăng theo kích thước bãi.
- **H3:** PG-Beam không kém L2BR trên bài nhỏ (giữ tối ưu).
- **H4:** PG-Beam với G-Value tốt hơn PG-Beam chỉ dùng LB1, **và** tốt hơn khi dùng đầu giá trị trên bộ mã hóa LSTM + Transformer (cùng dữ liệu).

---

## 6. Dữ liệu và tài nguyên (công khai)

| Tài nguyên | Nội dung | Trạng thái |
|---|---|---|
| Repo L2BR, commit `941ce86afc09cf4e1c62400b9c5d17407a82d297` | Mã (MIT), mô hình IL/SIL/RL cho rBRP và uBRP | **[ĐÃ KIỂM CHỨNG]** |
| CVS (Caserta et al., 2012) | 21 nhóm × 40 bài, tới 100 container | **[ĐÃ KIỂM CHỨNG]** `datasets/CVS` |
| ZQLZ (Zhu et al., 2012) | 125 nhóm × 100 bài | **[ĐÃ KIỂM CHỨNG]** `datasets/ZQLZ` |
| LL (Lee & Lee, 2010) | 50 bài R + 20 bài U, tới 720 container | **[ĐÃ KIỂM CHỨNG]** `datasets/LL` |
| Bài sinh ngẫu nhiên để huấn luyện G-Value | Bộ sinh `instance_generator.py` của L2BR | **[ĐÃ KIỂM CHỨNG]** có trong repo |
| Bộ bài lớn bổ sung (Tricoire et al., ĐH Vienna; Bacci et al.) | Kiểm tra tổng quát hóa | **[CHƯA]** tải thử |

Dữ liệu benchmark thuộc tác giả gốc. Khi công bố, chỉ đăng mã và hướng dẫn tải.

---

## 7. Baseline

| Phương pháp | Nguồn số liệu |
|---|---|
| L2BR best-of-M (Base, Large, Targeted) | Chạy lại bằng mô hình công khai, cùng phần cứng, cùng thời gian |
| Algo-TS, Algo-TS-TL (uBRP) | Số liệu công bố trong L2BR; chạy lại nếu tải được mã **[CHƯA]** |
| BBS (rBRP) | Số liệu công bố; chạy lại nếu mã không cần phần mềm trả phí **[CHƯA]** |
| GLAH, ACO-C, IP, CRP-DRL, DAM, MDDAM | Số liệu công bố trong L2BR |
| Đối chứng nội bộ | Beam không LB; beam không chính sách; beam với đầu giá trị LSTM; G-Value bỏ cạnh "chắn" |

---

## 8. Thực nghiệm

- **RQ1 (H1, H3):** PG-Beam so với L2BR ở các mức thời gian 1, 5, 30 giây mỗi bài, trên toàn bộ CVS và ZQLZ uBRP.
- **RQ2 (H2):** đường cong chất lượng theo thời gian và theo quy mô.
- **RQ3 (H4):** đóng góp của G-Value và của cấu trúc đồ thị.
- **RQ4:** phá kỷ lục ở những nhóm nào; kiểm tra thêm rBRP và LL.
- **Chỉ số:** số lần dời trung bình, khoảng cách tới kỷ lục, số nhóm phá kỷ lục, thời gian trung bình mỗi bài, tương quan hạng Spearman của V(s).
- **Thống kê:** kiểm định Wilcoxon theo cặp trên cùng bài, khoảng tin cậy bootstrap 95%, 5 hạt giống cho phần ngẫu nhiên và huấn luyện.
- **Công bằng:** cùng máy, cùng ngân sách thời gian, báo cáo cấu hình phần cứng; mọi siêu tham số chọn trên tập kiểm định sinh ngẫu nhiên, không trên benchmark.

---

## 9. Kết quả kỳ vọng và cổng quyết định

| Cổng | Khi nào | Điều kiện đi tiếp | Nếu không đạt |
|---|---|---|---|
| **G0** | Tuần 2 | PG-Beam tối thiểu (V = LB1) hoặc best-of-M với M lớn cải thiện ≥ **0,3%** so với L2BR mặc định, cùng thời gian, trên ít nhất một nhóm uBRP lớn | Dừng; chuyển sang phương án dự phòng (Mục 11) |
| **G1** | Tuần 6 | H1 đạt: cải thiện ≥ **0,5%** trung bình trên CVS uBRP H≥6, có ý nghĩa thống kê | Viết bài ngắn hoặc hội nghị |
| **G2** | Tuần 9 | G-Value có Spearman cao hơn LB1 rõ ràng và cao hơn đầu giá trị LSTM | Bỏ G-Value, giữ "PG-Beam + giá trị LSTM" |
| **G3** | Tuần 14 | H4 đạt và cải thiện ≥ **2%** ở bài lớn nhất | Nhắm Q2 thay vì Q1 |

**Khả năng công bố (ước lượng, không đảm bảo):** đạt G1 thì nhắm **Q2**; đạt cả G2 và G3 thì có cơ sở nhắm **Q1**.

---

## 10. Tiến độ (20 tuần)

| Tuần | Giai đoạn | Task (xem `TASKS_Agent_Coding.md`) |
|---|---|---|
| 1–2 | Hạ tầng và Bước 0 | T0.x, T1.x, cổng G0 |
| 3–6 | PG-Beam đầy đủ | T2.x, cổng G1 |
| 7–11 | G-Value | T3.x, cổng G2 |
| 12–14 | Benchmark đầy đủ, đối chứng, (tùy chọn) chưng cất | T4.x, cổng G3 |
| 15–20 | Hình, bảng, gói tái lập, viết bài | T5.x |

---

## 11. Phương án dự phòng

1. **BRP ngẫu nhiên** (thứ tự lấy chỉ biết theo nhóm), bài phân tích giá trị của thông tin. Chỉ phù hợp Q2 thấp hoặc hội nghị. Những gì đã kiểm chứng:
   - **Dữ liệu:** SCRP của Bacci, Mattia, Ventura (2023, *Soft Computing* 27(7):4223–4233), tại https://www.iasi.cnr.it/~tbacci/scrp.html. Gồm 2.880 bài: LargeBatches và SmallBatches, mỗi loại 24 cấu hình (5–10 cột, 3–6 tầng) × 60 bài. Mã heuristic RIRH bằng C++ thuần, giấy phép GPL, không cần Gurobi.
   - **Công trình trùng:** đã có DRL cho thứ tự lấy không chắc chắn (Wen, Yan, Song, Kim, Feng, SSRN 6118627; chưa đọc toàn văn), Tang et al. (2024, *Computers & Industrial Engineering*) và Zhang et al. (2025, *Advanced Engineering Informatics*).
   - **Thiên lệch hindsight của SIL:** liệt kê toàn bộ 200 bài mỗi cấu hình (3–4 cột, 7–9 container, rBRP, mô hình nhóm). Bắt chước lời giải tốt nhất theo hindsight kém tối ưu trung bình chỉ **0,00–0,15%** (tệ nhất 6,67%, sai ở tới 58/200 bài). Khoảng cách giữa "biết trước tương lai" và tối ưu thực tế khoảng 3–6%.
   - Muốn đi hướng này, phải viết lại script kiểm chứng: quy hoạch động chính xác theo kỳ vọng, so với chính sách bỏ phiếu hindsight.
2. **Chuyển PG-Beam sang bài toán anh em** (pre-marshalling), sau khi kiểm tra dữ liệu công khai **[CHƯA]**.

---

## 12. Tạp chí mục tiêu (phải kiểm tra phân hạng hiện hành trên SJR hoặc JCR)

European Journal of Operational Research; INFORMS Journal on Computing; Computers & Operations Research; Transportation Research Part E; IEEE Transactions on Automation Science and Engineering; Computers & Industrial Engineering. Chọn ladder (1 thử sức, 2 mục tiêu, 1 an toàn) sau cổng G1.

---

## 13. Rủi ro

| Rủi ro | Mức | Cách giảm |
|---|---|---|
| Dư địa cải thiện nhỏ trên bài lớn | **Cao** | Cổng G0 sau 2 tuần |
| Trùng công trình chưa đọc toàn văn (Hottung et al. 2020; Q-learning BRP 2025; ICLR 2026 multi-action self-improvement) | Trung bình | Đọc trước tuần 3; không dùng chữ "đầu tiên" khi chưa chắc |
| Nhãn G-Value không phải tối ưu | Trung bình | Kiểm tra độ lệch nhãn trên bài nhỏ có lời giải chính xác |
| Thiếu GPU (máy hiện tại chỉ có CPU) | Trung bình | PG-Beam chạy được trên CPU; huấn luyện G-Value nhẹ; dùng GPU miễn phí (Kaggle) khi cần |
| Mã L2BR gán cứng `'cuda'` | Thấp | Task T0.1 |

---

## 14. Danh sách việc phải kiểm chứng trước khi viết bài

- [ ] Đọc toàn văn Hottung et al. (2020), Q-learning cho BRP (J. Heuristics 2025), ICLR 2026 *Multi-action self-improvement for NCO*, Wen et al. (SSRN)
- [ ] Đối chiếu định nghĩa LB1 và các cận mạnh hơn (Jin & Tanaka 2023; Tanaka & Mizuno 2018)
- [ ] Tải mã Algo-TS (Tanaka) và BBS (Bacci), kiểm tra yêu cầu phần mềm
- [ ] Tải bộ bài lớn của ĐH Vienna và Bacci
- [ ] Kiểm tra phân hạng tạp chí
