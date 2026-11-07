# Bản viết lại dễ hiểu: *A unified learning framework for the block relocation problem: A new performance baseline*

*Tác giả: Woo-Jin Shin và cộng sự, European Journal of Operational Research (2026). Mã nguồn: github.com/operagang/L2BR*

## 1. Tóm tắt trong 30 giây

Bài toán: một bãi container có nhiều cột chồng. Muốn lấy container ở đáy thì phải dời các container đè lên nó. Mục tiêu là lấy hết container theo thứ tự định trước, với **số lần dời ít nhất**.

Các tác giả đề xuất **L2BR**, một mô hình học máy (neural network) học cách giải bài toán này. Ý tưởng chính là học theo hai giai đoạn:
1. **Bắt chước lời giải tối ưu** của thuật toán chính xác trên các bài toán nhỏ, để có một "nền tảng" tốt.
2. **Tự cải thiện**: mô hình tự tạo nhiều lời giải cho bài toán lớn, chọn lời giải tốt nhất, rồi học lại chính lời giải đó.

Kết quả: L2BR tìm được lời giải tối ưu với nhiều bài nhỏ và vừa, đặt được kỷ lục mới cho nhiều bài lớn, chỉ trong vài giây. Nó cũng vượt các phương pháp học tăng cường (RL) trước đây.

## 2. Bài toán: dời container để lấy đúng cái cần

Hình ví dụ trong bài (Fig. 1) có 3 cột, mỗi cột tối đa 4 tầng, tổng 9 container. Số càng nhỏ thì càng cần lấy sớm.

- Container số **1** nằm trên cùng cột 1, lấy được ngay.
- Container số **2** nằm dưới container số 4, phải dời 4 đi chỗ khác (cột 3) trước.
- Lấy 2, 3, 4, 5 theo thứ tự.
- Container số **9** nằm đè lên 6 và 8, phải dời 9 sang cột 1 mới lấy được phần còn lại.

Tổng cộng: **4 lần dời**. Các lần dời này không tạo ra giá trị trực tiếp, chỉ làm tốn công, nên mục tiêu là giảm chúng xuống mức thấp nhất.

Có hai biến thể của bài toán:
- **rBRP (hạn chế):** chỉ được dời container đang chắn container cần lấy.
- **uBRP (không hạn chế):** được dời bất kỳ container nào ở đỉnh cột sang cột khác. Bài toán này khó hơn nhưng cho lời giải tốt hơn.

Thứ tự lấy container được biết trước. Container mới không đến trong lúc đang lấy.

## 3. Vì sao bài toán này khó cho máy học?

Các phương pháp học tăng cường (RL) thường xây lời giải từng bước, và mỗi bước là một hành động. Với bài toán giao hàng hay lập lịch, số bước luôn hữu hạn. Nhưng với uBRP, có thể xảy ra hai tình huống xấu:

- **Lặp vòng (cycle):** dời A sang B, rồi dời B về lại, rồi lặp mãi. Lời giải không bao giờ kết thúc.
- **Lời giải quá dài:** một số lời giải chạy rất lâu, làm tín hiệu học bị nhiễu, tốn bộ nhớ và thời gian.

Các RL trước đây gần như chỉ làm được với rBRP vì rBRP không có vòng lặp. Với uBRP, chưa có phương pháp học nào được công bố làm baseline trước đó.

## 4. Ý tưởng cốt lõi của L2BR

Hãy so sánh với cách học của con người. Một học sinh có thể:

1. **Chép bài giải mẫu của thầy** (giai đoạn IL). Cách này ổn định, không bị lạc vào vòng lặp, nhưng chỉ làm được với bài nhỏ, vì bài giải tối ưu cho bài lớn quá tốn công tìm.
2. **Tự làm nhiều bài lớn, chọn bài làm tốt nhất rồi học lại đúng bài đó** (giai đoạn SIL). Cách này không cần thầy nữa, nên mở rộng được cho bài lớn.

### Giai đoạn 1: Bắt chước (Imitation Learning, IL)
- Dùng thuật toán chính xác (solver) để tạo tập dữ liệu gồm các cặp (trạng thái bãi, hành động tối ưu).
- Huấn luyện mô hình để dự đoán hành động giống như solver.
- Vì dữ liệu đều là lời giải hoàn chỉnh, mô hình không bị kẹt trong vòng lặp.

### Giai đoạn 2: Tự cải thiện (Self-Improvement Learning, SIL)
- Với mỗi bài toán ngẫu nhiên lớn, mô hình tạo nhiều lời giải (rollout).
- Chỉ giữ lời giải hoàn chỉnh **tốt nhất**.
- Huấn luyện lại mô hình trên chính những lời giải tốt nhất đó.
- Lặp lại cho đến khi ổn định.

**Mẹo "dừng sớm" (Early-Stop Sampling, ESS):** nếu một lời giải đang chạy đã dài hơn lời giải tốt nhất hiện có, thì dừng ngay. Nó không thể thắng, nên không cần đi tiếp. Mẹo này giảm thời gian tính toán khoảng 80% ở bãi 10×10 (từ 11,59 giây xuống 2,49 giây cho mỗi lần tính trung bình).

**Vì sao SIL tốt hơn RL ở đây:** RL học từ nhiều quỹ đạo, kể cả quỹ đạo lặp vòng hay quá dài, nên tín hiệu học yếu. SIL chỉ cần **một** lời giải hoàn chỉnh tốt nhất cho mỗi bài, nên học ổn định và tiết kiệm bộ nhớ hơn.

Ở thí nghiệm bãi 10×10, SIL hội tụ nhanh hơn RL khoảng **5,2 lần** và đạt lời giải tốt hơn.

## 5. Mô hình mạng nơron (đơn giản hóa)

Mô hình gồm hai phần:

**Bộ mã hóa (Encoder):** đọc trạng thái bãi.
- Mỗi cột được đọc bằng một mạng LSTM để hiểu thứ tự các container trong cột.
- Các cột được "nói chuyện" với nhau qua các lớp Transformer, để biết cột nào đang ảnh hưởng cột nào.
- Thiết kế này không phụ thuộc số cột hay chiều cao, nên một mô hình huấn luyện trên bãi nhỏ có thể dùng cho bãi lớn hơn.

**Bộ giải mã (Decoder):** với mỗi cặp (cột nguồn, cột đích) có thể dời, tính một điểm; các cặp không hợp lệ bị loại. Sau đó chọn hành động theo xác suất.

Mô hình không cần luật thủ công hay đặc trưng do chuyên gia thiết kế.

## 6. Kết quả chính

Bài báo thử trên ba bộ dữ liệu chuẩn: **CVS** (21 nhóm cấu hình), **ZQLZ** (125 cấu hình, có bài rất đặc, chỉ còn 1 ô trống), và **LL** (bài lớn, có tới 720 container). Có ba phiên bản:
- **L2BR-B:** chỉ học IL trên bài nhỏ.
- **L2BR-L:** học thêm SIL trên bài lớn.
- **L2BR-T:** học thêm SIL cho đúng quy mô đích.

**Các kết quả đáng chú ý:**

| Trường hợp | Kết quả |
|---|---|
| uBRP, bài nhỏ (đến khoảng 5×5) | L2BR-B tìm được lời giải tối ưu cho mọi bài |
| uBRP, bài 10 tầng (H=10) | L2BR-T giảm khoảng **11%** (S=6) và **14%** (S=10) số lần dời so với Algo-TS-TL, một heuristic mạnh |
| uBRP, bài lớn | Đặt kỷ lục mới cho nhiều quy mô, trong vòng 30 phút so với Algo-TS |
| uBRP, bài lớn kiểu LL | Đặt baseline đầu tiên, vì trước đây chưa có ai công bố kết quả này |
| rBRP, so với RL trước đây (DAM, MDDAM, CRP-DRL) | L2BR tốt hơn ở mọi quy mô so sánh được |
| rBRP, so với BBS (heuristic mạnh nhất) | Chênh lệch trung bình chỉ khoảng 0,04% đến 0,2% |
| Tốc độ | Với hầu hết bài, lời giải có trong khoảng **1 giây** |

Ví dụ với bãi 5×5 uBRP, RL chỉ đạt khoảng 100 lần dời, trong khi IL và SIL đạt khoảng 17 lần.

## 7. Điểm mạnh và hạn chế

**Điểm mạnh:**
- Giải quyết được cả hai biến thể rBRP và uBRP bằng một khung thống nhất.
- Xử lý được vấn đề vòng lặp và lời giải quá dài, vốn làm RL trước đây yếu.
- Một mô hình có thể dùng cho nhiều kích thước bãi khác nhau.
- Nhanh: lời giải trong vài giây cho bài lớn.

**Hạn chế:**
- **IL phụ thuộc vào solver chính xác** để tạo dữ liệu, mà solver này chậm với bài lớn. Vì vậy phải dùng IL trên bài nhỏ rồi mới dùng SIL cho bài lớn.
- **Chi phí huấn luyện lớn:** ví dụ, một epoch SIL cho bài LL với 160 cột có thể mất hơn 7 giờ (Bảng A.12).
- **Phụ thuộc GPU:** mô hình chạy chủ yếu trên GPU.
- **Trường hợp ZQLZ rBRP** vẫn là nơi BBS cho kết quả tốt hơn một chút. Đây là ngoại lệ duy nhất mà bài báo thừa nhận.
- Một số so sánh với các phương pháp RL khác không tái lập được đầy đủ, vì mã nguồn của họ không công khai. Các tác giả dùng kết quả báo cáo trong bài gốc.

## 8. Kết luận và hướng phát triển

Bài báo cho thấy một cách học đơn giản nhưng hiệu quả: **học từ lời giải tốt của chính mình** thay vì học từ phần thưởng qua từng bước. Cách này tránh được điểm yếu của RL trong các bài có khả năng lặp vô hạn.

Các tác giả đề xuất mở rộng sang các biến thể khác của bài toán như BRP có container trùng độ ưu tiên (duplicated BRP) và BRP trực tuyến (online BRP, khi container đến trong lúc đang vận hành).

---

**Thuật ngữ nhanh:**
- **Block / container:** đơn vị hàng được xếp chồng.
- **Stack / cột:** một chồng container.
- **Tier / tầng:** vị trí theo chiều cao trong cột.
- **Relocation / dời:** chuyển một container sang chỗ khác để lấy container bên dưới.
- **IL (Imitation Learning):** học bắt chước lời giải có sẵn.
- **SIL (Self-Improvement Learning):** học từ lời giải tốt nhất do chính mô hình tạo ra.
- **ESS:** dừng sớm các lời giải không có triển vọng.
- **Heuristic:** thuật toán tìm lời giải tốt nhưng không đảm bảo tối ưu.
