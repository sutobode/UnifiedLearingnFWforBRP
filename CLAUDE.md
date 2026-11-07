# CLAUDE.md: hướng dẫn cho mọi coding agent và mọi session của dự án BRP

Dự án nghiên cứu **Block Relocation Problem (BRP)**, mở rộng L2BR (Shin et al., EJOR 2026) bằng beam search có cận dưới và hàm giá trị đồ thị.
File này dành cho **bất kỳ agent hay session nào** làm tiếp dự án. `AGENTS.md` chỉ trỏ về đây.

## 0. Bắt đầu một session mới (đọc theo thứ tự, mất khoảng 5 phút)

1. Đọc file này hết (Mục 1–3 là bắt buộc).
2. Đọc `FINAL_De_xuat_nghien_cuu.md` (mục tiêu, giả thuyết, các cổng quyết định G0–G3) và phần task bạn được giao trong `TASKS_Agent_Coding.md`.
3. Đọc Mục 10 (trạng thái hiện tại) và Mục 9 (bài học đã gặp) của file này, rồi `kaggle/RUNS.md` (các lần chạy Kaggle đã làm).
4. Kiểm tra môi trường:
   ```
   python kaggle/kaggle_cli.py auth     # phải in "credential source" và username, không in khóa
   git status ; git log --oneline -5
   ```
5. Làm đúng **một task**, trên nhánh `task/<mã-task>`. Kết thúc theo Mục 7, **cập nhật Mục 10**.
6. **Tiết kiệm token (Mục 11):** đọc kết quả bằng `python kaggle/digest.py <slug>`, không đọc nguyên file log; chỉ đọc phần task được giao trong `TASKS_Agent_Coding.md`.

Công cụ cần thiết đều nằm **trong repo** (`kaggle/kaggle_cli.py`, `kaggle/jobs/`, `brp/`, ...). Không dựa vào skill hay plugin riêng của một phiên bản agent cụ thể.

## 1. Quy tắc cốt lõi (không thương lượng)

1. **Mọi code được viết và sửa ở máy local.** Không soạn hay sửa logic trực tiếp trên Kaggle (kể cả notebook trong trình duyệt).
2. **Mọi việc cần GPU chạy trên Kaggle, thông qua CLI** (`python -m kaggle ...`, bọc trong `kaggle/kaggle_cli.py`) hoặc SDK. Không chạy GPU ở nơi khác, không SSH hay tunnel vào Kaggle (vi phạm điều khoản).
3. **Việc tính toán nhẹ chạy ở local; việc nặng chạy trên Kaggle** (Mục 2).
4. **Mọi lần chạy, local hay Kaggle, đều phải sinh file log tương ứng** (Mục 3). Một lần chạy không có log thì coi như chưa chạy.
5. Quy trình bắt buộc: **viết code ở local, test ở local (CPU), đẩy lên Kaggle bằng CLI, theo dõi log, tải kết quả về, kiểm tra, ghi vào `kaggle/RUNS.md`**.

## 2. Phân chia máy local và Kaggle

**Local:** Windows 11, **chỉ có CPU** (PyTorch bản CPU, không có `nvidia-smi`, không có g++). Python từ Miniconda. Shell: Git Bash hoặc PowerShell.
**Kaggle:** tài khoản `minhthuong123`, GPU `Tesla T4` (đã xác nhận cấp được), torch `2.11.0+cu130`.

| Chạy ở local | Chạy trên Kaggle (GPU) |
|---|---|
| Viết, sửa, test code (`pytest`) | Suy luận mô hình trên benchmark đầy đủ (CVS, ZQLZ, LL) |
| Debug trên 2–10 bài nhỏ | PG-Beam hoặc best-of-M với M lớn |
| Mô phỏng, cận dưới, bộ giải chính xác cho bài nhỏ | Sinh dữ liệu huấn luyện G-Value |
| Xử lý dữ liệu nhẹ, vẽ hình, dựng bảng từ JSONL | Huấn luyện mọi mô hình (G-Value, chưng cất, SIL) |
| Đọc và tổng hợp log, thống kê kết quả | Lô thực nghiệm nhiều seed, nhiều nhóm bài |

**Quy tắc chọn:** cần GPU, hoặc ước tính quá khoảng **5 phút CPU**, hoặc quét nhiều bài và nhiều seed, thì chạy trên Kaggle. Khi không chắc, đo thử 2–3 bài ở local rồi ngoại suy. Thử ở local luôn ở quy mô nhỏ để bắt lỗi **trước** khi tốn quota GPU. Quota GPU khoảng 30 giờ/tuần (xem kaggle.com/settings); không dùng GPU cho việc thuần CPU (`--accelerator none`).

## 3. Log: mọi job phải có file log để xem

### 3.1 Job chạy ở local
Mỗi lần chạy tạo `logs/<task>/<run_id>/` (`run_id` = giờ + hash cấu hình ngắn):

| File | Nội dung |
|---|---|
| `run.log` | Log đọc được, mỗi dòng có giờ và tiền tố `[PROGRESS]`, `[RESULT]`, `[SUMMARY]`, `[WARN]`, `[ERROR]` |
| `results.jsonl` | Một dòng mỗi bài: `instance_id`, `group`, `method`, `seed`, `obj`, `time_s`, `config_hash`, lời giải (dãy hành động) |
| `summary.json` | Chỉ số tổng hợp cuối cùng |
| `config.yaml` | Bản chụp đúng cấu hình đã dùng |
| `env.json` | Git commit, phiên bản Python và torch, phần cứng, giờ bắt đầu và kết thúc |

### 3.2 Job chạy trên Kaggle: hai lớp log, cả hai bắt buộc
**Phía job** (code của ta, ghi vào `/kaggle/working`, tự được tải về): `run.log`, `results.jsonl`, `summary.json`, `env.json` (và `config.yaml`) đúng định dạng 3.1. Mọi dòng in ra phải đồng thời vào `run.log` và được `flush`. Mẫu cài đặt: `kaggle/jobs/smoke_l2br/main.py` (hàm `log(tag, msg)`).

**Phía máy local** (do `kaggle/kaggle_cli.py` tạo), mỗi lần `run` có **thư mục riêng, không bao giờ ghi đè**:
```
kaggle_out/<slug>/<run_id>/      # run_id = giờ local lúc bắt đầu, ví dụ 20261107_211200
kaggle_out/<slug>/LATEST.txt     # tên thư mục của lần chạy mới nhất
```
| File trong thư mục chạy | Nội dung |
|---|---|
| `live.log` | Log trực tiếp trong lúc job chạy, mỗi dòng có giờ local |
| `status.log` | Các lần đổi trạng thái (queued, running, complete, error) kèm giờ |
| `<slug>.log` | Log đầy đủ do Kaggle tạo (tải về khi kết thúc) |
| `run.log`, `results.jsonl`, `summary.json`, `env.json` | Output của job |

Xem trên trình duyệt: `https://www.kaggle.com/code/minhthuong123/<slug>` (tab Logs).

### 3.3 Quy tắc về log
- **Đầu mỗi job** in: `torch`, `cuda available`, tên GPU, git commit, tham số chính. Job cần GPU mà không có thì **dừng ngay bằng `[ERROR]`** (Kaggle có thể lặng lẽ chạy CPU khi tài khoản chưa xác minh hoặc hết quota).
- **Cuối mỗi job** in một dòng `[SUMMARY] {json}` ra stdout, để vẫn có số liệu nếu việc tải file output lỗi.
- Bọc `main()` trong `try/except`: lỗi phải được ghi vào `run.log` kèm traceback, không chỉ nằm trong log của Kaggle.
- Sau mỗi lần chạy: **đọc log trước khi báo thành công**. `complete` chỉ nghĩa là script đã thoát, chưa phải kết quả đúng.
- Ghi mỗi lần chạy Kaggle vào `kaggle/RUNS.md` (slug, giờ, GPU, kết quả, đường dẫn log), kể cả lần thất bại.
- **Không bao giờ in khóa, token hay nội dung `.env` vào log.**

## 4. Lệnh Kaggle (chạy từ gốc repo)

```
python kaggle/kaggle_cli.py auth                       # nguồn xác thực + kiểm tra kết nối (không in khóa)
python kaggle/kaggle_cli.py run kaggle/jobs/<job> [--accelerator NvidiaTeslaT4|NvidiaTeslaP100|none] [--timeout SEC] [--poll SEC]
python kaggle/kaggle_cli.py status <slug>              # queued / running / complete / error
python kaggle/kaggle_cli.py logs <slug> [-f]           # in log một lần, hoặc theo dõi trực tiếp
python kaggle/kaggle_cli.py fetch <slug>               # tải output của lần chạy mới nhất
python kaggle/digest.py <slug> [--run RUN_ID]          # TÓM TẮT ~15 dòng của một lần chạy (dùng cái này để đọc kết quả)
```
`run` làm: push job, stream log trực tiếp, theo dõi trạng thái, tải output, và thoát với mã lỗi nếu job lỗi. **Chạy nền** (`run_in_background`) cho job dài, và dùng một Monitor lọc các dòng `STATUS`, `[RESULT]`, `[SUMMARY]`, `[ERROR]`, `Traceback` để nhận thông báo, thay vì chờ thủ công.

**Cấu trúc job:** `kaggle/jobs/<slug>/{kernel-metadata.json, main.py}` (riêng tư, GPU, internet bật). `kernels push` **chỉ tải file `code_file`** lên, nên code nhiều file phải đưa vào dataset riêng tư (ví dụ `minhthuong123/brp-code`, cập nhật mỗi lần code đổi, **trước** khi đẩy job) rồi `sys.path.insert` trong `main.py`. Dataset gắn vào job nằm dưới `/kaggle/input/...`; đừng hard-code đường dẫn, hãy duyệt thư mục tìm file đánh dấu.

**Xác thực:** `KAGGLE_API_TOKEN` trong `.env` ở gốc repo (nằm trong `.gitignore`) hoặc biến môi trường của tiến trình. `kaggle_cli.py` tự nạp `.env`. Luôn gọi `python -m kaggle`, không gọi `kaggle` (shim `kaggle.exe` thường không có trong PATH trên Windows).

**Giới hạn của Kaggle cần nhớ:**
- CLI **không có lệnh hủy phiên đang chạy** (chỉ có `kernels delete`, xóa vĩnh viễn). Dừng phiên phải làm ở trình duyệt: kaggle.com/code, **Active Events**, **Stop**.
- `queued` lâu nghĩa là Kaggle chưa cấp GPU, thường do **các phiên GPU khác của cùng tài khoản đang giữ suất** (Mục 9). Lỗi `500` của log stream khi còn `queued` là bình thường, công cụ tự thử lại.
- Mỗi phiên tối đa khoảng 12 giờ: đặt `--timeout` khoảng 9 giờ, ghi checkpoint hoặc JSONL theo từng bài để chạy tiếp được.
- Chỉ file dưới `/kaggle/working` được tải về (khoảng 20 GB).
- Cài thêm gói và `git clone` trong job cần bật internet.

## 5. Quy tắc chạy thí nghiệm

- **Mã gốc L2BR** cố định ở commit `941ce86afc09cf4e1c62400b9c5d17407a82d297`. Không sửa trực tiếp; mọi thay đổi lưu thành patch trong `patches/` (hoặc, với job một file, thay chuỗi có kiểm tra như trong `smoke_l2br/main.py`). Mã gốc có hai điểm không chạy sẵn được: gán cứng `'cuda'` (`decoder.py`, `encoder.py`, `ess.py`, chạy được trên Kaggle nhưng không chạy trên CPU local) và đường dẫn Windows `'datasets\\CVS'` trong `benchmark_loader.py` (hỏng trên Linux/Kaggle).
- **Không chọn siêu tham số trên benchmark** (CVS, ZQLZ, LL); chỉ chọn trên bài sinh ngẫu nhiên. Benchmark chỉ để báo cáo.
- Mọi lời giải phải được **phát lại bằng bộ mô phỏng tham chiếu** (`brp/sim.py`) và đếm lại số lần dời. Số báo cáo lấy từ số đếm lại.
- Mọi phần ngẫu nhiên có seed; ghi seed vào log.
- So sánh với L2BR: cùng máy, cùng ngân sách thời gian; ghi rõ baseline là chạy lại hay lấy từ bài báo.
- Báo cáo **số thật, kể cả kết quả xấu**. Cổng quyết định (G0–G3) không đạt thì nói thẳng và dừng ở cổng đó.

## 6. Bảo mật và an toàn

- `.env` và mọi khóa nằm ngoài git. Không đưa khóa vào code, dataset, log hay commit.
- **Không xóa notebook, dataset hay kernel trên Kaggle** nếu người dùng chưa yêu cầu rõ.
- Dữ liệu benchmark (CVS, ZQLZ, LL) thuộc tác giả gốc; không tái phân phối, chỉ ghi hướng dẫn tải.
- Mã tải từ internet (kể cả L2BR) chỉ chạy sau khi đã xem qua; mô hình `.pt` nạp bằng `weights_only=True`.
- Không dùng `git reset --hard`, `git clean`, hay xóa thư mục kết quả mà chưa kiểm tra nội dung.

## 7. Cách làm việc và bàn giao giữa các session

- Quyết định nào tự làm được thì **tự quyết**, nêu lựa chọn trong báo cáo cuối thay vì hỏi lại. Chỉ hỏi khi thiếu thông tin mà người khác không có (ví dụ khóa, quyền truy cập).
- Một phiên làm **một task** trong `TASKS_Agent_Coding.md`; nhánh `task/<mã-task>`.
- Test trước hoặc cùng code (`pytest`), phải qua hết trước khi báo xong.
- **Kết thúc mỗi task báo**: (1) file đã tạo hoặc sửa, (2) lệnh chạy test và kết quả, (3) số liệu nghiệm thu thực tế so với tiêu chí, (4) điều chưa làm được và lý do.
- **Bàn giao:** trước khi dừng, cập nhật Mục 10 (đánh dấu task, ghi số liệu chính, ghi việc dang dở) và `kaggle/RUNS.md`. Session sau chỉ cần đọc hai chỗ đó để biết đang ở đâu.
- Việc **người dùng phải tự làm** (không giao agent): đọc toàn văn các bài báo liên quan, chứng minh các mệnh đề, kiểm tra phân hạng tạp chí, viết bài, dừng phiên trên trình duyệt Kaggle.
- Khi cần chờ job: dùng chạy nền và Monitor, không dùng `sleep` trong lệnh chạy trước mặt.

## 8. Bố cục repo

```
CLAUDE.md, AGENTS.md          # hướng dẫn cho agent (file này)
FINAL_De_xuat_nghien_cuu.md   # đề xuất chốt
TASKS_Agent_Coding.md         # kế hoạch task
BRP_L2BR_giai_thich_de_hieu.md
1-s2.0-S0377221726007551-main.pdf   # bài gốc (đối chiếu số liệu)
.env                          # khóa Kaggle (gitignore)
kaggle/                       # kaggle_cli.py, jobs/<slug>/, RUNS.md
kaggle_out/<slug>/<run_id>/   # log và kết quả tải về từ Kaggle (gitignore)
logs/<task>/<run_id>/         # log của lần chạy local (gitignore)
results/                      # kết quả đã tổng hợp (gitignore)
third_party/L2BR              # mã gốc, commit cố định (tạo bởi T0.1)
patches/                      # patch lên mã L2BR
brp/  experiments/  tests/    # mã của ta (tạo dần theo TASKS)
```

## 9. Bài học đã gặp (đã kiểm chứng, tránh lặp lại)

| Vấn đề | Nguyên nhân và cách xử lý |
|---|---|
| Job Kaggle nằm `queued` lâu, `500` ở log stream | Các phiên GPU khác của cùng tài khoản đang giữ suất. Dừng chúng ở trình duyệt (Active Events, Stop) thì job được cấp máy ngay trong vài chục giây |
| Không dừng được phiên bằng CLI | CLI không có lệnh hủy. `kernels delete` xóa hẳn notebook và xóa luôn cả kernel của ta nếu bị dùng nhầm |
| `FileNotFoundError: 'datasets\\CVS'` trên Kaggle | `benchmark_loader.py` của L2BR gán cứng đường dẫn Windows. Job thay `'datasets\\CVS'` bằng `'datasets/CVS'` sau khi clone (có ghi `patched` vào log) |
| `mv` thư mục bị `Permission denied` trên Windows | Một tiến trình (hoặc shell có thư mục làm việc nằm trong đó) đang giữ thư mục. Dùng `cp -r`, và đừng `cd` vào `kaggle_out/<slug>` khi có job đang ghi |
| `UnicodeEncodeError` khi in kết quả `kernels list` | Console Windows dùng cp1252. `kaggle_cli.py` đã ép UTF-8; khi tự gọi thì đặt `PYTHONIOENCODING=utf-8` |
| Heredoc Python có đường dẫn Windows báo `SyntaxError (unicode error)` | `\U` trong chuỗi thường. Dùng `Edit` tool hoặc chuỗi raw, không nhét đường dẫn `C:\Users\...` vào chuỗi thường |
| `status` của một kernel trả về `permission 'kernels.get' was denied` | Kernel đã bị xóa hoặc chưa tồn tại (xảy ra khi người dùng xóa sạch). Đẩy lại job là xong |
| `kernels push` chỉ tải một file | Cần dataset riêng tư cho code nhiều file (Mục 4) |
| Biến `KAGGLE_API_TOKEN` | Không có trong biến môi trường hệ thống của máy này; nằm trong `.env` ở gốc repo, `kaggle_cli.py` tự nạp |

## 10. Trạng thái hiện tại (cập nhật cuối mỗi task)

*Cập nhật lần cuối: 07/11/2026.*

| Mục | Trạng thái |
|---|---|
| Đề xuất chốt, kế hoạch task, `CLAUDE.md` | Xong |
| Kết nối Kaggle bằng CLI và `.env` | **Xong, đã kiểm tra** (`auth`, `status`, `run`, `logs`, `fetch`) |
| Kaggle cấp GPU và chạy L2BR | **Xong:** Tesla T4, `cuda available: True` |
| Chuỗi log đầy đủ (live, status, run.log, JSON) | **Xong, đã kiểm tra** trên job `brp-smoke-l2br` |
| Tái lập Bảng 2 trên Kaggle (L2BR Base, CVS uBRP H=3, N=2560) | **Xong:** S=3, 4, 5 cho 4.975, 6.025, 6.850, khớp bài báo (chênh 0,0%). Chi tiết `kaggle/RUNS.md` |
| Phần còn lại của T0.1 (khung dự án local, patch thiết bị, bộ nạp L2BR ở local), T0.2 – T0.5, các giai đoạn sau | **Chưa làm** |
| Cổng G0 (dư địa cải thiện) | **Chưa đo.** Đây là câu hỏi quyết định của toàn dự án; làm T1.x sớm |

**Việc kế tiếp:** T0.1 (khung dự án, patch thiết bị, bộ nạp L2BR ở local), T0.2 (bộ mô phỏng tham chiếu), T0.3 (cận dưới và bộ giải chính xác).

## 11. Tiết kiệm token (token của agent là chi phí chính; tuân thủ mọi mục)

1. **Đọc kết quả bằng digest:** `python kaggle/digest.py <slug>` (khoảng 15 dòng: trạng thái, GPU, kết quả, số lỗi, vài dòng cuối). Chỉ mở log đầy đủ khi digest báo `ERRORS > 0`, và khi đó dùng `grep -n "\[ERROR\]" run.log | head -20` hoặc đọc có `offset`/`limit`, không đọc cả file. Không dán `results.jsonl` hay log nguyên vào ngữ cảnh.
2. **Theo dõi job bằng một Monitor lọc chặt**, không thăm dò lặp: lọc `STATUS|\[RESULT\]|\[SUMMARY\]|\[ERROR\]|Traceback` và loại `500 Server`. Chạy `kaggle_cli.py run` ở chế độ nền. Không `sleep`, không gọi `status` lặp đi lặp lại.
3. **Đọc tài liệu theo phần:** `FINAL_De_xuat_nghien_cuu.md` và `TASKS_Agent_Coding.md` dài. Tìm đúng mục bằng Grep (ví dụ `grep -n "^#### T1.2" TASKS_Agent_Coding.md`) rồi đọc đoạn đó. Chỉ file này (`CLAUDE.md`) cần đọc hết, một lần mỗi session. Không đọc lại PDF bài gốc; số cần đối chiếu đã nằm trong tiêu chí nghiệm thu của task.
4. **Việc xác định thì giao cho script**, không để agent "suy luận": thống kê, dựng bảng, vẽ hình, so với số liệu bài báo. Script in tối đa khoảng 30 dòng.
5. **Giới hạn đầu ra lệnh:** `pytest -q -x --tb=short 2>&1 | tail -25`; thêm `| head -n 30` hoặc `| tail -n 30` cho lệnh in nhiều. Trong job, lọc bỏ thanh tiến trình tqdm (đừng ghi từng dòng tqdm vào `run.log`).
6. **Job tự báo cáo gọn:** cuối job in đúng một dòng `[SUMMARY] {json}` và ghi `summary.json` nhỏ; mỗi bài chỉ một dòng `[RESULT]` hoặc một dòng JSONL.
7. **Không tìm kiếm web hay nghiên cứu lại** những gì đã kiểm chứng (Mục 9 và `FINAL_De_xuat_nghien_cuu.md`). Chỉ tra cứu khi task thật sự cần.
8. **Phân vai model (khuyến nghị):** việc cơ học (chạy và theo dõi job, viết test theo mô tả rõ, tái lập số liệu, dựng bảng) dùng model nhỏ và rẻ. Dùng model lớn cho thiết kế thuật toán (PG-Beam, G-Value), chứng minh, phân tích kết quả và quyết định ở các cổng G0–G3.
9. **Một task một session, bàn giao bằng Mục 10 và `kaggle/RUNS.md`:** session sau không phải khám phá lại. Báo cáo cuối task ngắn: số liệu chính và điều chưa làm, không chép lại nội dung file.
10. **Sửa nhỏ thì dùng `Edit`**, không viết lại cả file; không đọc lại file vừa ghi chỉ để kiểm tra.
