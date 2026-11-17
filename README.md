# Dự báo volatility: GARCH vs HAR vs LSTM vs Gaussian Process
Dựa trên *Machine Learning in Finance* (Dixon, Halperin, Bilokon): Ch. 3 (GP), Ch. 6 (chuỗi thời gian, GARCH), Ch. 8 (LSTM).
Hai pha thí nghiệm: **(1) dữ liệu Heston tự sinh** (biết phương sai thật) và **(2) dữ liệu thật VN30**.

## Cài đặt & chạy
```bash
pip install -r requirements.txt
python run_synthetic.py --quick                 # chạy thử nhanh (vài phút)
python run_synthetic.py                         # thí nghiệm đầy đủ (4 regime x 2 cỡ train x 3 seed)
python run_real.py --symbols VCB FPT HPG VNINDEX   # can data/raw/<MA>.csv, xem muc "Du lieu thuc"
# không có torch / muốn bỏ LSTM:  thêm  --models naive garch har gp
```
Kết quả: `results/synthetic/*` và `results/real/*` (CSV + hình PNG).

## Dữ liệu thật (tải CSV thủ công)
vnstock bản 6 không còn hàm `Vnstock()` và đọc API của công ty chứng khoán bằng khoá riêng của bạn, nên cách ổn định nhất là tải CSV.
1. Tải lịch sử giá đóng cửa hằng ngày (>= 8-10 năm) của mỗi mã, lưu thành `data/raw/<MÃ>.csv` (ví dụ `data/raw/VCB.csv`, `data/raw/VNINDEX.csv`).
2. Loader tự nhận cột ngày và giá đóng cửa (`date/Ngày/time`, `close/Giá đóng cửa/Price`...), dấu phân cách `, ; tab`, ngày `dd/mm/yyyy`
   hoặc `yyyy-mm-dd`, số kiểu `1,234.5` hoặc `1.234,5`. Nếu không nhận ra cột, nó in danh sách cột để bạn đổi tên thành `date` và `close`.
3. Kiểm tra trước khi chạy: `python check_data.py VCB FPT HPG VNINDEX`
4. Chạy: `python run_real.py --symbols VCB FPT HPG VNINDEX`

Cần giá **đã điều chỉnh** (chia tách, cổ phiếu thưởng). Return vượt ±7.5%/ngày bị loại (biên độ HOSE ±7%) và số ngày bị loại được in ra.
Nếu `check_data.py` báo nhiều outlier, nguồn dữ liệu chưa điều chỉnh giá.

## Cấu trúc
```
volfc/heston.py       mô phỏng Heston (Euler, full truncation) + 4 regime; trả về return và phương sai thật
volfc/features.py     feature nhân quả (log realised variance 1/5/22 ngày) + target h ngày tới
volfc/models.py       Naive, GARCH(1,1)-t, HAR (log-tuyến tính), GP, LSTM
volfc/evaluate.py     walk-forward, RMSE/MAE/QLIKE, kiểm định Diebold-Mariano, calibration
volfc/experiments.py  chạy mọi model trên 1 chuỗi, phân tích theo chế độ calm/turbulent
run_synthetic.py      Pha 1;   run_real.py   Pha 2
```

## Thiết kế thí nghiệm (đáp ứng 4 yêu cầu của thầy)
1. **Bài toán:** tại cuối ngày t, dự báo phương sai trung bình h ngày tới (h=5 mặc định).
   Pha 1: đích "thật" = phương sai tích phân Heston (ẩn trong thực tế); các model chỉ được huấn luyện trên proxy
   quan sát được (mean r² của h ngày tới). Pha 2: chỉ có proxy.
2. **Lý thuyết cần viết:** Heston SDE + rời rạc hóa; stylised facts; ARCH/GARCH, MLE, dự báo nhiều bước;
   HAR; GP regression (kernel, marginal likelihood, predictive distribution); LSTM + QLIKE loss; walk-forward; QLIKE/DM.
3. **Model:** Naive (rolling 22d), GARCH(1,1)-t, HAR, **GP (Matérn-5/2 ARD, hàm trung bình tuyến tính HAR)**, LSTM (QLIKE loss).
4. **Thí nghiệm:** so sánh theo regime Heston, theo cỡ tập train (500/1500), theo giai đoạn calm/turbulent (thực),
   calibration của khoảng tin cậy GP, kiểm định DM so với GARCH.

## Lưu ý phương pháp (nên nêu trong báo cáo)
- **Không rò rỉ dữ liệu:** mỗi lần refit chỉ dùng cặp (t, đích) với t+h ≤ s; dự báo tại t chỉ dùng r[:t+1].
- **QLIKE** bền với nhiễu của proxy (Patton 2011) nên là metric chính; RMSE/MAE tính trên vol quy năm (%).
- **Khoảng tin cậy GP** là khoảng cho *log-proxy* quan sát (đã gồm nhiễu quan sát), nên rộng hơn khoảng cho phương sai ẩn.
- **Chọn thiết kế GP:** bản GP "trơn" (không có trung bình tuyến tính, length-scale tự do) fit nhiễu và kém hẳn HAR trên dữ liệu
  Heston; bản hiện tại (GP trên phần dư HAR, length-scale ≥ 2 trên đặc trưng chuẩn hóa, biên độ ≤ 1) được chọn sau thử nghiệm
  nhỏ trên seed 0–3. Thí nghiệm chính dùng seed 100+ để không trùng. Đây nên được viết thành một phát hiện: *GP cần prior mượt và đủ dữ liệu khi
  tín hiệu yếu, nhiễu cao*.
- Heston là mô hình 1 nhân tố nên GARCH/HAR có lợi thế "đúng họ mô hình"; kết quả thường lệch theo regime (vol-of-vol cao làm đuôi dày).
- Kết quả chạy thật của bạn mới là số liệu cho báo cáo; không dùng số trong các lần chạy thử.


## Thí nghiệm bổ sung (`run_extra.py`)
| Lệnh | Câu hỏi |
|---|---|
| `python run_extra.py --exp leverage-synth` | Trên Heston với ρ = 0 / -0.5 / -0.9, thêm hiệu ứng đòn bẩy (GJR-GARCH, HAR-L, GP-L) có giúp không? (có phương sai thật) |
| `python run_extra.py --exp leverage --symbols VCB FPT HPG VNM VNINDEX` | Tương tự trên dữ liệu thật (GARCH vs GJR, HAR vs HAR-L, GP vs GP-L) |
| `python run_extra.py --exp horizon --symbols ...` | Thứ hạng model có đổi khi h = 1, 5, 22? |
| `python run_extra.py --exp learning --symbols ...` | Learning curve: cùng tập test, cửa sổ train 250/500/1000/1500/toàn bộ |

Thêm `--quick` để chạy thử; kết quả ở `results/extra/<tên thí nghiệm>/` (`runs.csv` + hình).
`HAR-L`/`GP-L` thêm 2 đặc trưng: return có dấu của ngày t và trung bình 5 ngày của min(r, 0). LSTM vốn đã nhận return có dấu.
Trong `learning`, tập test cố định (bắt đầu tại max(cửa sổ lớn nhất, 50% dữ liệu)) nên các cửa sổ so sánh được với nhau.

## Mở rộng nếu còn thời gian
GP định giá quyền chọn/Greeks trên chính bộ mô phỏng Heston (Ch. 3 của sách); GARCH-LSTM lai; sparse GP; horizon h=1 và 22.