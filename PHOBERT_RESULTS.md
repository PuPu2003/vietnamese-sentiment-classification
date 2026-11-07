# Kết quả fine-tuning PhoBERT

## Thiết lập thực nghiệm

Mô hình `vinai/phobert-base` được fine-tune toàn bộ encoder cùng classification head Dropout và Linear để phân loại cảm xúc tiếng Việt thành ba lớp, gồm tiêu cực (0), trung lập (1) và tích cực (2).

Dữ liệu thực nghiệm được lấy mẫu và chia phân tầng ở cấp văn bản với seed 42. Tập huấn luyện có 26.208 văn bản, tập xác thực có 3.276 và tập kiểm tra có 3.276. Mọi đoạn của một văn bản thuộc cùng một tập.

| Tham số | Giá trị |
| --- | --- |
| Learning rate | 2e-5 |
| Batch size hiệu dụng | 16 văn bản |
| Optimizer | AdamW |
| Weight decay | 0,01 |
| Max length | 256 token, gồm hai token đặc biệt |
| Overlap | 64 token nội dung |
| Số epoch hoàn tất | 50 |
| Epoch được chọn | 10 |
| Validation Macro-F1 tốt nhất | 0,8565 |

Văn bản được tách từ trước khi token hóa. Mỗi văn bản dài được chia thành các đoạn chồng lấp. Xác suất của mỗi lớp ở cấp văn bản là trung bình xác suất softmax qua các đoạn. Loss huấn luyện là negative log-likelihood của nhãn đúng ở cấp văn bản.

Mô hình đánh giá được chọn theo Macro-F1 trên tập xác thực. Tập kiểm tra được dùng để đánh giá cấu hình đã chọn.

## Kết quả trên tập kiểm tra

| Chỉ số | PhoBERT |
| --- | --- |
| Accuracy | 88,61% |
| Precision macro | 86,20% |
| Recall macro | 87,07% |
| F1 weighted | 88,71% |
| Macro-F1 | 86,61% |
| Log loss | 0,7512 |
| Số dự đoán sai | 373 / 3.276 |
| Tỷ lệ lỗi | 11,39% |

### Kết quả theo lớp

| Lớp | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| Tiêu cực | 90,70% | 91,03% | 90,86% | 814 |
| Trung lập | 74,44% | 79,08% | 76,69% | 674 |
| Tích cực | 93,46% | 91,11% | 92,27% | 1.788 |

### Confusion matrix

Hàng là nhãn thật, cột là nhãn dự đoán.

| Nhãn thật / Nhãn dự đoán | Tiêu cực | Trung lập | Tích cực |
| --- | --- | --- | --- |
| Tiêu cực | 741 | 51 | 22 |
| Trung lập | 49 | 533 | 92 |
| Tích cực | 27 | 132 | 1.629 |

## So sánh với TF-IDF + Logistic Regression

Baseline dùng TF-IDF unigram và bigram với tối đa 60.000 đặc trưng. Logistic Regression được chọn với C = 1 sau khi thử C thuộc {0,1; 1; 10; 100} trên validation. Hai mô hình được đánh giá trên cùng tập kiểm tra.

| Mô hình | Accuracy | Precision macro | Recall macro | F1 weighted | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 83,52% | 81,88% | 78,78% | 83,00% | 79,82% |
| PhoBERT | 88,61% | 86,20% | 87,07% | 88,71% | 86,61% |

PhoBERT tăng 6,79 điểm phần trăm Macro-F1 và giảm số dự đoán sai từ 540 xuống 373 trong lần chia dữ liệu này.

## Nhận xét

- Trung lập là lớp có F1 thấp nhất của PhoBERT. Hai hướng nhầm lẫn nổi bật là tích cực thành trung lập với 132 mẫu và trung lập thành tích cực với 92 mẫu.
- Validation Macro-F1 đạt đỉnh ở epoch 10. Đến epoch 50, train loss giảm về 0,0084 nhưng validation loss tăng lên 1,4171 và validation Macro-F1 còn 0,8258, cho thấy dấu hiệu quá khớp.
- Log loss của PhoBERT cao hơn baseline (0,7512 so với 0,4298). Có 336 dự đoán sai của PhoBERT mang xác suất lớp được chọn từ 0,9 trở lên. Cần phân tích thêm chất lượng xác suất khi sử dụng mô hình.
- Kết quả hiện tại dựa trên một tập lấy mẫu và một seed chia dữ liệu. Chưa có kiểm định ý nghĩa thống kê hoặc đánh giá độc lập theo miền nội dung.
