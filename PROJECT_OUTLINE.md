# Đề cương báo cáo khoa học

## Tên đề tài

**Phân loại cảm xúc văn bản tiếng Việt bằng fine-tuning PhoBERT và so sánh với TF-IDF kết hợp Logistic Regression**

## Tóm tắt

Viết khoảng 150–250 từ, trình bày bài toán phân loại cảm xúc tiếng Việt, đặc điểm dữ liệu nghiên cứu, phương pháp sử dụng, thiết kế đánh giá và kết quả chính.

Nêu phương pháp fine-tuning PhoBERT với chia đoạn chồng lấp, tổng hợp xác suất ở cấp văn bản và so sánh với TF-IDF kết hợp Logistic Regression. Kết thúc bằng mức cải thiện Macro-F1 quan sát được và giới hạn chính của nghiên cứu.

**Từ khóa:** phân loại cảm xúc, tiếng Việt, PhoBERT, học chuyển giao, TF-IDF, Logistic Regression, văn bản dài.

## 1. Giới thiệu

### 1.1. Bối cảnh nghiên cứu

Trình bày nhu cầu khai thác cảm xúc từ đánh giá sản phẩm, phản hồi dịch vụ, bình luận và phản hồi trong giáo dục. Giải thích ý nghĩa của việc tự động phân loại một lượng lớn văn bản tiếng Việt.

Nêu các đặc điểm gây khó khăn, gồm từ nhiều tiếng, cách viết không thống nhất, biểu đạt cảm xúc phụ thuộc ngữ cảnh, nhận xét nhiều chiều và mất cân bằng nhãn.

### 1.2. Vấn đề nghiên cứu

Phát biểu bài toán dự đoán một nhãn cảm xúc cho toàn bộ văn bản. Văn bản có thể chứa nhiều câu và vượt giới hạn đầu vào của mô hình. Làm rõ yêu cầu tổng hợp thông tin từ nhiều đoạn để đưa ra kết quả chung.

### 1.3. Mục tiêu nghiên cứu

- Chuẩn hóa dữ liệu cảm xúc tiếng Việt từ nhiều nguồn.
- Áp dụng PhoBERT cho phân loại ba lớp tiêu cực, trung lập và tích cực.
- Xử lý văn bản dài bằng chia đoạn có chồng lấp và tổng hợp xác suất.
- So sánh với TF-IDF kết hợp Logistic Regression trên cùng tập kiểm tra.
- Phân tích hiệu suất từng lớp, các dạng nhầm lẫn và xu hướng quá khớp.

### 1.4. Câu hỏi nghiên cứu

1. PhoBERT đạt hiệu suất như thế nào so với TF-IDF kết hợp Logistic Regression trên dữ liệu nghiên cứu?
2. Mức chênh lệch hiệu suất giữa các lớp tiêu cực, trung lập và tích cực là bao nhiêu?
3. Hiệu suất trên tập xác thực thay đổi như thế nào trong quá trình fine-tuning?
4. Những nhầm lẫn nào xuất hiện phổ biến và xác suất dự đoán thể hiện điều gì về các trường hợp sai?

### 1.5. Phạm vi và đóng góp

Xác định phạm vi là phân loại ba lớp ở cấp văn bản. Trình bày đóng góp ở mức xây dựng quy trình xử lý, ứng dụng PhoBERT với cơ chế tổng hợp dự đoán từ các đoạn và đánh giá thực nghiệm trên bộ dữ liệu nghiên cứu.

Giới thiệu ngắn bố cục các phần còn lại của báo cáo.

## 2. Cơ sở lý thuyết và nghiên cứu liên quan

### 2.1. Phân loại cảm xúc văn bản tiếng Việt

Định nghĩa phân loại cảm xúc có giám sát và ý nghĩa của ba lớp. Phân tích những trường hợp ranh giới giữa trung lập với có cảm xúc, cũng như văn bản chứa đồng thời đánh giá tích cực và tiêu cực.

Tổng quan các hướng tiếp cận dựa trên đặc trưng thống kê và mô hình ngôn ngữ tiền huấn luyện.

### 2.2. TF-IDF và Logistic Regression

- Giải thích tần suất từ, tần suất tài liệu nghịch đảo và biểu diễn TF-IDF.
- Mô tả unigram, bigram và vai trò của cụm từ trong biểu đạt cảm xúc.
- Trình bày Logistic Regression cho phân loại nhiều lớp và regularization.
- Thảo luận khả năng biểu diễn thông tin từ vựng và giới hạn khi cần hiểu ngữ cảnh.

### 2.3. Transformer và PhoBERT

- Trình bày self-attention và biểu diễn từ phụ thuộc ngữ cảnh.
- Giới thiệu PhoBERT và cơ sở lựa chọn cho văn bản tiếng Việt.
- Giải thích tách từ tiếng Việt, subword tokenization và token đặc biệt.
- Mô tả fine-tuning toàn bộ encoder cùng bộ phân loại cho nhiệm vụ đích.

### 2.4. Xử lý văn bản dài

Giới thiệu giới hạn độ dài đầu vào, cơ chế chia đoạn có chồng lấp và cách tổng hợp dự đoán từ nhiều đoạn. Làm rõ vai trò của overlap đối với việc giữ ngữ cảnh ở ranh giới đoạn.

### 2.5. Nghiên cứu liên quan

Tổng hợp các công trình gốc về PhoBERT, VnCoreNLP và phân loại cảm xúc tiếng Việt. Với mỗi nhóm nghiên cứu, trình bày dữ liệu, phương pháp, tiêu chí đánh giá và mối liên hệ với thiết kế của đề tài.

Phần này cần hoàn thiện trích dẫn học thuật trong bản báo cáo đầy đủ.

## 3. Dữ liệu và tiền xử lý

### 3.1. Mô tả bộ dữ liệu

Trình bày các nguồn gồm đánh giá nhà hàng, phản hồi sinh viên, đánh giá sản phẩm và bình luận tiếng Việt. Với mỗi nguồn, cung cấp xuất xứ, miền nội dung, số mẫu, nhãn gốc và điều kiện sử dụng.

Giải thích cách kết hợp tiêu đề với nội dung để hình thành văn bản đầy đủ đối với nguồn có nhiều trường nội dung.

### 3.2. Chuẩn hóa nhãn

| Nhãn | Giá trị | Ý nghĩa |
| --- | --- | --- |
| Tiêu cực | 0 | Đánh giá hoặc thái độ tiêu cực |
| Trung lập | 1 | Thông tin hoặc nhận xét được gán nhãn trung lập |
| Tích cực | 2 | Đánh giá hoặc thái độ tích cực |

Mô tả quy tắc quy đổi nhãn của từng nguồn về hệ ba lớp. Giải thích cơ sở quy đổi thang đánh giá sao và xác minh ánh xạ nhãn số theo tài liệu nguồn.

### 3.3. Làm sạch và tách từ

- Chuẩn hóa Unicode và khoảng trắng.
- Xử lý bản ghi thiếu nội dung hoặc nhãn không hợp lệ.
- Loại văn bản trùng cùng nhãn và xử lý văn bản trùng có nhãn mâu thuẫn.
- Tách từ bằng VnCoreNLP để phù hợp với đầu vào PhoBERT.
- Minh họa bằng một số văn bản trước và sau tiền xử lý.

### 3.4. Lựa chọn dữ liệu thực nghiệm

Mô tả quá trình tạo tập lấy mẫu dùng trong nghiên cứu, gồm tiêu chí lựa chọn, quy mô, phân bố nhãn, phương pháp lấy mẫu và seed.

Mục tiêu huấn luyện hiện tại là nhãn cứng. Nếu trình bày các điểm xác suất bổ sung trong dữ liệu, cần giải thích nguồn và vai trò của chúng đối với quá trình chuẩn bị dữ liệu.

### 3.5. Thống kê và chia dữ liệu

Trình bày số lượng mẫu trước và sau làm sạch, phân bố nhãn, phân bố độ dài theo token và tỷ lệ văn bản vượt giới hạn đầu vào.

| Tập | Tỷ lệ | Số văn bản trong thực nghiệm |
| --- | --- | --- |
| Huấn luyện | 80% | 26.208 |
| Xác thực | 10% | 3.276 |
| Kiểm tra | 10% | 3.276 |

Chia phân tầng theo nhãn với seed 42 ở cấp văn bản trước khi tạo các đoạn. Mọi đoạn của một văn bản thuộc cùng một tập.

Hai mô hình sử dụng cùng tập kiểm tra. Tập xác thực phục vụ lựa chọn tham số và mô hình tốt nhất. Tập kiểm tra phục vụ đánh giá cấu hình đã chọn.

## 4. Phương pháp nghiên cứu

### 4.1. Tổng quan phương pháp

Trình bày sơ đồ từ văn bản đã tiền xử lý đến dự đoán cảm xúc, với hai nhánh TF-IDF kết hợp Logistic Regression và PhoBERT với chia đoạn chồng lấp.

### 4.2. Phương pháp cơ sở

Mô tả biểu diễn TF-IDF unigram và bigram, giới hạn 60.000 đặc trưng và Logistic Regression. Vocabulary cùng trọng số IDF được học trên tập huấn luyện. Hệ số regularization được lựa chọn trên tập xác thực.

### 4.3. Kiến trúc phân loại sử dụng PhoBERT

1. Tokenizer tạo Input IDs và Attention Mask từ văn bản đã tách từ.
2. PhoBERT Encoder trích xuất contextual representations cho từng đoạn.
3. Biểu diễn tại token đầu chuỗi được đưa vào bộ phân loại.
4. Dropout và Linear Layer tạo logits của ba lớp.
5. Softmax chuyển logits thành xác suất cảm xúc của từng đoạn.
6. Tổng hợp xác suất các đoạn để dự đoán cho toàn bộ văn bản.

Toàn bộ encoder và bộ phân loại được fine-tune trên tập huấn luyện.

### 4.4. Chia đoạn có chồng lấp

Trình bày giới hạn 256 token, gồm hai token đặc biệt. Mỗi đoạn chứa tối đa 254 token nội dung, overlap 64 token và bước trượt 190 token.

| Đoạn | Khoảng token minh họa cho văn bản 500 token |
| --- | --- |
| 1 | 1–254 |
| 2 | 191–444 |
| 3 | 381–500 |

Ranh giới đoạn được xác định theo vị trí token. Thảo luận tác động của việc lặp thông tin ở vùng overlap và của đoạn cuối ngắn hơn.

### 4.5. Tổng hợp dự đoán ở cấp văn bản

Với văn bản có K đoạn, xác suất của mỗi lớp được lấy trung bình đều qua các đoạn.

$$
p(c \mid x) = \frac{1}{K}\sum_{k=1}^{K}p_k(c)
$$

Nhãn văn bản là lớp có xác suất tổng hợp cao nhất.

$$
\hat{y} = \arg\max_c p(c \mid x)
$$

### 4.6. Hàm mất mát

Trình bày negative log-likelihood ở cấp văn bản.

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log p(y_i \mid x_i)
$$

Trong đó, y_i là nhãn đúng và xác suất được tính sau khi tổng hợp các đoạn. Mỗi văn bản đóng góp ngang nhau vào loss trung bình.

## 5. Thiết kế thực nghiệm

### 5.1. Cấu hình huấn luyện

| Thành phần | Cấu hình PhoBERT đã thực nghiệm |
| --- | --- |
| Mô hình tiền huấn luyện | vinai/phobert-base |
| Learning rate | 2e-5 |
| Batch size hiệu dụng | 16 văn bản |
| Max length | 256 token |
| Overlap | 64 token |
| Optimizer | AdamW |
| Weight decay | 0,01 |
| Số epoch | 50 |
| Tiêu chí lựa chọn mô hình | Macro-F1 cao nhất trên tập xác thực |

Baseline thử C thuộc tập {0,1; 1; 10; 100}, lựa chọn C = 1 theo Macro-F1 xác thực.

### 5.2. Theo dõi quá trình học và lựa chọn mô hình

Mô tả cách theo dõi Training Loss, Validation Loss và Validation Macro-F1 theo epoch. Giải thích tiêu chí lựa chọn mô hình tốt nhất và vai trò của early stopping.

Thực nghiệm hiện có hoàn tất 50 epoch với early stopping tắt. Mô hình đánh giá cuối được chọn tại epoch 10, đạt Validation Macro-F1 bằng 0,8565.

### 5.3. Thực nghiệm đối chứng đề xuất

| Yếu tố | Cấu hình so sánh | Mục tiêu phân tích |
| --- | --- | --- |
| Learning rate | 1e-5, 2e-5, 3e-5 | Độ ổn định và hiệu suất fine-tuning |
| Batch size | 8, 16 | Khả năng tổng quát hóa |
| Max length | 128, 256 | Mức độ giữ thông tin mỗi đoạn |
| Overlap | 0, 32, 64 | Ảnh hưởng của ngữ cảnh chồng lấp |
| Xử lý văn bản dài | Cắt phần đầu, chia đoạn chồng lấp | Hiệu quả riêng của cơ chế chia đoạn |

Các cấu hình này là đề xuất bổ sung. Phần kết quả chỉ trình bày những cấu hình có số liệu thực tế. Nếu tiếp tục tối ưu sau khi đã xem test hiện tại, cần thiết kế phần đánh giá giữ lại phù hợp để bảo đảm tính độc lập của kết quả cuối.

### 5.4. Tiêu chí đánh giá

Trình bày Accuracy, Precision, Recall, F1-score, Macro-F1 và confusion matrix. Precision và Recall tổng hợp sử dụng macro average. Báo cáo thêm weighted F1 để phản ánh phân bố mẫu.

$$
\operatorname{MacroF1} = \frac{F1_{negative}+F1_{neutral}+F1_{positive}}{3}
$$

Ưu tiên Macro-F1 do dữ liệu mất cân bằng lớp. Sử dụng log loss như chỉ số bổ sung để phân tích chất lượng xác suất.

## 6. Kết quả thực nghiệm

### 6.1. So sánh hiệu suất tổng thể

Trình bày kết quả trên cùng 3.276 văn bản kiểm tra.

| Mô hình | Accuracy | Precision macro | Recall macro | F1 weighted | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 83,52% | 81,88% | 78,78% | 83,00% | 79,82% |
| PhoBERT | 88,61% | 86,20% | 87,07% | 88,71% | 86,61% |

Phân tích mức tăng 6,79 điểm phần trăm Macro-F1 của PhoBERT trong thiết lập đã đánh giá.

### 6.2. Kết quả theo từng lớp

| Lớp | F1 của TF-IDF + Logistic Regression | F1 của PhoBERT |
| --- | --- | --- |
| Tiêu cực | 83,64% | 90,86% |
| Trung lập | 67,12% | 76,69% |
| Tích cực | 88,68% | 92,27% |

Thảo luận hiệu suất thấp hơn ở lớp trung lập. Phân tích Precision, Recall và support từng lớp để giải thích sự đánh đổi giữa khả năng phát hiện và số dự đoán sai.

### 6.3. Confusion matrix

Trình bày confusion matrix dạng số lượng mẫu và dạng tỷ lệ chuẩn hóa theo nhãn thật cho cả hai mô hình.

Đối chiếu các nhầm lẫn nổi bật, gồm 132 văn bản tích cực bị PhoBERT dự đoán thành trung lập và 195 văn bản trung lập bị baseline dự đoán thành tích cực.

### 6.4. Learning curves

Trình bày đồ thị Training Loss cùng Validation Loss theo epoch và đồ thị Validation Macro-F1 theo epoch.

- Training Loss giảm về 0,0084 tại epoch 50.
- Validation Loss tăng lên 1,4171 tại epoch 50.
- Validation Macro-F1 đạt đỉnh 0,8565 tại epoch 10 và còn 0,8258 tại epoch 50.

Phân tích dấu hiệu quá khớp và giải thích việc chọn mô hình bằng tập xác thực.

### 6.5. Phân tích lỗi dự đoán

PhoBERT dự đoán sai 373 văn bản, baseline sai 540 văn bản. Cả hai cùng sai 231 văn bản. PhoBERT đúng trong khi baseline sai 309 văn bản, trường hợp ngược lại có 142 văn bản.

Chọn ví dụ thực tế để phân tích nhận xét nhiều chiều, phủ định, ngữ cảnh không đầy đủ và trường hợp nhãn cần rà soát. Phân nhóm nguyên nhân sau khi đọc, đối chiếu từng mẫu.

### 6.6. Chất lượng xác suất

Log loss của PhoBERT là 0,7512, so với 0,4298 của baseline. Trong các dự đoán sai, PhoBERT có 336 mẫu với xác suất lớp được chọn từ 0,9 trở lên, baseline có 20 mẫu.

Thảo luận sự khác biệt giữa chất lượng nhãn dự đoán và chất lượng xác suất. Các quan sát này hỗ trợ phân tích xu hướng tự tin cao khi dự đoán sai. Đánh giá calibration đầy đủ cần thêm phép đo hoặc đồ thị tương ứng.

## 7. Thảo luận

### 7.1. Trả lời câu hỏi nghiên cứu

Tổng hợp câu trả lời dựa trên bảng so sánh, kết quả từng lớp, learning curves và phân tích lỗi.

### 7.2. Ý nghĩa của kết quả

Phân tích lợi ích quan sát được của PhoBERT trong thiết lập nghiên cứu. Liên hệ hiệu suất lớp trung lập với đặc điểm dữ liệu và ví dụ lỗi đã rà soát. Phân biệt quan sát thực nghiệm với giả thuyết giải thích cần kiểm chứng.

### 7.3. Giới hạn và độ tin cậy

- Kết quả dựa trên một lần chia dữ liệu và một tập lấy mẫu.
- Chưa có đánh giá qua nhiều seed hoặc kiểm định ý nghĩa thống kê.
- Tính nhất quán của nhãn giữa các nguồn cần được xác minh.
- Chưa có đối chứng đầy đủ về ảnh hưởng riêng của overlap và phép tổng hợp xác suất.
- Khả năng tổng quát hóa sang miền mới cần được đánh giá độc lập.
- Quá trình lấy mẫu cần được mô tả đủ để tái lập và đánh giá tính đại diện.

## 8. Kết luận và hướng nghiên cứu tiếp theo

### 8.1. Kết luận

Tóm tắt bài toán, phương pháp xử lý văn bản dài, thiết lập so sánh và kết quả chính. Kết luận về mức cải thiện Macro-F1 trên tập kiểm tra đã sử dụng, đồng thời nêu hạn chế về quá khớp và chất lượng xác suất.

### 8.2. Hướng nghiên cứu tiếp theo

- Rà soát và chuẩn hóa nhãn giữa các nguồn.
- Thực nghiệm đối chứng về max length, overlap và chiến lược tổng hợp dự đoán.
- Đánh giá qua nhiều seed, báo cáo trung bình và độ lệch chuẩn.
- Nghiên cứu hiệu chỉnh xác suất bằng dữ liệu xác thực.
- Đánh giá theo miền nội dung và độ dài văn bản.

## Tài liệu tham khảo

Hoàn thiện danh mục tài liệu được trích dẫn, tập trung vào công trình gốc về PhoBERT, VnCoreNLP, Transformer, phân loại cảm xúc tiếng Việt và các nguồn dữ liệu nghiên cứu.

## Danh mục bảng và hình dự kiến

| Loại | Nội dung |
| --- | --- |
| Bảng 1 | Đặc điểm và nguồn gốc dữ liệu |
| Bảng 2 | Quy tắc chuẩn hóa nhãn |
| Bảng 3 | Thống kê dữ liệu và phân bố các tập |
| Bảng 4 | Cấu hình huấn luyện và lựa chọn tham số |
| Bảng 5 | So sánh hiệu suất tổng thể |
| Bảng 6 | Precision, Recall và F1-score từng lớp |
| Bảng 7 | Ví dụ dự đoán sai và phân tích nguyên nhân |
| Hình 1 | Sơ đồ phương pháp nghiên cứu |
| Hình 2 | Phân bố nhãn và độ dài văn bản |
| Hình 3 | Kiến trúc phân loại sử dụng PhoBERT |
| Hình 4 | Chia đoạn có overlap và tổng hợp xác suất |
| Hình 5 | Confusion matrix hai mô hình |
| Hình 6 | Learning curves |
