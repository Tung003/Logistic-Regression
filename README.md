Logistic Regression

1.  Giới thiệu

Hồi quy logistic là một thuật toán học máy có giám sát được sử dụng cho
các vấn đề phân loại. Không giống như hồi quy tuyến tính dự đoán các giá
trị liên tục, nó dự đoán xác suất đầu vào thuộc về một lớp cụ thể. Nó
được sử dụng để phân loại nhị phân, trong đó đầu ra có thể là một trong
hai loại có thể có như Có/ Không, Đúng/ Sai hoặc 0/ 1. Nó sử dụng hàm
sigmoid để chuyển đổi đầu vào thành giá trị xác suất giữa 0 và 1. Hãy
xem những điều cơ bản về hồi quy logistic và các khái niệm cốt lõi của
nó.

2.  Cách hoạt động

ví dụ: Hãy xem xét vấn đề phát hiện xem một người có bị bệnh tim hay
không bị bệnh tim.

![idea](outputs/media/image1.png){width="5.583333333333333in"
height="2.78125in"}

Đầu vào có thể biểu diễn như vector
${x = \lbrack x}_{0},x_{1},x_{2},..,x_{n}$\], với mỗi thành phần $x_{i}$
tương ứng với một đặc trưng trong một mẫu bệnh án.

\- Để có mô hình đáp ứng được điều này ta thử quay lại bài toán hồi quy
tuyến tính $f(x_{n}) = \omega_{n}^{T}.x_{n} + b_{n}$.

![ex](outputs/media/image2.png){width="5.763888888888889in"
height="2.454861111111111in"}

$$f(x_{n}) = 29.79,\  \in R\ $$

vậy làm cách nào để kết quả đầu ra có giá trị rời rạc?

3.  Giải pháp

Giải pháp cho vấn đề trên ta có hàm Sigmoid đây là một giải pháp để tạo
ra giá trị rời rạc, xem đồ thị của hàm Sigmoid.
$\widehat{y\ } = g(f(x_{n})) = \frac{1}{1 + e^{- (f(x_{n}))}},\widehat{y\ },g(f(x_{n})) \in \lbrack 0,1\rbrack$

![Screenshot from 2025-06-11
23-13-10](outputs/media/image3.png){width="5.759722222222222in"
height="3.254861111111111in"}

\- Xét một ví dụ xác suất thống kê cho việc tung đồng xu. Tung đồng xu 5
lần được 3 lần mặt ngửa (đặt là 1) và 2 lần mặt sấp (đặt là 0).

![Screenshot from 2025-06-16
17-07-23](outputs/media/image4.png){width="5.7625in"
height="2.890277777777778in"}

Để dễ tính ta lấy log hai vế.

![Screenshot from 2025-06-16
17-06-22](outputs/media/image5.png){width="5.767361111111111in"
height="1.4506944444444445in"}

Mục tiêu của bài toán là tối đa hoá dự đoán càng gần nhãn thực tế thì
càng tốt: nhãn thực tế y=1, dự đoán $\widehat{y\ }$=0.999 kết luận tốt
nhưng nếu nhãn thực tế y=1, dự đoán $\widehat{y\ }$=0.001 kết luận xấu.

\- giờ hãy xem ý nghĩa thự tế của biểu thức (\*).

nếu y=1, (\*)\<=\> $log(P(y|x) = log({\widehat{y\ }}^{\ }$);

+------------+------------+---------------+---------------------------+
| y          | $${\       | $             | ý nghĩa                   |
|            | widehat{y\ | log({\widehat |                           |
| (thực tế)  |  }}^{\ }$$ | {y\ }}^{\ }$) |                           |
|            |            |               |                           |
|            | (dự đoán)  |               |                           |
+------------+------------+---------------+---------------------------+
| 1          | 0.999      | -0.0004345    | tốt ($log(P(y|x))$ gần 0) |
+------------+------------+---------------+---------------------------+
| 1          | 0.9        | -0.0457574    | khá tốt                   |
+------------+------------+---------------+---------------------------+
| 1          | 0.4        | -0.39794      | không tốt                 |
+------------+------------+---------------+---------------------------+
| 1          | 0.001      | -3            | tệ                        |
+------------+------------+---------------+---------------------------+

kết luận: khi mô hình dự đoán ${\widehat{y\ }}^{\ }$gần 1 thì
$log(P(y|x))$ gần 0 =\> mô hình tốt, ngược lại ${\widehat{y\ }}^{\ }$gần
0 thì $log(P(y|x))$ âm rất lớn =\> mô hình tệ.

nếu y=0, (\*)\<=\> $log({1 - \widehat{y\ }}^{\ }$)

+------------+------------+---------------+---------------------------+
| y          | $${\       | $log(         | ý nghĩa                   |
|            | widehat{y\ | {1 - \widehat |                           |
| (thực tế)  |  }}^{\ }$$ | {y\ }}^{\ }$) |                           |
|            |            |               |                           |
|            | (dự đoán)  |               |                           |
+------------+------------+---------------+---------------------------+
| 0          | 0.999      | -3            | tệ                        |
+------------+------------+---------------+---------------------------+
| 0          | 0.9        | -1            | không tốt                 |
+------------+------------+---------------+---------------------------+
| 0          | 0.4        | -0.221848     | trung bình                |
+------------+------------+---------------+---------------------------+
| 0          | 0.001      | -0.0004345    | tốt ($log(P(y|x))$ gần 0) |
+------------+------------+---------------+---------------------------+

![Screenshot from 2025-06-16
17-05-12](outputs/media/image6.png){width="5.821527777777778in"
height="3.671527777777778in"}

Vậy tiếp theo ta cần phải làm gì khi đã có được hàm Loss.

Để có được mô hình dự đoán tốt thì ta cần cực tiểu hoá hàm Loss

4.  Thuật toán

Trong tập huấn luyện có n mẫu.

![Screenshot from 2025-06-16
17-03-55](outputs/media/image7.png){width="5.764583333333333in"
height="0.8833333333333333in"}

\- đạo hàm riêng Loss với w.

![Screenshot from 2025-06-16
17-02-09](outputs/media/image8.png){width="5.761805555555555in"
height="5.8493055555555555in"}

\- đạo hàm riêng Loss với w.

![Screenshot from 2025-06-16
17-02-58](outputs/media/image9.png){width="5.763194444444444in"
height="0.9173611111111111in"}

5.  ![](outputs/media/image10.png){width="4.409448818897638e-3in"
    height="2.8346456692913387e-3in"}![](outputs/media/image11.png){width="4.409448818897638e-3in"
    height="2.8346456692913387e-3in"}Kiểm thử

    chạy thử thuật toán

    Nhận thấy accuracy của mô hình khá cao tuy nhiên mục tiêu dự đoán
    những người bị bệnh thật sự rất kém Recall= 7%

    -thấy rằng Model accuracy: 0.837 dễ nhầm lẫn thành model dự đoán tốt
    tuy nhiên Recall= 0.066 lại rất thấp

    -\>dẫn đến việc dự đoán người thực sự mắc bệnh thì model dự đoán lại
    rất tệ nhìn vào confusion_matrix có đến

    122 người thực sự mắc bệnh mà chỉ dự đoán được 8 người

    **Nguyên nhân**

    do bộ dữ liệu bị mát cân bằng giữa class bị bệnh (y=1) và class
    không bị bệnh (y=0), nên mô hình tập trung

    vào học các đặc trưng của các samples không bị bệnh (y=0).

    **Giải pháp**

    có 2 cách điển hình.

    \- cách 1: Cân bằng dữ liệu huấn luyện nhân bản dữ liệu những class
    bị bệnh để model có thể học được

    nhiều hơn về class bị bệnh(y=1).

    +kết quả:

    Before SMOTE:

    Number of samples in each class: Counter({0: 2489, 1: 435})

    After SMOTE:

    Number of samples in each class: Counter({0: 2489, 1: 2489})

    \- cách 2: Ta sẽ cần tìm điểm cân bằng sao cho model có thể dự đoán
    được nhiều người bị bệnh thật sự

    nhưng cũng không được cảnh báo nhầm nhiều người từ không bị bệnh
    thành bị bệnh. bằng cách tìm điểm ngưỡng

    để cân bằng F1_score. tuy nhiên nếu muốn model không bỏ sót người bị
    bệnh thực sự thì ta lại phải đánh đổi

    việc model cảnh báo nhầm những người không bị bệnh nhưng lại được dự
    đoán là bị bệnh.

**\
**

**Tìm điểm tối ưu.**

![Find_threshold_best](outputs/media/image12.png){width="4.773611111111111in"
height="2.6069444444444443in"}

Điểm tối ưu cho F1 và Recall

Cho mô hình thay đổi ngưỡng từ 0.1 đến 0.9 để tìm ngưỡng tối ưu cho 2
metrics F1 score và Recall.

+kết quả: khi áp dụng cả cách 1 và 2 ta tìm được điểm ngưỡng tốt nhất
như sau.

Threshold best F1-score: 0.46 with F1 = 0.378

Threshold best Recall: 0.32 with Recall = 0.795

**Tinh chỉnh**

Điều chỉnh tham số mô hình cụ thể là 2 ngưỡng mới tìm được ở trên để xem
xét mô hình

![](outputs/media/image13.png){width="4.409448818897638e-3in"
height="2.6771653543307085e-3in"}![](outputs/media/image14.png){width="4.409448818897638e-3in"
height="2.6771653543307085e-3in"}

Best F1 Best Recall

![](outputs/media/image15.png){width="4.566929133858267e-3in"
height="3.425196850393701e-3in"}![](outputs/media/image16.png){width="4.566929133858267e-3in"
height="3.425196850393701e-3in"}

Best F1 confusion matrix Best Recall confusion matrix
