# Nhom10_NhanDienKhuonMat
 Data Mining Final's Project
## HDSD
Đây là một ứng dụng web local cho nhận dạng khuôn mặt. Dưới đây là tóm tắt cách sử dụng:

1. Cài đặt các gói cần thiết: chi tiết ở requirement.txt

2. Chạy ứng dụng: chạy nó bằng lệnh python app.py. Điều này sẽ khởi chạy máy chủ Flask.

3. Truy cập giao diện web: Khi máy chủ Flask đang chạy, bạn có thể truy cập giao diện web bằng cách mở trình duyệt và redirect đến link hiện trên terminal (e.g: http://localhost:5000).

4. Chụp ảnh huấn luyện: Nhập tên của bạn vào trường "Enter your name" và nhấp vào nút "Capture". 200 ảnh khuôn mặt của bạn sẽ được chụp bằng camera thiết bị (hiển thị ở một luồng video hiển thị bên cạnh). Tất cả các ảnh sẽ được gán label là tên bạn đã nhập. Vì vậy, nếu muốn nhận diện nhiều người, bạn hãy lần lượt nhập tên từng người và để họ đứng trước camera.

5. Huấn luyện mô hình: Sau khi chụp ảnh, chọn một mô hình huấn luyện (KNN, Frequent Pattern hoặc Genetic) từ menu dropdown và nhấp vào nút "Recognize Me". Mô hình sẽ được huấn luyện bằng đã chọn bằng các ảnh đã chụp. Cần ít nhất là 200 ảnh để thực hiện huấn luyện (hiện tại, 100 ảnh này có thể có cùng hoặc khác label, sẽ được cập nhật để đảm bảo 100 ảnh/label sau). Lưu ý: các mô hình sẽ có thời gian huấn luyện khác nhau, lâu nhất là Genetic, hãy đợi đến khi ứng dụng thông báo huấn luyện thành công!

6. Thực hiện nhận dạng khuôn mặt: Luồng video từ camera sẽ được hiển thị trên trang web. Nếu nhận dạng khuôn mặt được bật và mô hình đã được huấn luyện, các khuôn mặt được nhận dạng sẽ được đánh dấu bằng một hình chữ nhật và được gắn nhãn với tên tương ứng của chúng.

Đi kèm với ứng dụng là: 
+ Tệp ipynb: chứa các file Jupyter Notebook: FPM.ipynb, Genetic.ipynb, LazyLearner.ipynb biểu diễn các mô hình/thuật toán sử dụng trong ứng dụng, lần lượt là: Frequent Pattern, Di truyền và K-Nearest Neighbor(Lazy Learner). Trong các file này có chi tiết các bước xây dựng và các sử dụng chúng trên tập dữ liệu CMU Face, cũng như cách xử lý dữ liệu trước khi sử dụng. Đồng thời, các mô hình này được đánh giá với 2 đơn vị đo: Accuracy và F1.
+ Tệp CMU_FACE_Data: chứa CMU FACE dataset lấy từ https://archive.ics.uci.edu/ml/datasets/CMU+Face+Images, chi tiết nằm trong báo cáo.
+ Chi tiết về mặt cơ sở lý thuyết nằm ở trong báo cáo đính kèm.
+ Slide thuyết trình.

## Links
Link SourceForge: https://sourceforge.net/projects/nhom10-nhandienkhuonmat/  
Link Github: https://github.com/KomaHomu/Nhom10_NhanDienKhuonMat_
