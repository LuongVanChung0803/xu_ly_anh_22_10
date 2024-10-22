import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processing App')

        # Layout chính
        main_layout = QVBoxLayout()

        # Khung hiển thị ảnh gốc và ảnh xử lý
        image_layout = QHBoxLayout()
        self.original_label = QLabel('Original Image')
        self.processed_label = QLabel('Processed Image')
        self.original_label.setFixedSize(300, 300)
        self.processed_label.setFixedSize(300, 300)
        self.original_label.setStyleSheet("border: 1px solid black;")
        self.processed_label.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.processed_label)

        # Các button cho từng chức năng
        button_layout = QHBoxLayout()
        load_button = QPushButton('Load Image')
        negative_button = QPushButton('Negative Image')
        contrast_button = QPushButton('Increase Contrast')
        log_button = QPushButton('Log Transform')
        hist_button = QPushButton('Histogram Equalization')

        # Thêm sự kiện cho các button
        load_button.clicked.connect(self.load_image)
        negative_button.clicked.connect(self.negative_image)
        contrast_button.clicked.connect(self.increase_contrast)
        log_button.clicked.connect(self.log_transform)
        hist_button.clicked.connect(self.histogram_equalization)

        button_layout.addWidget(load_button)
        button_layout.addWidget(negative_button)
        button_layout.addWidget(contrast_button)
        button_layout.addWidget(log_button)
        button_layout.addWidget(hist_button)

        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        self.original_image = None
        self.processed_image = None

    # Load ảnh từ file
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image files (*.jpg *.png *.jpeg)')
        if file_name:
            self.original_image = cv2.imread(file_name)
            self.show_image(self.original_image, self.original_label)

    # Hiển thị ảnh
    def show_image(self, img, label):
        qformat = QImage.Format.Format_RGB888 if len(img.shape) == 3 else QImage.Format.Format_Indexed8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, qformat)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    # Hàm tạo ảnh âm tính
    def negative_image(self):
        if self.original_image is not None:
            self.processed_image = 255 - self.original_image
            self.show_image(self.processed_image, self.processed_label)

    # Hàm tăng độ tương phản
    def increase_contrast(self):
        if self.original_image is not None:
            lab = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            self.processed_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            self.show_image(self.processed_image, self.processed_label)

    # Hàm biến đổi log
    def log_transform(self):
        if self.original_image is not None:
            c = 255 / np.log(1 + np.max(self.original_image))
            self.processed_image = c * (np.log(self.original_image + 1))
            self.processed_image = np.array(self.processed_image, dtype=np.uint8)
            self.show_image(self.processed_image, self.processed_label)

    # Hàm cân bằng histogram
    def histogram_equalization(self):
        if self.original_image is not None:
            img_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.equalizeHist(img_gray)
            self.show_image(self.processed_image, self.processed_label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    ex.show()
    sys.exit(app.exec())
