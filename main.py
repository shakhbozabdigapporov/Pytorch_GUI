import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QGridLayout
import numpy as np
import sys

class VideoPlayer(QMainWindow):
    def __init__(self, parent=None, video1='1.mp4'):
        super(VideoPlayer, self).__init__(parent)

        self.width, self.height = 640, 360
        self.counter = 0 #time_out
        self.setWindowTitle("Video Player")

        layout = QGridLayout()

        self.label1 = QLabel("Detection", self)
        layout.addWidget(self.label1, 0, 1, 1, 2)

        self.label2 = QLabel("Segmentation", self)
        layout.addWidget(self.label2, 0, 3, 1, 2)

        self.video_label1 = QLabel(self)
        layout.addWidget(self.video_label1, 1, 0, 1, 2)

        self.video_label2 = QLabel(self)
        layout.addWidget(self.video_label2, 1, 2, 1, 2)

        self.label3 = QLabel("Joint Prediction", self)
        layout.addWidget(self.label3, 2, 2, 1, 2)

        self.video_label3 = QLabel(self)
        layout.addWidget(self.video_label3, 3, 1, 1, 2)

        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(lambda: self.update_frame(self.video_capture1, self.video_label1))

        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(lambda: self.update_frame(self.video_capture2, self.video_label2))

        self.timer3 = QTimer(self)
        self.timer3.timeout.connect(lambda: self.update_frame(self.video_capture3, self.video_label3))

        video_file1 = video1
        self.video_capture1 = cv2.VideoCapture(video_file1)

        video_file2 = '1.mp4'
        self.video_capture2 = cv2.VideoCapture(video_file2)

        video_file3 = '1.mp4'
        self.video_capture3 = cv2.VideoCapture(video_file3)

        self.timer1.start(30)
        self.timer2.start(30)
        self.timer3.start(30)

    def update_frame(self, vid_cap, vid_label):
        ret, frame = vid_cap.read()
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height))
        except:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.counter += 1
            if self.counter == 100:
                sys.exit(0)

        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        vid_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication([])
    player = VideoPlayer()
    player.show()
    app.exec_()
