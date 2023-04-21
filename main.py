import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QGridLayout
import numpy as np
import sys

class VideoPlayer(QMainWindow):
    def __init__(self, parent=None, video1='1.mp4', video2='1.mp4', video3='1.mp4', **kwargs):
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

        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.update_frame(self.video_capture1, self.video_label1))
        self.timer.timeout.connect(lambda: self.update_frame(self.video_capture2, self.video_label2))
        self.timer.timeout.connect(lambda: self.update_frame(self.video_capture3, self.video_label3))

        self.video_capture1 = cv2.VideoCapture(video1)

        self.video_capture2 = cv2.VideoCapture(video2)

        self.video_capture3 = cv2.VideoCapture(video3)

        self.timer.start(30)

    def update_frame(self, vid_cap, vid_label):
        ret, frame = vid_cap.read()
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height))
        except:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame = cv2.putText(frame, f"{10-int(str(self.counter)[0])}", (300, 200), cv2.FONT_HERSHEY_DUPLEX, 3.0, (125, 246, 55), 3)
            self.counter += 1
            if self.counter == 100:
                sys.exit(0)

        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        vid_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication([])
    player = VideoPlayer(video1='1.mp4', video2='1.mp4', video3='1.mp4')
    player.show()
    app.exec_()
