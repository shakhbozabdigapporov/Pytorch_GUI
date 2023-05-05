import cv2
from mhn.model import MHN, simplified_model
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QGridLayout, QPushButton
import numpy as np
import sys
import time


class VideoPlayer(QMainWindow):
    def __init__(self, parent=None, video='1.mp4', **kwargs):
        super(VideoPlayer, self).__init__(parent)
        self.video = video
        self.initialize_model()
        self.create_windows()


    def create_windows(self):
        
        self.width, self.height = 640, 360
        self.counter = 0 #time_out
        self.setWindowTitle("Video Player")

        layout = QGridLayout()

        self.start_button = QPushButton("Start inference", self)
        layout.addWidget(self.start_button, 1, 0)
        self.start_button.clicked.connect(self.button_clicked)

        self.pause_button = QPushButton("Pause inference", self)
        layout.addWidget(self.pause_button, 2, 0)

        self.stop_button = QPushButton("Stop inference", self)
        layout.addWidget(self.stop_button, 3, 0)
        self.stop_button.clicked.connect(self.start_capture)

        self.label1 = QLabel("Detection", self)
        layout.addWidget(self.label1, 0, 3, 1, 2)

        self.label2 = QLabel("Segmentation", self)
        layout.addWidget(self.label2, 0, 5, 1, 2)

        self.video_label1 = QLabel(self)
        layout.addWidget(self.video_label1, 1, 2, 1, 2)

        self.video_label2 = QLabel(self)
        layout.addWidget(self.video_label2, 1, 4, 1, 2)

        self.label3 = QLabel("Joint Prediction", self)
        layout.addWidget(self.label3, 2, 4, 1, 2)

        self.video_label3 = QLabel(self)
        layout.addWidget(self.video_label3, 3, 3, 1, 2)

        self.frame_count = 0
        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

        self.start_capture()

    def start_capture(self):
        self.empty_frame()
        self.video_capture = cv2.VideoCapture(self.video)

    def button_clicked(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.update_frame(self.video_capture))

        self.timer.start(1)

    def update_frame(self, vid_cap):
        ret, frame = vid_cap.read()

        try:
            self.frame_count =+ 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height))  

            draw_boxes, draw_segmentation, draw_all = self.infer(frame)

            draw_boxes = QImage(draw_boxes.data, draw_boxes.shape[1], draw_boxes.shape[0], QImage.Format_RGB888)
            draw_boxes_pixmap = QPixmap.fromImage(draw_boxes)

            draw_segmentation = QImage(draw_segmentation.data, draw_segmentation.shape[1], draw_segmentation.shape[0], QImage.Format_RGB888)
            draw_segmentation_pixmap = QPixmap.fromImage(draw_segmentation)

            draw_all = QImage(draw_all.data, draw_all.shape[1], draw_all.shape[0], QImage.Format_RGB888)
            draw_all = QPixmap.fromImage(draw_all)
            if self.frame_count % 10 == 0:
                time.sleep(10)
            
            self.video_label1.setPixmap(draw_boxes_pixmap)
            self.video_label2.setPixmap(draw_segmentation_pixmap)
            self.video_label3.setPixmap(draw_all)

        except:
            if self.frame_count % 10 == 0:
                time.sleep(10)
            self.empty_frame()

            self.counter += 1
            if self.counter == 100:
                sys.exit(0)

        self.pause_button.clicked.connect(self.timer.stop)

    def initialize_model(self):
        # print("Model Initializing")
        model_path = "mhn/weights/mhn_regnety_384x640.onnx"
        anchor_path = "mhn/weights/mhn_regnety_anchors_384x640.npy"
        simplified_model(model_path) # Remove unused nodes
        self.roadEstimator = MHN(model_path, anchor_path, conf_thres=0.25, iou_thres=0.5)

    def infer(self, frame):
        t1 = time.time()
        seg_map, filtered_boxes, filtered_scores, filtered_indexes = self.roadEstimator(frame)
        draw_segmentation = self.roadEstimator.draw_segmentation(frame)
        draw_boxes = self.roadEstimator.draw_boxes(frame)
        draw_all = self.roadEstimator.draw_2D(frame)
        print(f"{1 / (time.time() - t1)} fps")
        return draw_boxes, draw_segmentation, draw_all

    def empty_frame(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        frame = QPixmap.fromImage(frame)
        self.video_label1.setPixmap(frame)
        self.video_label2.setPixmap(frame)
        self.video_label3.setPixmap(frame)

if __name__ == '__main__':
    app = QApplication([])
    player = VideoPlayer(video='inference/bridge_songdo.mp4')
    player.show()
    app.exec_()
