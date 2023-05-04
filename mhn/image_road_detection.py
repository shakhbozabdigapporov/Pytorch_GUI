import cv2
# from imread_from_url import imread_from_url

from model import MHN, simplified_model

model_path = "./regnety_008_384x640s.onnx"
# model_path = "./hybridnets_384x640.onnx"
anchor_path = "../anchors_384x640.npy"

# Initialize road detector
simplified_model(model_path) # Remove unused nodes
roadEstimator = MHN(model_path, anchor_path, conf_thres=0.5, iou_thres=0.5)

# img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/2021-02-23_Tuesday_16.02.01-16.11.18_UTC-3_Route_S-40_%28Chile%29.webm/1920px--2021-02-23_Tuesday_16.02.01-16.11.18_UTC-3_Route_S-40_%28Chile%29.webm.jpg")
img = cv2.imread("./image.jpg")

# Update road detector
seg_map, filtered_boxes, filtered_scores, filtered_indexes = roadEstimator(img)

combined_img = roadEstimator.draw_2D(img)
cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Road Detections", combined_img)

cv2.imwrite("output.jpg", combined_img)
cv2.waitKey(0)