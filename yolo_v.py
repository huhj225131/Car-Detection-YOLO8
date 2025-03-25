import numpy as np
import argparse
import cv2
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to input image")
ap.add_argument("-y", "--yolo", required=True,help="path to YOLO dir")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold",type=float, default=0.3,
                help="threshold for non-maxima suppression")
args = vars(ap.parse_args())

#load model
modelPath = os.path.sep.join([args["yolo"], "best.pt"])
model = YOLO(modelPath)

# Create random color for class name
LABELS = model.names
np.random.seed(22)
colors = np.random.randint(0,225,size=(len(LABELS), 3), dtype="uint8")
# results = model.predict(args["video"], conf=args["confidence"], iou= args["threshold"])
# image = cv2.imread(args["video"])
# annotator = Annotator(image)
cap = cv2.VideoCapture(args["video"])
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video= cv2.VideoWriter('result.mp4',fourcc, frame_fps,(frame_width, frame_height) )
while True:
    ret, image = cap.read()
    if not ret :
        break
    results = model.predict(image, conf=args["confidence"], iou= args["threshold"])
    for result in results:
        annotator = Annotator(image)
        for box in result.boxes:
            b = tuple(map(int,box.xyxy[0]))
            c = box.cls
            color = tuple(map(int,colors[int(c),:]))
            annotator.box_label(b,label=f"{model.names[int(c)]}-{round(box.conf.item(),2)} ", color=color,)
        image = annotator.result()
        video.write(image)
    # cv2.imshow('YOLO V8 Detection', image)
video.release()
cap.release()
cv2.destroyAllWindows()
print("Video đã lưu thành công: result.mp4")
