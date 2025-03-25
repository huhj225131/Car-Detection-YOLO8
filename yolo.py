import numpy as np
import argparse
import cv2
import os
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
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
results = model.predict(args["image"], conf=args["confidence"], iou= args["threshold"])
image = cv2.imread(args["image"])
annotator = Annotator(image)

for result in results:
    for box in result.boxes:
        b = tuple(map(int,box.xyxy[0]))
        c = box.cls
        color = tuple(map(int,colors[int(c),:]))
        annotator.box_label(b, model.names[int(c)], color=color)
    image = annotator.result()

output_path = "result.jpg"
cv2.imwrite(output_path, image)
cv2.imshow('YOLO V8 Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
