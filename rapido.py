from ultralytics import YOLO
import os

curdir = os.path.dirname(__file__)

model = YOLO(curdir + "/weights/nano.pt")
model.track(0, show=True)
