from ultralytics import YOLO
import os

curdir = os.path.dirname(__file__)

model = YOLO(curdir + "/weights/nano.pt")
model.track(curdir + "/videos/exemplo.mp4", show=True)
