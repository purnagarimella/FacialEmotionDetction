from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("best.pt")
# Load the model

results = model.predict(source="0", show=True)