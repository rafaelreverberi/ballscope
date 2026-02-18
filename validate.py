from ultralytics import YOLO

model = YOLO("models/football-ball-detection.pt")

print("KLASSEN IM MODELL:")
print(model.names)