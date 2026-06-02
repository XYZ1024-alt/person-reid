def get_person_crops(frame, yolo_model):
    results = yolo_model.predict(frame, classes=[0], conf=0.5)
    height, width = frame.shape[:2]
    person_crops = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))

        if y2 - y1 < 64:
            continue

        crop_img = frame[y1:y2, x1:x2]
        person_crops.append({"coords": (x1, y1, x2, y2), "img": crop_img})

    return person_crops
