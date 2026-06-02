import os

import cv2

from models.loader import init_reid, init_yolo
from modules.detector import get_person_crops
from modules.reid_engine import compute_similarity, extract_feature


REID_CHECKPOINT_PATH = "outputs/robust_person_reid/best.pth"
TARGET_IMG_PATH = "data/target.jpg"
VIDEO_PATH = "data/video.mp4"


def main():
    yolo_model = init_yolo("yolov8n.pt")
    reid_predictor = init_reid(REID_CHECKPOINT_PATH)

    if not os.path.exists(TARGET_IMG_PATH):
        raise FileNotFoundError(TARGET_IMG_PATH)

    target_img = cv2.imread(TARGET_IMG_PATH)
    target_crops = get_person_crops(target_img, yolo_model)
    if not target_crops:
        raise RuntimeError("No person detected in target image.")

    target_feat = extract_feature(reid_predictor, target_crops[0]["img"])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        person_crops = get_person_crops(frame, yolo_model)
        for person in person_crops:
            x1, y1, x2, y2 = person["coords"]
            feat = extract_feature(reid_predictor, person["img"])
            similarity = compute_similarity(target_feat, feat)

            if similarity > 0.7:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID: Target [{similarity:.2f}]",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)

        cv2.imshow("Pedestrian Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
