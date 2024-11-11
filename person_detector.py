import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

class PersonDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def process_frame(self, frame: np.ndarray, roi_points: List[Tuple[int, int]]) -> Tuple[np.ndarray, int]:
        frame_copy = frame.copy()
        person_count = 0

        results = self.model(frame_copy, classes=[0])  # class 0 is person

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check if person is inside ROI (if ROI exists)
                if len(roi_points) == 4:
                    person_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    if not cv2.pointPolygonTest(np.array(roi_points), person_center, False) >= 0:
                        continue
                
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                person_count += 1

        # Draw ROI if exists
        if len(roi_points) == 4:
            cv2.polylines(frame_copy, [np.array(roi_points)], True, (0, 255, 0), 2)

        return frame_copy, person_count