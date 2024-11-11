import cv2
from typing import List

class CameraManager:
    @staticmethod
    def detect_cameras() -> List[str]:
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(f"Camera {i}")
                cap.release()
        return available_cameras if available_cameras else ["No cameras found"]