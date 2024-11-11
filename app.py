import customtkinter as ctk
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
from typing import List, Tuple
import os

class InnovationLabCounter(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.camera_active = False
        self.roi_points: List[Tuple[int, int]] = []
        self.drawing_roi = False
        self.current_count = 0
        self.gradient_index = 0
        self.colors = ['#FF0000', '#FF00FF', '#0000FF', '#00FF00']  # Vibrant colors
        self.camera = None
        self.model = YOLO('yolov8x.pt')  # Using larger YOLO model

        # Configure window
        self.setup_window()
        self.setup_ui()
        
    def setup_window(self):
        self.title("Innovation Lab")
        self.attributes('-fullscreen', True)
        ctk.set_appearance_mode("dark")
        
    def setup_ui(self):
        # Main container
        self.main_frame = ctk.CTkFrame(self, fg_color="black")
        self.main_frame.pack(fill="both", expand=True)
        
        # Title with gradient animation
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="INNOVATION LAB",
            font=("Helvetica", 96, "bold"),
            text_color=self.colors[0]
        )
        self.title_label.pack(pady=40)
        
        # Counter display
        self.counter_label = ctk.CTkLabel(
            self.main_frame,
            text="Waiting to detect...",
            font=("Helvetica", 48)
        )
        self.counter_label.pack(pady=20)

        # Video frame
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(pady=20, expand=True)

        # Settings button (gear icon)
        gear_icon = "⚙️"
        self.settings_button = ctk.CTkButton(
            self.main_frame,
            text=gear_icon,
            width=50,
            height=50,
            fg_color="transparent",
            hover_color="gray20",
            command=self.show_settings
        )
        self.settings_button.place(relx=0.98, rely=0.98, anchor="se")

        # Start animation and initialize
        self.animate_title()
        self.initialize_camera()

    def show_settings(self):
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        
        # Camera selection
        cameras = self.get_available_cameras()
        camera_var = ctk.StringVar(value=str(self.current_camera))
        camera_label = ctk.CTkLabel(settings_window, text="Select Camera:")
        camera_label.pack(pady=10)
        camera_menu = ctk.CTkOptionMenu(
            settings_window,
            values=[str(i) for i in range(len(cameras))],
            variable=camera_var,
            command=self.change_camera
        )
        camera_menu.pack(pady=10)
        
        # ROI button
        roi_button = ctk.CTkButton(
            settings_window,
            text="Set ROI",
            command=self.start_roi_selection
        )
        roi_button.pack(pady=10)

    def animate_title(self):
        """Animate the title with gradient colors"""
        self.gradient_index = (self.gradient_index + 1) % len(self.colors)
        self.title_label.configure(text_color=self.colors[self.gradient_index])
        self.after(1000, self.animate_title)

    def get_available_cameras(self):
        """Detect available cameras"""
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def initialize_camera(self):
        """Initialize the first available camera"""
        cameras = self.get_available_cameras()
        if cameras:
            self.current_camera = cameras[0]
            self.start_camera()

    def change_camera(self, camera_index):
        """Change to selected camera"""
        self.current_camera = int(camera_index)
        if self.camera_active:
            self.stop_camera()
            self.start_camera()

    def start_camera(self):
        """Start camera feed"""
        self.camera = cv2.VideoCapture(self.current_camera)
        self.camera_active = True
        self.update_frame()

    def stop_camera(self):
        """Stop camera feed"""
        if self.camera is not None:
            self.camera.release()
        self.camera_active = False

    def start_roi_selection(self):
        """Start ROI selection mode"""
        self.roi_points = []
        self.drawing_roi = True

    def update_frame(self):
        """Update video frame and detect people"""
        if self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                # Detect people
                results = self.model(frame, classes=[0])  # class 0 is person
                count = 0
                
                # Draw detections and count people
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Check if person is inside ROI (if ROI exists)
                        if len(self.roi_points) == 4:
                            person_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            if not cv2.pointPolygonTest(np.array(self.roi_points), person_center, False) >= 0:
                                continue
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        count += 1

                # Draw ROI if exists
                if len(self.roi_points) == 4:
                    cv2.polylines(frame, [np.array(self.roi_points)], True, (0, 255, 0), 2)

                # Update counter text
                counter_text = f"{count} person currently enjoying the lab" if count == 1 else f"{count} people currently enjoying the lab"
                self.counter_label.configure(text=counter_text)

                # Convert frame to PhotoImage and display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo

            self.after(10, self.update_frame)

    def run(self):
        """Start the application"""
        self.mainloop()

if __name__ == "__main__":
    app = InnovationLabCounter()
    app.run()