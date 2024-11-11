import customtkinter as ctk
from typing import Callable

class GradientTitle(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, height=100, fg_color="transparent")
        
        self.title_label = ctk.CTkLabel(
            self,
            text="INNOVATION LAB",
            font=("Helvetica", 48, "bold")
        )
        self.title_label.pack(expand=True)
        self.start_animation()

    def start_animation(self):
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        def update_color(index=0):
            self.title_label.configure(text_color=colors[index])
            self.after(1000, update_color, (index + 1) % len(colors))
        update_color()

class ControlPanel(ctk.CTkFrame):
    def __init__(self, master, camera_var: ctk.StringVar, 
                 camera_values: list, toggle_callback: Callable,
                 roi_callback: Callable):
        super().__init__(master)
        
        self.camera_menu = ctk.CTkOptionMenu(
            self,
            variable=camera_var,
            values=camera_values,
            width=200
        )
        self.camera_menu.pack(side="left", padx=10)
        
        self.start_button = ctk.CTkButton(
            self,
            text="Start Camera",
            command=toggle_callback
        )
        self.start_button.pack(side="left", padx=10)
        
        self.roi_button = ctk.CTkButton(
            self,
            text="Set ROI",
            command=roi_callback
        )
        self.roi_button.pack(side="left", padx=10)

    def update_start_button(self, is_active: bool):
        self.start_button.configure(text="Stop Camera" if is_active else "Start Camera")

    def update_roi_button(self, is_drawing: bool):
        self.roi_button.configure(text="Drawing ROI..." if is_drawing else "Set ROI")