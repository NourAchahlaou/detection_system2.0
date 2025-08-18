import cv2
import torch
from fastapi import HTTPException
from detection.app.service.model_service import load_my_model
import numpy as np

class DetectionSystem:
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
        self.device = self.get_device()  # Get the device (CPU or GPU)
        self.model = self.get_my_model()  # Load the model once

    def get_device(self):
        """Check for GPU availability and return the appropriate device."""
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device('cuda')
        else:
            print("Using CPU")
            return torch.device('cpu')

    def get_my_model(self):
        """Load the YOLO model based on available device."""
        model = load_my_model()
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found.")

        # Move model to the appropriate device
        model.to(self.device)

        # Convert to half precision if using a GPU
        if self.device.type == 'cuda':
            model.half()  # Convert model to FP16

        return model

    def detect_and_contour(self, frame, target_label):
        # Convert the frame to a tensor
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).float().to(self.device)
        frame_tensor /= 255.0  # Normalize
        frame_tensor = frame_tensor.half() if self.device.type == 'cuda' else frame_tensor

        try:
            results = self.model(frame_tensor.unsqueeze(0))[0]  # Batch dim
        except Exception as e:
            print(f"Detection failed: {e}")
            return frame, False, 0, 0, 0, 0.0

        class_names = self.model.names
        target_color = (0, 255, 0)  # Green
        other_color = (0, 0, 255)  # Red

        detected_target = False
        non_target_count = 0
        total_pieces_detected = 0
        correct_pieces_count = 0
        max_confidence = 0.0

        if results.boxes is None or len(results.boxes) == 0:
            return frame, detected_target, non_target_count, total_pieces_detected, correct_pieces_count, max_confidence

        frame_height, frame_width = frame.shape[:2]

        for box in results.boxes:
            confidence = box.conf.item()
            max_confidence = max(max_confidence, confidence)

            if confidence < self.confidence_threshold:
                continue

            total_pieces_detected += 1

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            class_id = int(box.cls.item())
            detected_label = class_names[class_id]

            if detected_label == target_label:
                color = target_color
                detected_target = True
                correct_pieces_count += 1
            else:
                color = other_color
                non_target_count += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # ==============================
            # ✅ Dynamic label placement
            # ==============================
            confidence_percent = confidence * 100
            label = f"{detected_label}: {confidence_percent:.1f}%"

            font_scale = 0.5
            font_thickness = 1
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Default: place above box
            label_x = x1
            label_y = y1 - 10

            # Prevent vertical overflow
            if label_y - label_height < 0:
                label_y = y1 + label_height + 5
            if label_y > frame_height - 5:
                label_y = frame_height - 5

            # Prevent horizontal overflow
            if label_x < 0:
                label_x = 5
            if label_x + label_width > frame_width:
                label_x = frame_width - label_width - 5

            # Background rectangle
            padding = 3
            bg_x1 = max(0, label_x - padding)
            bg_y1 = max(0, label_y - label_height - padding)
            bg_x2 = min(frame_width, label_x + label_width + padding)
            bg_y2 = min(frame_height, label_y + padding)

            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

            # Draw text
            cv2.putText(frame, label, (label_x, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return frame, detected_target, non_target_count, total_pieces_detected, correct_pieces_count, max_confidence

    @staticmethod
    def resize_frame_optimized(frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
        """Optimized frame resizing with better interpolation."""
        if frame.shape[:2] != target_size[::-1]:  # Check if resize is needed
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        return frame