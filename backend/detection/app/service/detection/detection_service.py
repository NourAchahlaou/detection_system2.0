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
            return frame, False, 0, 0, 0, 0.0  # Return zeros for counts and confidence

        class_names = self.model.names
        target_color = (0, 255, 0)  # Green
        other_color = (0, 0, 255)  # Red

        detected_target = False
        non_target_count = 0
        total_pieces_detected = 0
        correct_pieces_count = 0
        max_confidence = 0.0  # Initialize confidence variable

        # Check if there are any detection boxes
        if results.boxes is None or len(results.boxes) == 0:
            return frame, detected_target, non_target_count, total_pieces_detected, correct_pieces_count, max_confidence

        # Get frame dimensions for boundary checks
        frame_height, frame_width = frame.shape[:2]
        
        for box in results.boxes:
            confidence = box.conf.item()
            
            # Update max_confidence regardless of threshold
            max_confidence = max(max_confidence, confidence)
            
            if confidence < self.confidence_threshold:
                continue

            total_pieces_detected += 1  # Count all detections above threshold

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

            # Draw thinner bounding box (thickness = 1 instead of 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            # Format confidence as percentage
            confidence_percent = confidence * 100
            label = f"{detected_label}: {confidence_percent:.1f}%"
            
            # Get label dimensions
            font_scale = 0.5
            font_thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            label_width, label_height = label_size
            
            # Enhanced label positioning with proper boundary checks
            # Default position: above the box
            label_x = x1
            label_y = y1 - 10
            
            # Check if label fits above the box
            if label_y - label_height < 5:  # Not enough space above
                # Try to place below the box
                if y2 + label_height + 15 < frame_height:  # Space below
                    label_y = y2 + label_height + 10
                else:  # Place inside the box at the top
                    label_y = y1 + label_height + 5
            
            # Check horizontal boundaries
            if label_x + label_width > frame_width - 5:  # Too far right
                label_x = max(5, frame_width - label_width - 5)
            elif label_x < 5:  # Too far left
                label_x = 5
            
            # Ensure label_y is within frame bounds
            label_y = max(label_height + 5, min(label_y, frame_height - 5))
            
            # Draw label background rectangle with padding
            padding = 3
            bg_x1 = label_x - padding
            bg_y1 = label_y - label_height - padding
            bg_x2 = label_x + label_width + padding
            bg_y2 = label_y + padding
            
            # Ensure background rectangle is within frame bounds
            bg_x1 = max(0, bg_x1)
            bg_y1 = max(0, bg_y1)
            bg_x2 = min(frame_width, bg_x2)
            bg_y2 = min(frame_height, bg_y2)
            
            # Draw background rectangle
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (label_x, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return frame, detected_target, non_target_count, total_pieces_detected, correct_pieces_count, max_confidence

    @staticmethod
    def resize_frame_optimized(frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
        """Optimized frame resizing with better interpolation."""
        if frame.shape[:2] != target_size[::-1]:  # Check if resize is needed
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        return frame