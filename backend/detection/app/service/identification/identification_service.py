import cv2
import torch
from fastapi import HTTPException
from detection.app.service.model_service import load_my_model
import numpy as np

class IdentificationDetectionSystem:
    """
    Detection system specifically designed for piece identification
    Uses a single color for all detected pieces and focuses on identification rather than validation
    """
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = self.get_device()
        self.model = self.get_my_model()
        
        # Identification-specific settings
        self.identification_color = (0, 255, 255)  # Yellow color for all pieces
        self.font_scale = 0.6
        self.font_thickness = 1
        self.box_thickness = 2

    def get_device(self):
        """Check for GPU availability and return the appropriate device."""
        if torch.cuda.is_available():
            print("Using GPU for identification")
            return torch.device('cuda')
        else:
            print("Using CPU for identification")
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
            model.half()

        return model

    def identify_and_annotate(self, frame):
        """
        Identify all pieces in the frame and annotate them with a single color
        Returns: (annotated_frame, detection_details)
        """
        # Convert the frame to a tensor
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).float().to(self.device)
        frame_tensor /= 255.0  # Normalize
        frame_tensor = frame_tensor.half() if self.device.type == 'cuda' else frame_tensor

        try:
            results = self.model(frame_tensor.unsqueeze(0))[0]  # Batch dim
        except Exception as e:
            print(f"Identification detection failed: {e}")
            return frame, []

        class_names = self.model.names
        detection_details = []

        # Check if there are any detection boxes
        if results.boxes is None or len(results.boxes) == 0:
            return frame, detection_details

        # Get frame dimensions for boundary checks
        frame_height, frame_width = frame.shape[:2]
        print(f"Frame dimensions: {frame_width}x{frame_height}")
        for i, box in enumerate(results.boxes):
            confidence = box.conf.item()
            
            if confidence < self.confidence_threshold:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            class_id = int(box.cls.item())
            piece_label = class_names[class_id]

            # Store detection details
            detection_info = {
                'piece_id': f"piece_{i}",
                'label': piece_label,
                'confidence': confidence,
                'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'center': {'x': (x1 + x2) // 2, 'y': (y1 + y2) // 2}
            }
            detection_details.append(detection_info)

            # Draw bounding box with identification color (yellow)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.identification_color, self.box_thickness)
            
            # ==============================
            # ✅ Dynamic label placement (same style as DetectionSystem)
            # ==============================
            confidence_percent = confidence * 100
            label = f"{piece_label}: {confidence_percent:.1f}%"

            font_scale = 0.5   # same style
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

        return frame, detection_details

    def get_detailed_identification_results(self, frame):
        """
        Get detailed identification results without modifying the frame
        Returns comprehensive information about detected pieces
        """
        # Convert the frame to a tensor
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).float().to(self.device)
        frame_tensor /= 255.0
        frame_tensor = frame_tensor.half() if self.device.type == 'cuda' else frame_tensor

        try:
            results = self.model(frame_tensor.unsqueeze(0))[0]
        except Exception as e:
            print(f"Detailed identification failed: {e}")
            return []

        class_names = self.model.names
        detailed_results = []

        if results.boxes is None or len(results.boxes) == 0:
            return detailed_results

        for i, box in enumerate(results.boxes):
            confidence = box.conf.item()
            
            if confidence < self.confidence_threshold:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            class_id = int(box.cls.item())
            piece_label = class_names[class_id]

            # Calculate additional metrics
            width = x2 - x1
            height = y2 - y1
            area = width * height
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            piece_info = {
                'piece_id': f"piece_{i}",
                'label': piece_label,
                'confidence': round(confidence, 3),
                'confidence_percentage': round(confidence * 100, 1),
                'bounding_box': {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': width, 'height': height
                },
                'area': area,
                'center': {'x': center_x, 'y': center_y},
                'confidence_level': self._get_confidence_level(confidence),
                'class_id': class_id
            }
            
            detailed_results.append(piece_info)

        # Sort by confidence (highest first)
        detailed_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detailed_results

    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        elif confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"

    def set_identification_color(self, color_bgr: tuple):
        """
        Set the color used for identification annotations
        Args:
            color_bgr: Color in BGR format (B, G, R)
        """
        self.identification_color = color_bgr

    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold for identification"""
        if 0.1 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            return True
        return False

    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'model_classes': list(self.model.names.values()) if hasattr(self.model, 'names') else [],
            'identification_color': self.identification_color,
            'total_classes': len(self.model.names) if hasattr(self.model, 'names') else 0
        }

    @staticmethod
    def resize_frame_optimized(frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
        """Optimized frame resizing with better interpolation."""
        if frame.shape[:2] != target_size[::-1]:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        return frame

    def batch_identify(self, frames: list):
        """
        Identify pieces in multiple frames at once
        Args:
            frames: List of numpy arrays (frames)
        Returns:
            List of identification results for each frame
        """
        results = []
        for i, frame in enumerate(frames):
            try:
                annotated_frame, detection_details = self.identify_and_annotate(frame)
                results.append({
                    'frame_index': i,
                    'annotated_frame': annotated_frame,
                    'detection_details': detection_details,
                    'pieces_count': len(detection_details),
                    'success': True
                })
            except Exception as e:
                results.append({
                    'frame_index': i,
                    'annotated_frame': frame,  # Return original frame on error
                    'detection_details': [],
                    'pieces_count': 0,
                    'success': False,
                    'error': str(e)
                })
        
        return results

