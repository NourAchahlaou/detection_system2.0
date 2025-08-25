import cv2
import torch
from fastapi import HTTPException
from detection.app.service.model_service import load_my_model, get_model_for_group, extract_group_from_piece_label
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DetectionSystem:
    def __init__(self, confidence_threshold=0.5, target_piece_label=None, 
                 crop_size=800, overlap_ratio=0.5, nms_threshold=0.4):
        """
        Enhanced detection system with sliding window support.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            target_piece_label: Target piece to detect
            crop_size: Size of sliding window crops (should match training crop size)
            overlap_ratio: Overlap between sliding windows (0.5 = 50% overlap)
            nms_threshold: Non-Maximum Suppression threshold for duplicate removal
        """
        self.confidence_threshold = confidence_threshold
        self.target_piece_label = target_piece_label
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.nms_threshold = nms_threshold
        self.device = self.get_device()
        self.model = self.get_my_model()
        self.current_group = None

    def get_device(self):
        """Check for GPU availability and return the appropriate device."""
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device('cuda')
        else:
            print("Using CPU")
            return torch.device('cpu')

    def get_my_model(self):
        """Load the YOLO model based on available device and target piece."""
        model = load_my_model(self.target_piece_label)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found.")

        model.to(self.device)

        if self.device.type == 'cuda':
            model.half()

        if self.target_piece_label:
            self.current_group = extract_group_from_piece_label(self.target_piece_label)
            print(f"Loaded model for group: {self.current_group}")

        return model

    def switch_model_for_piece(self, new_target_piece_label):
        """Switch to a different model if needed."""
        new_group = extract_group_from_piece_label(new_target_piece_label)
        
        if new_group != self.current_group:
            print(f"Switching from group {self.current_group} to group {new_group}")
            
            new_model = get_model_for_group(new_group)
            if new_model is None:
                print(f"Could not load model for group {new_group}, keeping current model")
                return False
            
            self.model = new_model
            self.model.to(self.device)
            if self.device.type == 'cuda':
                self.model.half()
            
            self.current_group = new_group
            self.target_piece_label = new_target_piece_label
            print(f"Successfully switched to model for group {new_group}")
            return True
        else:
            self.target_piece_label = new_target_piece_label
            return True

    def generate_sliding_windows(self, frame_height: int, frame_width: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate sliding window coordinates for the frame.
        
        Returns:
            List of (x1, y1, x2, y2) coordinates for each window
        """
        windows = []
        step_size = int(self.crop_size * (1 - self.overlap_ratio))
        
        for y in range(0, frame_height - self.crop_size + 1, step_size):
            for x in range(0, frame_width - self.crop_size + 1, step_size):
                x2 = min(x + self.crop_size, frame_width)
                y2 = min(y + self.crop_size, frame_height)
                
                # Adjust x1, y1 to maintain crop_size when possible
                x1 = max(0, x2 - self.crop_size)
                y1 = max(0, y2 - self.crop_size)
                
                windows.append((x1, y1, x2, y2))
        
        # Add edge windows if the frame doesn't divide evenly
        # Right edge
        if frame_width > self.crop_size:
            x1 = frame_width - self.crop_size
            for y in range(0, frame_height - self.crop_size + 1, step_size):
                y2 = min(y + self.crop_size, frame_height)
                y1 = max(0, y2 - self.crop_size)
                if (x1, y1, frame_width, y2) not in windows:
                    windows.append((x1, y1, frame_width, y2))
        
        # Bottom edge
        if frame_height > self.crop_size:
            y1 = frame_height - self.crop_size
            for x in range(0, frame_width - self.crop_size + 1, step_size):
                x2 = min(x + self.crop_size, frame_width)
                x1 = max(0, x2 - self.crop_size)
                if (x1, y1, x2, frame_height) not in windows:
                    windows.append((x1, y1, x2, frame_height))
        
        # Bottom-right corner
        if frame_width > self.crop_size and frame_height > self.crop_size:
            corner_window = (frame_width - self.crop_size, frame_height - self.crop_size, 
                           frame_width, frame_height)
            if corner_window not in windows:
                windows.append(corner_window)
        
        return windows

    def detect_in_crop(self, crop: np.ndarray, crop_offset: Tuple[int, int]) -> List[Dict]:
        """
        Detect objects in a single crop and convert coordinates back to original frame.
        
        Args:
            crop: Image crop to process
            crop_offset: (offset_x, offset_y) of the crop in the original frame
            
        Returns:
            List of detections with global coordinates
        """
        detections = []
        offset_x, offset_y = crop_offset
        
        try:
            # Run detection on the crop
            results = self.model.predict(
                crop,
                conf=self.confidence_threshold,
                device=self.device,
                imgsz=512,  # This should match your training image size
                verbose=False
            )
            
            if not results or len(results) == 0:
                return detections
                
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return detections

            class_names = self.model.names
            
            for box in result.boxes:
                confidence = box.conf.item()
                
                if confidence < self.confidence_threshold:
                    continue

                # Get bounding box coordinates in crop space
                xyxy = box.xyxy[0].cpu().numpy()
                x1_crop, y1_crop, x2_crop, y2_crop = map(int, xyxy)
                
                # Convert to global frame coordinates
                x1_global = x1_crop + offset_x
                y1_global = y1_crop + offset_y
                x2_global = x2_crop + offset_x
                y2_global = y2_crop + offset_y
                
                class_id = int(box.cls.item())
                detected_label = class_names[class_id]
                
                detections.append({
                    'x1': x1_global,
                    'y1': y1_global,
                    'x2': x2_global,
                    'y2': y2_global,
                    'confidence': confidence,
                    'class_id': class_id,
                    'label': detected_label,
                    'crop_offset': crop_offset
                })
                
        except Exception as e:
            logger.error(f"Detection failed for crop at {crop_offset}: {e}")
            
        return detections

    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of detections after NMS
        """
        if len(detections) == 0:
            return detections
        
        # Convert detections to format suitable for NMS
        boxes = []
        scores = []
        class_ids = []
        
        for det in detections:
            boxes.append([det['x1'], det['y1'], det['x2'], det['y2']])
            scores.append(det['confidence'])
            class_ids.append(det['class_id'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        # Apply NMS per class
        keep_indices = []
        unique_classes = np.unique(class_ids)
        
        for class_id in unique_classes:
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = np.where(class_mask)[0]
            
            if len(class_boxes) == 0:
                continue
                
            # Apply NMS using OpenCV
            indices = cv2.dnn.NMSBoxes(
                class_boxes.tolist(),
                class_scores.tolist(),
                self.confidence_threshold,
                self.nms_threshold
            )
            
            if len(indices) > 0:
                if isinstance(indices[0], list):
                    indices = [idx[0] for idx in indices]
                keep_indices.extend(class_indices[indices])
        
        # Return filtered detections
        return [detections[i] for i in keep_indices]

    def detect_with_sliding_window(self, frame: np.ndarray, target_label: Optional[str] = None) -> Tuple[np.ndarray, bool, int, int, int, float]:
        """
        Detect objects using sliding window approach.
        
        Args:
            frame: Input frame
            target_label: Target piece label to search for
            
        Returns:
            Tuple of (annotated_frame, detected_target, non_target_count, 
                     total_pieces_detected, correct_pieces_count, max_confidence)
        """
        frame_height, frame_width = frame.shape[:2]
        
        # If frame is smaller than crop size, use original detection method
        if frame_height < self.crop_size or frame_width < self.crop_size:
            logger.info(f"Frame too small ({frame_width}x{frame_height}), using direct detection")
            return self.detect_and_contour(frame, target_label)
        
        logger.info(f"Using sliding window detection on {frame_width}x{frame_height} frame")
        
        # Generate sliding windows
        windows = self.generate_sliding_windows(frame_height, frame_width)
        logger.info(f"Generated {len(windows)} sliding windows")
        
        # Collect all detections
        all_detections = []
        
        for i, (x1, y1, x2, y2) in enumerate(windows):
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            # Ensure crop is the right size (pad if necessary at edges)
            crop_h, crop_w = crop.shape[:2]
            if crop_h != self.crop_size or crop_w != self.crop_size:
                # Pad crop to crop_size
                padded_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=crop.dtype)
                padded_crop[:crop_h, :crop_w] = crop
                crop = padded_crop
            
            # Detect in this crop
            crop_detections = self.detect_in_crop(crop, (x1, y1))
            all_detections.extend(crop_detections)
        
        logger.info(f"Found {len(all_detections)} raw detections before NMS")
        
        # Apply NMS to remove duplicates
        filtered_detections = self.apply_nms(all_detections)
        logger.info(f"After NMS: {len(filtered_detections)} detections")
        
        # Process and visualize results
        return self.visualize_detections(frame, filtered_detections, target_label)

    def visualize_detections(self, frame: np.ndarray, detections: List[Dict], target_label: Optional[str] = None) -> Tuple[np.ndarray, bool, int, int, int, float]:
        """
        Visualize detections on the frame and return statistics.
        """
        annotated_frame = frame.copy()
        
        detected_target = False
        non_target_count = 0
        total_pieces_detected = len(detections)
        correct_pieces_count = 0
        max_confidence = 0.0
        
        target_color = (0, 255, 0)  # Green for target
        other_color = (0, 0, 255)   # Red for others
        
        for det in detections:
            confidence = det['confidence']
            max_confidence = max(max_confidence, confidence)
            
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            detected_label = det['label']
            
            # Determine if this is our target
            if target_label and detected_label == target_label:
                color = target_color
                detected_target = True
                correct_pieces_count += 1
            else:
                color = other_color
                non_target_count += 1
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            confidence_percent = confidence * 100
            label_text = f"{detected_label}: {confidence_percent:.1f}%"
            
            # Dynamic label placement
            font_scale = 1.0
            font_thickness = 2
            (label_width, label_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Position label above bounding box, with fallback positions
            label_x = x1
            label_y = y1 - 10
            
            # Adjust for frame boundaries
            frame_height, frame_width = annotated_frame.shape[:2]
            
            if label_y - label_height < 0:
                label_y = y1 + label_height + 15
            if label_y > frame_height - 5:
                label_y = frame_height - 5
                
            if label_x < 0:
                label_x = 5
            if label_x + label_width > frame_width:
                label_x = frame_width - label_width - 5
            
            # Draw label background
            padding = 3
            bg_x1 = max(0, label_x - padding)
            bg_y1 = max(0, label_y - label_height - padding)
            bg_x2 = min(frame_width, label_x + label_width + padding)
            bg_y2 = min(frame_height, label_y + padding)
            
            cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label_text, (label_x, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return annotated_frame, detected_target, non_target_count, total_pieces_detected, correct_pieces_count, max_confidence

    def detect_and_contour(self, frame, target_label):
        """Original detection method for backward compatibility."""
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                imgsz=512,
            )
            
            if not results or len(results) == 0:
                return frame, False, 0, 0, 0, 0.0
                
            result = results[0]
            
        except Exception as e:
            print(f"Detection failed: {e}")
            return frame, False, 0, 0, 0, 0.0

        class_names = self.model.names
        target_color = (0, 255, 0)
        other_color = (0, 0, 255)

        detected_target = False
        non_target_count = 0
        total_pieces_detected = 0
        correct_pieces_count = 0
        max_confidence = 0.0

        if result.boxes is None or len(result.boxes) == 0:
            return frame, detected_target, non_target_count, total_pieces_detected, correct_pieces_count, max_confidence

        frame_height, frame_width = frame.shape[:2]

        for box in result.boxes:
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            confidence_percent = confidence * 100
            label = f"{detected_label}: {confidence_percent:.1f}%"

            font_scale = 1.0
            font_thickness = 2
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            label_x = x1
            label_y = y1 - 10

            if label_y - label_height < 0:
                label_y = y1 + label_height + 5
            if label_y > frame_height - 5:
                label_y = frame_height - 5

            if label_x < 0:
                label_x = 5
            if label_x + label_width > frame_width:
                label_x = frame_width - label_width - 5

            padding = 3
            bg_x1 = max(0, label_x - padding)
            bg_y1 = max(0, label_y - label_height - padding)
            bg_x2 = min(frame_width, label_x + label_width + padding)
            bg_y2 = min(frame_height, label_y + padding)

            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.putText(frame, label, (label_x, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return frame, detected_target, non_target_count, total_pieces_detected, correct_pieces_count, max_confidence

    @staticmethod
    def resize_frame_optimized(frame: np.ndarray, target_size=(512, 512)) -> np.ndarray:
        """Optimized frame resizing with better interpolation."""
        if frame.shape[:2] != target_size[::-1]:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        return frame