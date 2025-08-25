import cv2
import torch
from fastapi import HTTPException
from detection.app.service.model_service import get_model_for_group, extract_group_from_piece_label
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LabelPlacementManager:
    """
    Manages label placement to prevent overlapping labels
    """
    
    def __init__(self):
        self.placed_rectangles = []
    
    def clear(self):
        """Clear all placed rectangles for a new frame"""
        self.placed_rectangles = []
    
    def rectangles_overlap(self, rect1, rect2):
        """
        Check if two rectangles overlap
        rect format: (x1, y1, x2, y2)
        """
        x1_1, y1_1, x2_1, y2_1 = rect1
        x1_2, y1_2, x2_2, y2_2 = rect2
        
        # Check if rectangles don't overlap
        if x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1:
            return False
        return True
    
    def find_free_label_position(self, bounding_box, label_width, label_height, 
                                frame_width, frame_height, padding=3):
        """
        Find a free position for label that doesn't overlap with existing labels
        
        Args:
            bounding_box: (x1, y1, x2, y2) of the detection box
            label_width: Width of the label text
            label_height: Height of the label text
            frame_width: Frame width
            frame_height: Frame height
            padding: Padding around label
            
        Returns:
            (label_x, label_y, bg_x1, bg_y1, bg_x2, bg_y2) - label position and background rectangle
        """
        x1, y1, x2, y2 = bounding_box
        
        # Initial position (above the bounding box)
        label_x = x1
        label_y = y1 - 10
        
        # Adjust for frame boundaries horizontally
        if label_x < 0:
            label_x = 5
        if label_x + label_width > frame_width:
            label_x = frame_width - label_width - 5
        
        # Try positions above the box first, then below if needed
        positions_to_try = []
        
        # Above the box - multiple positions
        for offset in range(0, 200, 25):  # Try positions moving upward
            test_y = y1 - 10 - offset
            if test_y - label_height >= 0:  # Within frame bounds
                positions_to_try.append((label_x, test_y))
        
        # Below the box - multiple positions
        for offset in range(0, 200, 25):  # Try positions moving downward
            test_y = y2 + label_height + 15 + offset
            if test_y <= frame_height - 5:  # Within frame bounds
                positions_to_try.append((label_x, test_y))
        
        # Try each position until we find one that doesn't overlap
        for test_label_x, test_label_y in positions_to_try:
            # Calculate background rectangle for this position
            bg_x1 = max(0, test_label_x - padding)
            bg_y1 = max(0, test_label_y - label_height - padding)
            bg_x2 = min(frame_width, test_label_x + label_width + padding)
            bg_y2 = min(frame_height, test_label_y + padding)
            
            test_rect = (bg_x1, bg_y1, bg_x2, bg_y2)
            
            # Check if this rectangle overlaps with any existing ones
            overlaps = False
            for existing_rect in self.placed_rectangles:
                if self.rectangles_overlap(test_rect, existing_rect):
                    overlaps = True
                    break
            
            if not overlaps:
                # Found a free position
                self.placed_rectangles.append(test_rect)
                return test_label_x, test_label_y, bg_x1, bg_y1, bg_x2, bg_y2
        
        # If no free position found, use the first position (fallback)
        # This shouldn't happen often with enough position attempts
        label_x = x1
        label_y = max(label_height + padding, y1 - 10)
        
        # Ensure within frame bounds
        if label_x < 0:
            label_x = 5
        if label_x + label_width > frame_width:
            label_x = frame_width - label_width - 5
        if label_y > frame_height - 5:
            label_y = frame_height - 5
        
        bg_x1 = max(0, label_x - padding)
        bg_y1 = max(0, label_y - label_height - padding)
        bg_x2 = min(frame_width, label_x + label_width + padding)
        bg_y2 = min(frame_height, label_y + padding)
        
        fallback_rect = (bg_x1, bg_y1, bg_x2, bg_y2)
        self.placed_rectangles.append(fallback_rect)
        
        return label_x, label_y, bg_x1, bg_y1, bg_x2, bg_y2

class IdentificationDetectionSystem:
    """
    Detection system specifically designed for piece identification with group-based models
    Uses sliding window detection and focuses on identification rather than validation
    """
    
    def __init__(self, confidence_threshold=0.5, group_name=None, 
                 crop_size=800, overlap_ratio=0.5, nms_threshold=0.4):
        """
        Enhanced identification system with sliding window support.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            group_name: Group identifier to load the appropriate model (e.g., 'E539', 'G053')
            crop_size: Size of sliding window crops (should match training crop size)
            overlap_ratio: Overlap between sliding windows (0.5 = 50% overlap)
            nms_threshold: Non-Maximum Suppression threshold for duplicate removal
        """
        self.confidence_threshold = confidence_threshold
        self.group_name = group_name
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.nms_threshold = nms_threshold
        self.device = self.get_device()
        self.model = self.get_my_model()
        self.label_manager = LabelPlacementManager()  # Add label placement manager
        
        # Identification-specific settings
        self.identification_color = (224, 99, 1)   # Yellow color for all pieces
        self.font_scale = 1.0
        self.font_thickness = 2
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
        """Load the YOLO model for the specified group."""
        if not self.group_name:
            raise HTTPException(status_code=400, detail="Group name is required for identification.")
        
        model = get_model_for_group(self.group_name)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model not found for group {self.group_name}.")

        # Move model to the appropriate device
        model.to(self.device)

        # Convert to half precision if using a GPU
        if self.device.type == 'cuda':
            model.half()

        print(f"Loaded identification model for group: {self.group_name}")
        return model

    def switch_model_for_group(self, new_group_name):
        """Switch to a different group model."""
        if new_group_name != self.group_name:
            print(f"Switching from group {self.group_name} to group {new_group_name}")
            
            new_model = get_model_for_group(new_group_name)
            if new_model is None:
                print(f"Could not load model for group {new_group_name}, keeping current model")
                return False
            
            self.model = new_model
            self.model.to(self.device)
            if self.device.type == 'cuda':
                self.model.half()
            
            self.group_name = new_group_name
            print(f"Successfully switched to identification model for group {new_group_name}")
            return True
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

    def identify_in_crop(self, crop: np.ndarray, crop_offset: Tuple[int, int]) -> List[Dict]:
        """
        Identify pieces in a single crop and convert coordinates back to original frame.
        
        Args:
            crop: Image crop to process
            crop_offset: (offset_x, offset_y) of the crop in the original frame
            
        Returns:
            List of identifications with global coordinates
        """
        identifications = []
        offset_x, offset_y = crop_offset
        
        try:
            # Run identification on the crop
            results = self.model.predict(
                crop,
                conf=self.confidence_threshold,
                device=self.device,
                imgsz=512,  # This should match your training image size
                verbose=False
            )
            
            if not results or len(results) == 0:
                return identifications
                
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return identifications

            class_names = self.model.names
            
            for i, box in enumerate(result.boxes):
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
                piece_label = class_names[class_id]
                
                identifications.append({
                    'piece_id': f"piece_{len(identifications)}",
                    'label': piece_label,
                    'confidence': confidence,
                    'x1': x1_global,
                    'y1': y1_global,
                    'x2': x2_global,
                    'y2': y2_global,
                    'class_id': class_id,
                    'center': {'x': (x1_global + x2_global) // 2, 'y': (y1_global + y2_global) // 2},
                    'crop_offset': crop_offset
                })
                
        except Exception as e:
            logger.error(f"Identification failed for crop at {crop_offset}: {e}")
            
        return identifications

    def apply_nms(self, identifications: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate identifications.
        
        Args:
            identifications: List of identification dictionaries
            
        Returns:
            Filtered list of identifications after NMS
        """
        if len(identifications) == 0:
            return identifications
        
        # Convert identifications to format suitable for NMS
        boxes = []
        scores = []
        class_ids = []
        
        for ident in identifications:
            boxes.append([ident['x1'], ident['y1'], ident['x2'], ident['y2']])
            scores.append(ident['confidence'])
            class_ids.append(ident['class_id'])
        
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
        
        # Return filtered identifications and reassign piece_ids
        filtered_identifications = [identifications[i] for i in keep_indices]
        for i, ident in enumerate(filtered_identifications):
            ident['piece_id'] = f"piece_{i}"
        
        return filtered_identifications

    def identify_with_sliding_window(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Identify pieces using sliding window approach.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, detection_details)
        """
        frame_height, frame_width = frame.shape[:2]
        
        # If frame is smaller than crop size, use original identification method
        if frame_height < self.crop_size or frame_width < self.crop_size:
            logger.info(f"Frame too small ({frame_width}x{frame_height}), using direct identification")
            return self.identify_and_annotate_direct(frame)
        
        logger.info(f"Using sliding window identification on {frame_width}x{frame_height} frame")
        
        # Generate sliding windows
        windows = self.generate_sliding_windows(frame_height, frame_width)
        logger.info(f"Generated {len(windows)} sliding windows")
        
        # Collect all identifications
        all_identifications = []
        
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
            
            # Identify in this crop
            crop_identifications = self.identify_in_crop(crop, (x1, y1))
            all_identifications.extend(crop_identifications)
        
        logger.info(f"Found {len(all_identifications)} raw identifications before NMS")
        
        # Apply NMS to remove duplicates
        filtered_identifications = self.apply_nms(all_identifications)
        logger.info(f"After NMS: {len(filtered_identifications)} identifications")
        
        # Annotate and return results
        return self.annotate_identifications(frame, filtered_identifications)

    def annotate_identifications(self, frame: np.ndarray, identifications: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Annotate identifications on the frame.
        """
        annotated_frame = frame.copy()
        detection_details = []
        
        frame_height, frame_width = frame.shape[:2]
        
        # Clear the label manager for this frame
        self.label_manager.clear()
        
        for ident in identifications:
            x1, y1, x2, y2 = ident['x1'], ident['y1'], ident['x2'], ident['y2']
            confidence = ident['confidence']
            piece_label = ident['label']
            piece_id = ident['piece_id']
            
            # Store detection details in the expected format
            detection_info = {
                'piece_id': piece_id,
                'label': piece_label,
                'confidence': confidence,
                'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'center': ident['center']
            }
            detection_details.append(detection_info)
            
            # Draw bounding box with identification color (yellow)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.identification_color, self.box_thickness)
            
            # Prepare label text
            confidence_percent = confidence * 100
            label = f"{piece_label}: {confidence_percent:.1f}%"

            font_scale = 1.0
            font_thickness = 2
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Use label placement manager to find free position
            label_x, label_y, bg_x1, bg_y1, bg_x2, bg_y2 = self.label_manager.find_free_label_position(
                (x1, y1, x2, y2), label_width, label_height, frame_width, frame_height
            )

            # Draw background rectangle
            cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

            # Draw text
            cv2.putText(annotated_frame, label, (label_x, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return annotated_frame, detection_details

    def identify_and_annotate_direct(self, frame):
        """
        Direct identification method for backward compatibility and small frames.
        """
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                imgsz=512,
                verbose=False
            )
            
            if not results or len(results) == 0:
                return frame, []
                
            result = results[0]
            
        except Exception as e:
            print(f"Direct identification failed: {e}")
            return frame, []

        class_names = self.model.names
        detection_details = []

        # Check if there are any detection boxes
        if result.boxes is None or len(result.boxes) == 0:
            return frame, detection_details

        # Get frame dimensions for boundary checks
        frame_height, frame_width = frame.shape[:2]
        
        # Clear the label manager for this frame
        self.label_manager.clear()
        
        for i, box in enumerate(result.boxes):
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
            
            # Prepare label text
            confidence_percent = confidence * 100
            label = f"{piece_label}: {confidence_percent:.1f}%"

            font_scale = 1.0
            font_thickness = 2
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Use label placement manager to find free position
            label_x, label_y, bg_x1, bg_y1, bg_x2, bg_y2 = self.label_manager.find_free_label_position(
                (x1, y1, x2, y2), label_width, label_height, frame_width, frame_height
            )

            # Draw background rectangle
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

            # Draw text
            cv2.putText(frame, label, (label_x, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        return frame, detection_details

    def identify_and_annotate(self, frame):
        """
        Main identification method that chooses between sliding window and direct methods.
        """
        return self.identify_with_sliding_window(frame)

    def get_detailed_identification_results(self, frame):
        """
        Get detailed identification results without modifying the frame
        Returns comprehensive information about detected pieces
        """
        # Use sliding window approach for detailed results
        frame_height, frame_width = frame.shape[:2]
        
        if frame_height < self.crop_size or frame_width < self.crop_size:
            return self._get_detailed_results_direct(frame)
        
        # Generate sliding windows and collect all identifications
        windows = self.generate_sliding_windows(frame_height, frame_width)
        all_identifications = []
        
        for x1, y1, x2, y2 in windows:
            crop = frame[y1:y2, x1:x2]
            
            # Ensure crop is the right size
            crop_h, crop_w = crop.shape[:2]
            if crop_h != self.crop_size or crop_w != self.crop_size:
                padded_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=crop.dtype)
                padded_crop[:crop_h, :crop_w] = crop
                crop = padded_crop
            
            crop_identifications = self.identify_in_crop(crop, (x1, y1))
            all_identifications.extend(crop_identifications)
        
        # Apply NMS and format results
        filtered_identifications = self.apply_nms(all_identifications)
        detailed_results = []
        
        for ident in filtered_identifications:
            width = ident['x2'] - ident['x1']
            height = ident['y2'] - ident['y1']
            area = width * height
            
            piece_info = {
                'piece_id': ident['piece_id'],
                'label': ident['label'],
                'confidence': round(ident['confidence'], 3),
                'confidence_percentage': round(ident['confidence'] * 100, 1),
                'bounding_box': {
                    'x1': ident['x1'], 'y1': ident['y1'], 
                    'x2': ident['x2'], 'y2': ident['y2'],
                    'width': width, 'height': height
                },
                'area': area,
                'center': ident['center'],
                'confidence_level': self._get_confidence_level(ident['confidence']),
                'class_id': ident['class_id']
            }
            detailed_results.append(piece_info)
        
        # Sort by confidence (highest first)
        detailed_results.sort(key=lambda x: x['confidence'], reverse=True)
        return detailed_results

    def _get_detailed_results_direct(self, frame):
        """Direct method for detailed results on small frames."""
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                imgsz=512,
                verbose=False
            )
            
            if not results or len(results) == 0:
                return []
                
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []
                
        except Exception as e:
            print(f"Detailed identification failed: {e}")
            return []

        class_names = self.model.names
        detailed_results = []

        for i, box in enumerate(result.boxes):
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
            'group_name': self.group_name,
            'model_classes': list(self.model.names.values()) if hasattr(self.model, 'names') else [],
            'identification_color': self.identification_color,
            'total_classes': len(self.model.names) if hasattr(self.model, 'names') else 0,
            'crop_size': self.crop_size,
            'overlap_ratio': self.overlap_ratio,
            'nms_threshold': self.nms_threshold
        }

    @staticmethod
    def resize_frame_optimized(frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
        """Optimized frame resizing with better interpolation."""
        if frame.shape[:2] != target_size[::-1]:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        return frame

    def batch_identify(self, frames: list):
        """
        Identify pieces in multiple frames at once using sliding window approach
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