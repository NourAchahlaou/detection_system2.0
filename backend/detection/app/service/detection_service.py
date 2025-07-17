import cv2
import torch
from fastapi import HTTPException
from detection.service.model_service import load_my_model

class DetectionSystem:
    def __init__(self, confidence_threshold=0.5):
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

        # Normalize the tensor to the range [0.0, 1.0]
        frame_tensor /= 255.0  # Normalize to [0, 1]

        # Use half precision if using a GPU
        frame_tensor = frame_tensor.half() if self.device.type == 'cuda' else frame_tensor

        # Perform detection on the received frame
        try:
            results = self.model(frame_tensor.unsqueeze(0))[0]  # Add batch dimension
        except Exception as e:
            print(f"Detection failed: {e}")
            return frame, False, 0  # Return the frame and zero non-target count

        class_names = self.model.names  # Retrieve the list of class names

        # Define colors: green for the target label, red for others
        target_color = (0, 255, 0)  # Green
        other_color = (0, 0, 255)  # Red

        detected_target = False
        non_target_count = 0  # Counter for pieces that shouldn't be passing

        for box in results.boxes:
            confidence = box.conf.item()
            if confidence < self.confidence_threshold:
                continue  # Skip this detection

            # Extract the bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            # Extract the class ID and corresponding label
            class_id = int(box.cls.item())
            detected_label = class_names[class_id]

            # Determine the color based on whether it's the target label
            if detected_label == target_label:
                color = target_color
                detected_target = True
            else:
                color = other_color
                non_target_count += 1  # Increment the count for non-target pieces

            # Draw the bounding box with the determined color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label = f"{detected_label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Calculate position for the label
            label_y = y1 - 10  # Default position above the bounding box
            if label_y < 0:  # If the label goes off the top of the frame
                label_y = y2 + label_size[1] + 5  # Position below the bounding box
            
            # Ensure label does not exceed frame width
            if x1 + label_size[0] > frame.shape[1]:  # If label exceeds right edge
                label_x = x1 - label_size[0]
            else:
                label_x = x1  # Default to the left of the bounding box

            # Draw a filled rectangle behind the text for better visibility
            cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5), 
                          (label_x + label_size[0], label_y), (0, 0, 0), -1)

            # Add the class name and confidence score
            cv2.putText(frame, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame, detected_target, non_target_count  # Return the frame, target detection status, and non-target count
