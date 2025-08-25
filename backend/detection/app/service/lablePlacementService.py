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