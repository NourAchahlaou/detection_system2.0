import cv2
import os

def add_bounding_box_to_image(image_path, label_path, output_path):
    """
    Adds bounding boxes from a label file to the corresponding image and saves the result.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file.
        output_path (str): Path to save the output image with bounding boxes.
    """
    print(f"Processing image: {image_path} with label: {label_path}")
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    
    height, width, _ = image.shape

    # Read the label file
    try:
        with open(label_path, 'r') as label_file:
            lines = label_file.readlines()
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_path}")
        return

    if not lines:
        print(f"Warning: Label file is empty for {label_path}")
        return

    # Draw each bounding box
    for line in lines:
        try:
            # YOLO format: class_id x_center y_center width height (all normalized)
            class_id, x_center, y_center, w, h = map(float, line.split())

            # Convert normalized coordinates to pixel values
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            # Calculate the top-left and bottom-right coordinates of the bounding box
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Draw the bounding box on the image (green color, thickness 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except ValueError:
            print(f"Error: Incorrect label format in file {label_path}: {line.strip()}")
            continue

    # Save the image with bounding boxes
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to {output_path}")

def process_images_and_labels(image_dir, label_dir, output_dir):
    """
    Processes images and their corresponding labels to add bounding boxes.
    Recursively traverses directories to process all images and labels.

    Args:
        image_dir (str): Directory containing the images.
        label_dir (str): Directory containing the label files.
        output_dir (str): Directory to save the images with bounding boxes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Loop through all files in the image directory, including subdirectories
    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith(".jpg"):  # Assuming images are in .jpg format
                image_path = os.path.join(root, filename)
                relative_path = os.path.relpath(image_path, image_dir)
                label_path = os.path.join(label_dir, relative_path.replace('.jpg', '.txt'))
                output_path = os.path.join(output_dir, relative_path)

                print(f"Looking for image: {image_path}")
                print(f"Looking for label: {label_path}")

                if os.path.exists(label_path):
                    # Add bounding boxes and save the new image
                    add_bounding_box_to_image(image_path, label_path, output_path)
                else:
                    print(f"Warning: No corresponding label file for {image_path}")

# Example usage
def main():
    # Directories for training and validation images and labels
    train_image_dir = r"C:\Users\hp\Desktop\airbus2.0\detection_System2.0\shared_data\dataset\dataset_custom\G053_cropped\images\train\G053.42874.105.03"
    train_label_dir = r"C:\Users\hp\Desktop\airbus2.0\detection_System2.0\shared_data\dataset\dataset_custom\G053_cropped\labels\train\G053.42874.105.03"
    output_train_dir = r"C:\Users\hp\Desktop\airbus2.0\detection_System2.0\shared_data\dataset\dataset_custom\G053_cropped\output\train\G053.42874.105.03"

    val_image_dir = r"C:\Users\hp\Desktop\airbus2.0\detection_System2.0\shared_data\dataset\dataset_custom\G053_cropped\images\valid\G053.42874.105.03"
    val_label_dir = r"C:\Users\hp\Desktop\airbus2.0\detection_System2.0\shared_data\dataset\dataset_custom\G053_cropped\labels\valid\G053.42874.105.03"
    output_val_dir = r"C:\Users\hp\Desktop\airbus2.0\detection_System2.0\shared_data\dataset\dataset_custom\G053_cropped\output\valid\G053.42874.105.03"

    # Process the images and labels
    print("Processing training images...")
    process_images_and_labels(train_image_dir, train_label_dir, output_train_dir)
    print("Processing validation images...")
    process_images_and_labels(val_image_dir, val_label_dir, output_val_dir)

if __name__ == "__main__":
    main()
