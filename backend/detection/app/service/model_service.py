import os
from ultralytics import YOLO

# Use the environment variable that matches your Docker configuration
MODELS_BASE_PATH = os.getenv('MODELS_BASE_PATH', '/app/shared/models')

def load_my_model():
    """Load YOLO model from the shared models directory."""
    
    # Use the shared models path from environment variable
    model_path = os.path.join(MODELS_BASE_PATH, 'model.pt')
    
    print(f"Models base path: {MODELS_BASE_PATH}")
    print(f"Model path: {model_path}")
    
    # Check if the model file exists
    if not os.path.isfile(model_path):
        print(f"Model file not found at: {model_path}")
        # List available files in the models directory for debugging
        if os.path.exists(MODELS_BASE_PATH):
            print(f"Available files in {MODELS_BASE_PATH}:")
            for file in os.listdir(MODELS_BASE_PATH):
                print(f"  - {file}")
        else:
            print(f"Models directory does not exist: {MODELS_BASE_PATH}")
        return None
    
    try:
        # Load the model if the file exists
        my_model = YOLO(model_path)
        print("Model loaded successfully.")
        return my_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None