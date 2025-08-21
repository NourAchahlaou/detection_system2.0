import os
from ultralytics import YOLO
from detection.app.db.models import piece
import re

# Use the environment variable that matches your Docker configuration
MODELS_BASE_PATH = os.getenv('MODELS_BASE_PATH', '/app/shared/models')

def extract_group_from_piece_label(piece_label: str) -> str:
    """Extract group identifier from piece label (e.g., 'E539.12345' -> 'E539')"""
    match = re.match(r'([A-Z]\d{3})', piece_label)
    if match:
        return match.group(1)
    return None

def load_my_model(target_piece_label=None):
    """
    Load YOLO model from the shared models directory.
    If target_piece_label is provided, load the specific group model.
    Otherwise, fallback to generic model.pt or first available model.
    """
    
    print(f"Models base path: {MODELS_BASE_PATH}")
    
    model_path = None
    
    # If target piece label is provided, try to load the specific group model
    if target_piece_label:
        group_name = extract_group_from_piece_label(target_piece_label)
        if group_name:
            group_model_path = os.path.join(MODELS_BASE_PATH, f'model_{group_name}.pt')
            print(f"Looking for group model: {group_model_path}")
            
            if os.path.isfile(group_model_path):
                model_path = group_model_path
                print(f"Found group model for {group_name}: {model_path}")
            else:
                print(f"Group model not found for {group_name}, will try fallback options")
    
    # If no specific model found, try fallback options
    if not model_path:
        # Try generic model.pt first
        generic_model_path = os.path.join(MODELS_BASE_PATH, 'model.pt')
        if os.path.isfile(generic_model_path):
            model_path = generic_model_path
            print(f"Using generic model: {model_path}")
        else:
            # If no generic model, try to find any available model
            if os.path.exists(MODELS_BASE_PATH):
                available_models = [f for f in os.listdir(MODELS_BASE_PATH) if f.endswith('.pt')]
                if available_models:
                    # Sort models to prefer group models over others
                    available_models.sort(key=lambda x: (not x.startswith('model_'), x))
                    model_path = os.path.join(MODELS_BASE_PATH, available_models[0])
                    print(f"Using first available model: {model_path}")
    
    print(f"Final model path: {model_path}")
    
    # Check if we found a valid model path
    if not model_path or not os.path.isfile(model_path):
        print(f"No valid model found")
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

def get_available_models():
    """
    Get a list of all available models in the models directory.
    Returns a dictionary with model names and their full paths.
    """
    available_models = {}
    
    if not os.path.exists(MODELS_BASE_PATH):
        print(f"Models directory does not exist: {MODELS_BASE_PATH}")
        return available_models
    
    for file in os.listdir(MODELS_BASE_PATH):
        if file.endswith('.pt'):
            full_path = os.path.join(MODELS_BASE_PATH, file)
            available_models[file] = full_path
    
    return available_models

def get_model_for_group(group_name):
    """
    Load a specific model for a given group name.
    
    Args:
        group_name (str): The group identifier (e.g., 'G053', 'E539')
    
    Returns:
        YOLO model or None if not found
    """
    group_model_path = os.path.join(MODELS_BASE_PATH, f'model_{group_name}.pt')
    
    if not os.path.isfile(group_model_path):
        print(f"Model for group {group_name} not found at: {group_model_path}")
        return None
    
    try:
        model = YOLO(group_model_path)
        print(f"Successfully loaded model for group {group_name}")
        return model
    except Exception as e:
        print(f"Error loading model for group {group_name}: {e}")
        return None