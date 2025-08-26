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
    print(f"Models base path: {MODELS_BASE_PATH}")
    print(f"Target piece label: {target_piece_label}")
    
    model_path = None
    
    # If target piece label is provided, try to load the specific group model
    if target_piece_label:
        group_name = extract_group_from_piece_label(target_piece_label)
        print(f"Extracted group name: {group_name}")
        
        if group_name:
            group_model_path = os.path.join(MODELS_BASE_PATH, f'model_{group_name}.pt')
            print(f"Looking for group model: {group_model_path}")
            
            if os.path.isfile(group_model_path):
                model_path = group_model_path
                print(f"Found group model for {group_name}: {model_path}")
            else:
                print(f"Group model not found for {group_name}, will try fallback options")
        else:
            print(f"Could not extract group from piece label: {target_piece_label}")
    
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