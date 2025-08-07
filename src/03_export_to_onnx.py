import torch
import torchvision.models as models
import torch.nn as nn
import os

# --- CONFIGURATION ---
PYTORCH_MODEL_PATH = './models/scrap_classifier.pth'
ONNX_MODEL_PATH = './models/scrap_classifier.onnx'
# Based on the TrashNet dataset
NUM_CLASSES = 6 
# --- END CONFIGURATION ---

def export_to_onnx():
    """
    Loads a trained PyTorch model and exports it to ONNX format.
    """
    print(f"Loading PyTorch model from {PYTORCH_MODEL_PATH}...")
    
    # 1. Re-create the model architecture
    # We must define the model structure again before loading the state dict
    model = models.resnet18(weights=None) # Don't need pretrained weights here
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 2. Load the trained weights
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")

    # 3. Create a dummy input tensor
    # The shape must match the model's expected input: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

    # 4. Export the model
    print(f"Exporting model to ONNX format at {ONNX_MODEL_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("\nModel has been converted to ONNX successfully.")
    print(f"Saved at: {ONNX_MODEL_PATH}")

if __name__ == '__main__':
    if not os.path.exists(PYTORCH_MODEL_PATH):
        print(f"Error: PyTorch model not found at '{PYTORCH_MODEL_PATH}'")
        print("Please run '02_train_model.py' first.")
    else:
        export_to_onnx()