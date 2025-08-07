import onnxruntime
import numpy as np
import cv2
import os
import time
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
ONNX_MODEL_PATH = './models/scrap_classifier.onnx'
# Simulate using images from the validation set
SIMULATION_IMG_DIR = './data/prepared_dataset/val'
RESULTS_CSV_PATH = './results/classification_log.csv'
CONFIDENCE_THRESHOLD = 0.70  # 70% confidence
SIMULATION_DELAY_S = 1.5 # Delay in seconds between frames
# Must be in the same order as during training
CLASS_NAMES = ['cardboard', 'e-waste', 'fabric', 'glass', 'metal', 'paper']
# --- END CONFIGURATION ---

# Image preprocessing must match the training script
def preprocess_image(image):
    """
    Preprocesses a single image for ONNX inference.
    Args:
        image (numpy.ndarray): Image in BGR format from OpenCV.
    Returns:
        numpy.ndarray: Preprocessed image ready for the model.
    """
    # 1. Resize and crop
    h, w, _ = image.shape
    scale = 256 / min(h, w)
    image = cv2.resize(image, (int(w * scale), int(h * scale)))
    
    h, w, _ = image.shape
    startx = w // 2 - (224 // 2)
    starty = h // 2 - (224 // 2)
    image = image[starty:starty + 224, startx:startx + 224]

    # 2. Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. HWC to CHW and normalize
    image = image.transpose((2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image / 255.0
    image = (image - mean[:, None, None]) / std[:, None, None]
    
    # 4. Add batch dimension
    return image.astype(np.float32)[np.newaxis, :]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

def run_simulation():
    """
    Main function to run the conveyor belt simulation.
    """
    print("--- Starting Scrap Classification Simulation ---")
    
    # 1. Load ONNX Model
    print(f"Loading model: {ONNX_MODEL_PATH}")
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    # 2. Prepare for simulation loop
    image_files = []
    for root, _, files in os.walk(SIMULATION_IMG_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"Error: No images found in '{SIMULATION_IMG_DIR}'.")
        return

    print(f"Found {len(image_files)} images to process for simulation.\n")
    
    results_log = []

    # 3. Real-Time Loop
    for i, image_path in enumerate(image_files):
        print(f"--- Frame {i+1}/{len(image_files)} ---")
        
        # Mimic frame capture
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
            
        # Preprocess frame for classification
        processed_frame = preprocess_image(frame)
        
        # Run Inference
        model_output = session.run(None, {input_name: processed_frame})[0]
        
        # Post-process output
        probabilities = softmax(model_output)[0]
        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        # Log to console
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Image: {os.path.basename(image_path)}")
        print(f"  -> Prediction: '{predicted_class_name}' | Confidence: {confidence:.2%}")
        
        # Check confidence threshold
        low_confidence_flag = False
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"  -> **WARNING**: Confidence ({confidence:.2%}) is below threshold ({CONFIDENCE_THRESHOLD:.2%})!")
            low_confidence_flag = True

        # Store result for CSV
        results_log.append({
            'timestamp': timestamp,
            'image_file': os.path.basename(image_path),
            'true_label': os.path.basename(os.path.dirname(image_path)), # From folder name
            'predicted_class': predicted_class_name,
            'confidence': f"{confidence:.4f}",
            'low_confidence_flag': low_confidence_flag
        })
        
        # Simulate conveyor belt delay
        time.sleep(SIMULATION_DELAY_S)
        print("-" * (len(str(i+1)) + len(str(len(image_files))) + 13))


    # 4. Save results to CSV
    df = pd.DataFrame(results_log)
    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSimulation complete. Results logged to '{RESULTS_CSV_PATH}'.")

if __name__ == '__main__':
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: ONNX model not found at '{ONNX_MODEL_PATH}'")
        print("Please run '03_export_to_onnx.py' first.")
    else:
        run_simulation()