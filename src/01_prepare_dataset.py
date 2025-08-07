import os
import shutil
import random

def prepare_dataset(source_dir, dest_dir, split_ratio=0.8):
    """
    Splits the dataset into training and validation sets.

    Args:
        source_dir (str): Path to the original dataset directory.
        dest_dir (str): Path to the destination directory where 'train' and 'val' folders will be created.
        split_ratio (float): The ratio of training data to the total data.
    """
    print(f"Preparing dataset from '{source_dir}'...")

    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
        print(f"Removed existing directory: {dest_dir}")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        # Create class subdirectories in train and val
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Get all image files for the class
        class_source_dir = os.path.join(source_dir, class_name)
        all_files = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]
        random.shuffle(all_files)

        # Split files
        split_point = int(len(all_files) * split_ratio)
        train_files = all_files[:split_point]
        val_files = all_files[split_point:]

        # Copy files to new directories
        for f in train_files:
            shutil.copy(os.path.join(class_source_dir, f), os.path.join(train_dir, class_name, f))

        for f in val_files:
            shutil.copy(os.path.join(class_source_dir, f), os.path.join(val_dir, class_name, f))
            
        print(f"  - Class '{class_name}': {len(train_files)} train, {len(val_files)} val")

    print("\nDataset preparation complete!")
    print(f"Train and validation sets are ready in '{dest_dir}'")

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # NOTE: Download the 'TrashNet' dataset and unzip it. 
    # The unzipped folder should be named 'dataset-resized'.
    # Update this path to point to where you've unzipped the dataset.
    SOURCE_DATASET_DIR = './data/dataset-resized' 
    
    # This is where the 'train' and 'val' folders will be created.
    DESTINATION_DIR = './data/prepared_dataset' 
    # --- END CONFIGURATION ---

    if not os.path.isdir(SOURCE_DATASET_DIR):
        print(f"Error: Source dataset directory not found at '{SOURCE_DATASET_DIR}'")
        print("Please download the TrashNet dataset and place it in the 'data' folder.")
    else:
        prepare_dataset(SOURCE_DATASET_DIR, DESTINATION_DIR)