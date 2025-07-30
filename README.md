# ConvNext Kidney Analysis

This project uses a ConvNext model to classify kidney tissue patches as normal, sclerotic, or background.

## Project Structure

- `extract_patches.py`: Extracts patches from whole-slide images (`.svs` files) and their corresponding annotations (`.geojson` files). It creates a balanced dataset of normal, sclerotic, and background patches.
- `dataset.py`: Defines the PyTorch `Dataset` and `DataLoader` for loading the extracted patches. It includes data augmentation for the training set.
- `model.py`: Defines the ConvNext-based classifier.
- `train.py`: Trains the model on the extracted patches.
- `test_patch_extraction.py`: A script to visualize the patch extraction process for debugging and verification.
- `analyze_class_balance.py`: Analyzes the class balance in the annotated dataset.
- `requirements.txt`: Contains the necessary Python packages to run the project.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data:**
    - Place your whole-slide images (`.svs`) in the `data/raw images` directory.
    - Place your GeoJSON annotations in the `data/geoJson annotations` directory.

## Usage

1.  **Extract Patches:**
    ```bash
    python extract_patches.py
    ```
    This will create a `patches` directory with subdirectories for each class and a `train_patches.txt` file listing the paths to the extracted patches and their corresponding labels.

2.  **Train the Model:**
    ```bash
    python train.py
    ```
    This will train the ConvNext model and save the best-performing model to the `checkpoints` directory.

3.  **Test Patch Extraction (Optional):**
    ```bash
    python test_patch_extraction.py
    ```
    This will generate visualizations of the patch extraction process, which can be found in the `visualization_outputs` directory.

4.  **Analyze Class Balance (Optional):**
    ```bash
    python analyze_class_balance.py
    ```
    This will print a report on the class balance in your dataset.
