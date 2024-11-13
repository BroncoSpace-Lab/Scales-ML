import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset
from PIL import Image
import numpy as np
import time
from collections import deque
import pandas as pd
import nannyml as nml
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

class FPSCounter:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def get_fps(self):
        if not self.frame_times:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1 / avg_time if avg_time > 0 else 0

def load_model_and_processor(model_name="microsoft/resnet-152"):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = ResNetForImageClassification.from_pretrained(model_name)
    return model, image_processor

def process_image(image, image_processor):
    if isinstance(image, np.ndarray):
         image = Image.fromarray(image)
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs

def classify_image(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    return predicted_class_idx, probabilities[0]

def create_monitoring_datasets(predictions_df, min_samples_per_class=10):
    """
    Split predictions into reference and analysis sets ensuring minimum samples per class
    with clean timestamp handling.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions
        min_samples_per_class (int): Minimum number of samples required per class
        
    Returns:
        tuple: (reference_df, analysis_df)
    """
    # Create clean timestamps without nanoseconds
    base_date = datetime(2024, 1, 1)  # Start from a clean date
    timestamps = [base_date + timedelta(minutes=i) for i in range(len(predictions_df))]
    predictions_df['timestamp'] = pd.to_datetime(timestamps).floor("s")
    
    # Initialize containers for split datasets
    reference_dfs = []
    analysis_dfs = []
    
    # Split while maintaining class balance
    for class_label in predictions_df['y_true'].unique():
        class_data = predictions_df[predictions_df['y_true'] == class_label].copy()
        
        if len(class_data) < min_samples_per_class * 2:
            print(f"Warning: Class {class_label} has fewer than {min_samples_per_class * 2} samples")
            continue
        
        # Split at midpoint while maintaining temporal order
        split_idx = len(class_data) // 2
        reference_dfs.append(class_data.iloc[:split_idx])
        analysis_dfs.append(class_data.iloc[split_idx:])
    
    if not reference_dfs or not analysis_dfs:
        raise ValueError("Not enough samples per class for meaningful split")
    
    # Combine and sort the splits
    reference_df = pd.concat(reference_dfs, ignore_index=True)
    analysis_df = pd.concat(analysis_dfs, ignore_index=True)
    
    # Sort by timestamp to maintain temporal order
    reference_df = reference_df.sort_values('timestamp').reset_index(drop=True)
    analysis_df = analysis_df.sort_values('timestamp').reset_index(drop=True)
    
    return reference_df, analysis_df

def main(model_name="microsoft/resnet-152", num_samples=5000, min_samples_per_class=10):
    # Load the "train" split for the reference dataset and the "test" split for the analysis dataset
    train_dataset = load_dataset("uoft-cs/cifar100", split="train")
    test_dataset = load_dataset("uoft-cs/cifar100", split="test")
    
    model, image_processor = load_model_and_processor(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Helper function to process and collect predictions
    def process_dataset(dataset, dataset_name="train"):
        class_counts = {}
        predictions = []
        
        for i, example in enumerate(dataset):
            # Set sample limit for each split
            if dataset_name == "train" and i >= num_samples:
                break
            elif dataset_name == "test" and i >= num_samples:
                break
                
            true_label = example["fine_label"]
            
            # For the "train" (reference) set, enforce a minimum samples rule per class
            if dataset_name == "train" and true_label in class_counts and class_counts[true_label] >= min_samples_per_class * 2:
                continue
                
            inputs = process_image(example["img"], image_processor)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            predicted_class_idx, probabilities = classify_image(model, inputs)
            
            # Store prediction data
            pred_data = {
                'y_true': true_label,
                'y_pred': predicted_class_idx,
            }
            
            # Add probability scores for each class
            for class_idx, prob in enumerate(probabilities.cpu().numpy()):
                pred_data[f'pred_proba_{class_idx}'] = prob
                
            predictions.append(pred_data)
            
            # Update class counts
            class_counts[true_label] = class_counts.get(true_label, 0) + 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} images from {dataset_name} set")
        
        return pd.DataFrame(predictions)
    
    # Process both the "train" and "test" splits
    reference_df = process_dataset(train_dataset, "train")
    analysis_df = process_dataset(test_dataset, "test")
    
    # Add timestamps for monitoring, with each set starting from a unique timestamp
    base_date_train = datetime(2024, 1, 1)
    base_date_test = base_date_train + timedelta(minutes=len(reference_df))
    
    # Add timestamps for monitoring, ensuring they have second precision from the start
    reference_df['timestamp'] = pd.Series([base_date_train + timedelta(minutes=i) for i in range(len(reference_df))])
    analysis_df['timestamp'] = pd.Series([base_date_test + timedelta(minutes=i) for i in range(len(analysis_df))])

    
    print("\nClass distribution in reference set:")
    print(reference_df['y_true'].value_counts().describe())
    
    print("\nClass distribution in analysis set:")
    print(analysis_df['y_true'].value_counts().describe())
    
    # Initialize NannyML estimator
    estimator = nml.CBPE(
        y_pred_proba={i: f'pred_proba_{i}' for i in range(100)},
        y_pred='y_pred',
        y_true='y_true',
        timestamp_column_name='timestamp',
        metrics=['accuracy'],
        problem_type="classification_multiclass",
        chunk_size=min_samples_per_class * 2  # Ensure chunks have enough samples
    )
    
    # Fit estimator on reference set and estimate on analysis set
    try:
        estimator.fit(reference_df)
        estimated_results = estimator.estimate(analysis_df)
        
        # Plot results
        print("\nModel Performance Monitoring Results:")
        estimated_results.plot().show()
        
        # Print some basic statistics
        print("\nReference Set Statistics:")
        print(f"Accuracy: {(reference_df['y_pred'] == reference_df['y_true']).mean():.3f}")
        print("\nAnalysis Set Statistics:")
        print(f"Accuracy: {(analysis_df['y_pred'] == analysis_df['y_true']).mean():.3f}")
        
    except Exception as e:
        print(f"Error during monitoring: {str(e)}")
        estimated_results = None
    
    return reference_df, analysis_df, estimated_results

if __name__ == "__main__":
    reference_df, analysis_df, estimated_results = main(
        num_samples=500,  # Increased sample size
        min_samples_per_class=10  # Minimum samples per class
    )
