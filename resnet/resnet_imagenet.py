import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset
from PIL import Image
import os

def load_model_and_processor(model_name="microsoft/resnet-152"):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = ResNetForImageClassification.from_pretrained(model_name)
    return model, image_processor

def process_image(image_path, image_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs

def classify_image(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    return predicted_class_idx

def main(model_name="microsoft/resnet-152"):
    dataset = load_dataset("imagenet-1k",split="val",cache_dir="/media/scalesagx/extra")
    model, image_processor = load_model_and_processor(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for example in dataset:
            image = example["image"]
            inputs = process_image(image, image_processor)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            predicted_class_idx = classify_image(model, inputs)
            predicted_class = model.config.id2label[predicted_class_idx]
            
            print(f"Image ID: {example['image_id']}, Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()

# Print PyTorch and Transformers versions for debugging
