from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch


model = ViTForImageClassification.from_pretrained("trained-vit-model")
processor = AutoImageProcessor.from_pretrained("trained-vit-model")


image = Image.open("testmodelling/randomcircleofcheese.webp").convert("RGB")


inputs = processor(images=image, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()


from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="aiexperimenting", split="train")
label_names = dataset.features['label'].names


print("Predicted label:", label_names[predicted_class_idx])