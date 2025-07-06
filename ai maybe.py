from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import ToPILImage
from PIL import Image

dataset = load_dataset("imagefolder", data_dir="aiexperimenting", split="train")

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=dataset.features['label'].num_classes,
    ignore_mismatched_sizes=True
)

to_pil = ToPILImage()

import numpy as np

def transform(example):
    image = example['image']

    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = to_pil(image)

    image = image.convert("RGB")

    inputs = processor(images=[image], return_tensors="pt")

    return {
        'pixel_values': inputs['pixel_values'][0],
        'label': example['label']
    }

dataset = dataset.map(transform)
dataset.set_format(type="torch", columns=["pixel_values", "label"])

training_args = TrainingArguments(
    output_dir="./vit-custom",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=50,

)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

model.save_pretrained("trained-vit-model")
processor.save_pretrained("trained-vit-model")