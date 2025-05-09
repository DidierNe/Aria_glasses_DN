from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load a local image (replace with an Aria frame later)
image = Image.open("IMG_7252.jpeg").convert("RGB")
print(f"{image.size}")
# Prepare input for the model
inputs = processor(images=image, return_tensors="pt")

# Generate a caption
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("🧠 Caption:", caption)
