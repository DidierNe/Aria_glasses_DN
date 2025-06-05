#This code must be run on your server account 
#wait for the image, run the model on gpu and send the caption throught port 8000


from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

@app.route('/caption', methods=['POST'])
def caption():
    image_file = request.files['image']
    image = Image.open(image_file.stream)
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
# export CUDA_VISIBLE_DEVICES=1
# python bridge_ssh.py
