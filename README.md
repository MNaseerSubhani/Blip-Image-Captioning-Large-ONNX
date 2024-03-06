# Blip-Image-Captioning-Large-ONNX
This GitHub repository serves as a comprehensive toolkit for converting the Salesforce/blip-image-captioning-large model, originally hosted on Hugging Face, to the ONNX (Open Neural Network Exchange) format.


# Image Captioning with BLIP (Blended Language-Image Pre-training) Model

This repository contains code for performing image captioning using the Salesforce BLIP (Blended Language-Image Pre-training) model. The BLIP model is capable of generating textual descriptions for given images, making it suitable for various vision-language tasks.

## Setup

Before running the code, make sure to install the required dependencies:

```bash
!pip install onnx
!pip install onnxruntime
```

## Usage

Load the Model and Perform Inference
The following code snippet demonstrates how to load the BLIP model and perform image captioning:
```
import numpy as np
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load an example image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Perform image captioning
text = None
inputs = processor(raw_image, text, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)


```

## Convert Model to ONNX Format

The BLIP model can be converted to the ONNX format for deployment. The code snippet below demonstrates how to convert both the vision model and text decoder model:


```
# Convert Vision Model to ONNX
VISION_MODEL_ONNX = 'vision_model.onnx'
vision_model = model.vision_model
# ... (code for exporting vision model to ONNX)

# Convert Text Decoder Model to ONNX
TEXT_DECODER_ONNX = 'text_decoder_model.onnx'
text_decoder_model = model.text_decoder
# ... (code for exporting text decoder model to ONNX)

```