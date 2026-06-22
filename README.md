Multilanguage Image Captioning
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
---

# Multilanguage Image Captioning

Upload an image and generate captions in English and multiple Indic languages.

## What it does

This app takes an uploaded image, generates an English caption using a vision-language model, and translates the caption into Indian languages such as Marathi, Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, and Punjabi.

## Pipeline

```text
Image Upload
    ↓
Image Preprocessing
    ↓
ViT-GPT2 Image Captioning Model
    ↓
English Caption
    ↓
Translation API
    ↓
Indic Language Caption
Tech Stack
Layer	Technology
Model	ViT-GPT2 image captioning
ML Framework	PyTorch
Model Library	Hugging Face Transformers
UI	Gradio
Translation	deep-translator
Deployment	Hugging Face Spaces
Run Locally
pip install -r requirements.txt
python app.py
Built With
Python
PyTorch
Hugging Face Transformers
Gradio
deep-translator
Pillow

Then run:

```powershell
git add README.md
git commit -m "Add README and Space config"
git push space main
git push origin main