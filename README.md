---
title: Multilanguage Image Captioning
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

This app takes an uploaded image, generates an English caption using a ViT-GPT2 vision-language model, and translates it into Indian languages such as Marathi, Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, and Punjabi.

## Pipeline

Image Upload → Image Preprocessing → ViT-GPT2 Captioning → English Caption → Translation → Indic Language Caption

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Gradio
- deep-translator
- Pillow
- Hugging Face Spaces

## Run Locally

```bash
pip install -r requirements.txt
python app.py