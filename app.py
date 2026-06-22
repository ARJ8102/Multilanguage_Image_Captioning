import time
from typing import Tuple

import gradio as gr
import torch
from PIL import Image
from deep_translator import GoogleTranslator
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

LANGUAGES = {
    "Marathi": "mr",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "English Only": "en",
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on: {device}")

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

model.to(device)
model.eval()

GENERATION_CONFIG = {
    "max_length": 20,
    "num_beams": 4,
}


def generate_english_caption(image: Image.Image) -> str:
    """
    Generate an English caption from an input image using ViT-GPT2.
    """
    if image is None:
        raise ValueError("No image provided.")

    image = image.convert("RGB")

    pixel_values = feature_extractor(
        images=[image],
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            **GENERATION_CONFIG
        )

    caption = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return caption.strip()


def translate_caption(caption: str, language_name: str) -> str:
    """
    Translate English caption into the selected target language.
    """
    target_code = LANGUAGES.get(language_name, "mr")

    if target_code == "en":
        return caption

    translated = GoogleTranslator(
        source="en",
        target=target_code
    ).translate(caption)

    return translated


def caption_image(image: Image.Image, language_name: str) -> Tuple[str, str, str]:
    """
    Full pipeline:
    image -> English caption -> translated caption
    """
    start_time = time.time()

    if image is None:
        return "Please upload an image.", "", ""

    try:
        english_caption = generate_english_caption(image)
        translated_caption = translate_caption(english_caption, language_name)

        processing_time = round(time.time() - start_time, 2)

        status = (
            f"Completed successfully in {processing_time} seconds. "
            f"Device used: {device}."
        )

        return english_caption, translated_caption, status

    except Exception as error:
        return "Error while generating caption.", str(error), "Pipeline failed."


custom_css = """
#title {
    text-align: center;
}

#subtitle {
    text-align: center;
    color: #555;
}

.caption-box {
    font-size: 18px;
}
"""


with gr.Blocks(title="IndicVision") as demo:
    gr.Markdown(
        """
        # IndicVision: Multilingual Image Captioning for Indian Languages
        """,
        elem_id="title"
    )

    gr.Markdown(
        """
        Upload an image and generate captions in English and Indian languages using a vision-language model.
        """,
        elem_id="subtitle"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Upload Image"
            )

            language_dropdown = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="Marathi",
                label="Select Output Language"
            )

            generate_button = gr.Button("Generate Caption", variant="primary")

        with gr.Column():
            english_output = gr.Textbox(
                label="English Caption",
                lines=3,
                elem_classes=["caption-box"]
            )

            translated_output = gr.Textbox(
                label="Translated Caption",
                lines=3,
                elem_classes=["caption-box"]
            )

            status_output = gr.Textbox(
                label="Status",
                lines=2
            )

    generate_button.click(
        fn=caption_image,
        inputs=[input_image, language_dropdown],
        outputs=[english_output, translated_output, status_output]
    )

    gr.Markdown(
        """
        ---
        Built with ViT-GPT2, Hugging Face Transformers, Gradio, and Google Translate.
        """
    )


if __name__ == "__main__":
    demo.launch(css=custom_css)