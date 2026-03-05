from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)
import io
from pydantic import BaseModel
import os
import uuid
from diffusers import StableDiffusionPipeline

app = FastAPI(title="Image Alt Text + Image Generator")

device = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = r"D:\generatedImages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

translate_model_name = "Helsinki-NLP/opus-mt-en-ru"
translate_tokenizer = MarianTokenizer.from_pretrained(translate_model_name)
translate_model = MarianMTModel.from_pretrained(
    translate_model_name
).to(device)


def translate_to_ru(text: str) -> str:
    inputs = translate_tokenizer(text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        translated = translate_model.generate(**inputs)
    return translate_tokenizer.decode(translated[0], skip_special_tokens=True)

image_model_id = "runwayml/stable-diffusion-v1-5"

image_pipe = StableDiffusionPipeline.from_pretrained(
    image_model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

image_pipe = image_pipe.to(device)

if device == "cuda":
    image_pipe.enable_attention_slicing()

class TextRequest(BaseModel):
    prompt: str


def generate_alt_text(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = blip_model.generate(**inputs)

    caption_en = processor.decode(out[0], skip_special_tokens=True)
    caption_ru = translate_to_ru(caption_en)

    return caption_ru


def generate_image_from_text(prompt: str) -> str:
    with torch.no_grad():
        image = image_pipe(prompt).images[0]

    filename = f"{uuid.uuid4()}.png"
    file_path = os.path.join(OUTPUT_DIR, filename)

    image.save(file_path)

    return file_path

@app.post("/generate-alt")
async def generate_alt(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        alt_text_ru = generate_alt_text(image)

        return {
            "filename": file.filename,
            "alt_text_ru": alt_text_ru,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(request: TextRequest):
    try:
        prompt = request.prompt

        file_path = generate_image_from_text(prompt)

        return {
            "prompt": prompt,
            "saved_to": file_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))