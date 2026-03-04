from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

app = FastAPI(title="Image Alt Text Generator")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.post("/generate-alt")
async def generate_alt(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs)

        caption = processor.decode(out[0], skip_special_tokens=True)

        return {
            "filename": file.filename,
            "alt_text": caption
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))