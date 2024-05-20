from fastapi import FastAPI
from diffusers import AutoPipelineForText2Image
import torch
import io
import base64
import functools

# NOTE - we configure docs_url to serve the interactive Docs at the root path
# of the app. This way, we can use the docs as a landing page for the app on Spaces.
app = FastAPI(docs_url="/")


@functools.cache
def load_model():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    return pipeline
    
@app.get("/generate")
def generate(text: str):
    with torch.inference_mode():
        pipeline = load_model()
        image = pipeline("cat", num_inference_steps=50).images[0]
         # BytesIO is a file-like buffer stored in memory
        imgByteArr = io.BytesIO()
        # image.save expects a file-like as a argument
        image.save(imgByteArr, format='PNG')
        # Turn the BytesIO object back into a bytes object
        imgByteArr = imgByteArr.getvalue()
        base64_encoded = base64.b64encode(imgByteArr)
        base64_string = base64_encoded.decode('utf-8')
        return {"output": base64_string}