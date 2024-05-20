from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image
from TTS.api import TTS
import whisper

import tempfile
import torch
import io
import base64
import functools
import sys, os

os.environ["COQUI_TOS_AGREED"] = "1"

class Text2SpeechRequest(BaseModel):
    text: str
    speaker_wav_base64: str

class Speech2TextRequest(BaseModel):
    speech_audio_base64: str
       
# NOTE - we configure docs_url to serve the interactive Docs at the root path
# of the app. This way, we can use the docs as a landing page for the app on Spaces.
app = FastAPI(docs_url="/")


@functools.cache
def load_sd_model():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    return pipeline

@functools.cache
def load_tts_model():
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

@functools.cache
def load_whisper_model():
    return whisper.load_model("small", model_dir="/whisper")
   
@app.get("/text2image")
def text2image(
    prompt: str, 
    width:int, 
    height:int):
    pipeline = load_sd_model()
    
    with torch.inference_mode():
        image = pipeline(prompt, num_inference_steps=50,width=width, height=height).images[0]
         # BytesIO is a file-like buffer stored in memory
        imgByteArr = io.BytesIO()
        # image.save expects a file-like as a argument
        image.save(imgByteArr, format='PNG')
        # Turn the BytesIO object back into a bytes object
        imgByteArr = imgByteArr.getvalue()
        base64_encoded = base64.b64encode(imgByteArr)
        base64_string = base64_encoded.decode('utf-8')
        return {"data": base64_string}
    
    
@app.post("/text2speech")
def text2speech(text2SpeechRequest: Text2SpeechRequest):
    with tempfile.TemporaryDirectory() as temp_dir:
        speaker_wav_file = temp_dir +"/temp.wav"
        with open(speaker_wav_file, "wb") as wav_file:
            decode_string = base64.b64decode(text2SpeechRequest.speaker_wav_base64)
            wav_file.write(decode_string)

            output_wav_file = temp_dir +"/output.wav"
            tts = load_tts_model()
            tts.tts_to_file(text=text2SpeechRequest.text, speaker_wav=speaker_wav_file, language="en", file_path = output_wav_file)
            return {"data": base64.b64encode(open(output_wav_file, "rb").read())}
    
@app.post("/speech2text")
def text2speech(speech2TextRequest: Speech2TextRequest):
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_file_path = temp_dir +"/temp.wav"
        with open(wav_file_path, "rb") as wav_file:
            decode_string = base64.b64decode(speech2TextRequest.speech_audio_base64)
            wav_file.write(decode_string)
            whisper = load_whisper_model()

            result = whisper.transcribe(wav_file_path, word_timestamps=True)
            return {"data": result }