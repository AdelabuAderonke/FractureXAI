import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
import streamlit as st

from config import BASE_MODEL_ID


@st.cache_resource(show_spinner="Loading model...")
def load_model(adapter_path: str):
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.tokenizer.padding_side = "right"
    return model, processor


def generate_response(image: Image.Image, prompt: str, model, processor, max_new_tokens: int = 30):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def postprocess(response_text: str, class_labels: list[str]):
    for i, label in enumerate(class_labels):
        if label in response_text:
            return i, label
    return None, response_text 