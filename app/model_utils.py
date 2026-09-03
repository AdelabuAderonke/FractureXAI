import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.cm as cm


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

# add gradcam function to generate heatmap for explainability


def find_vision_tower(model):
    candidates = []
    for name, module in model.named_modules():
        lname = name.lower()
        if "vision_tower" in lname or "vision_model" in lname or "siglip" in lname:
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError("Could not auto-locate the vision tower module.")
    candidates.sort(key=lambda x: len(x[0]))
    return candidates[0][1]


def compute_gradcam(model, processor, image: Image.Image, prompt: str, generated_text: str):
    vision_tower = find_vision_tower(model)
    activations = {}

    def forward_hook(module, inp, out):
        out = out[0] if isinstance(out, tuple) else out
        out.retain_grad()
        activations["value"] = out

    handle = vision_tower.register_forward_hook(forward_hook)

    try:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": generated_text}]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

        model.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits

        input_ids = inputs["input_ids"][0]
        log_probs = torch.log_softmax(logits[0], dim=-1)
        response_token_count = min(
            len(processor.tokenizer(generated_text)["input_ids"]), logits.shape[1] - 1
        )
        target = 0.0
        for t in range(logits.shape[1] - response_token_count - 1, logits.shape[1] - 1):
            target = target + log_probs[t, input_ids[t + 1]]

        target.backward()

        act = activations["value"]
        grad = act.grad
        if grad is None:
            raise RuntimeError("No gradient reached the vision tower.")

        weights = grad.mean(dim=1, keepdim=True)
        cam = (act * weights).sum(dim=-1)
        cam = torch.relu(cam)[0].detach().float().cpu().numpy()

        n = cam.shape[0]
        side = int(round(np.sqrt(n)))
        cam = cam[: side * side].reshape(side, side)

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
    finally:
        handle.remove()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    image_resized = image.convert("RGB")
    w, h = image_resized.size

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), resample=Image.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0

    heatmap = cm.jet(cam_arr)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    base_arr = np.array(image_resized).astype(np.float32)
    heat_arr = heatmap.astype(np.float32)
    blended = (1 - alpha) * base_arr + alpha * heat_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)