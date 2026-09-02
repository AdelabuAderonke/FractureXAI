import os
import time
import torch

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from PIL import Image


device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

base_model_id = "google/medgemma-4b-it"

adapter_path = os.path.abspath(
    "models/medgemma-4b-it-sft-lora-mendeley-final"
)

print(f"Adapter path: {adapter_path}")
print(f"Adapter exists: {os.path.isdir(adapter_path)}")

start = time.time()

# Load base model
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    dtype=torch.float32,
    device_map=None,
).to(device)

print("Base model loaded.")

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

print("LoRA adapter loaded.")

# Load processor from BASE model
processor = AutoProcessor.from_pretrained(
    base_model_id
)

print("Processor loaded.")
print(f"Model loaded in {time.time() - start:.1f}s")


# testing
test_image_path = "images/first_image.jpg"
image = Image.open(test_image_path).convert("RGB")

prompt = (
    "What type of bone fracture is shown in this X-ray?\n"
    "A: simple fracture\n"
    "B: comminuted fracture"
)

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

gen_start = time.time()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
print(f"Generation took {time.time() - gen_start:.1f}s")

result = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Result:", result)