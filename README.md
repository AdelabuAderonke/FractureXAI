# FractureXAI - An Explainable Vision-Language Clinical Decision Support Tool for Bone Fracture Diagnosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io/)
[![MedGemma](https://img.shields.io/badge/Model-MedGemma--4B-4285F4.svg)](https://huggingface.co/google/medgemma-4b-it)

A medical AI diagnostic tool that fine-tunes Google's MedGemma vision-language model to detect and classify bone fractures from radiographs, paired with Grad-CAM visual explanations so clinicians can see which regions of an X-ray drove each prediction.

> ⚠️ **Research prototype only — not validated for clinical use.**

## Features

- **Three Fine-Tuned Models**: independently trained LoRA adapters on a frozen MedGemma-4B base — fracture subtype classification, high-sensitivity fracture detection, and high-specificity fracture detection
- **Explainability**: Grad-CAM adapted for a Vision Transformer (SigLIP-based) vision encoder, producing a heatmap overlay on the original X-ray
- **Base vs. Fine-Tuned Benchmarking**: every model evaluated against the untuned base MedGemma on an identical held-out test set
- **Interactive Dashboard**: Streamlit-based web interface with mode and model selection

## Important: Notebooks Folder
**The `notebooks/` folder contains the standalone experimental workflows used during development.** They can be used to review individual parts of the project without running the full app.

**Used in the final project:**
- `notebooks/data_preparation.ipynb` — dataset loading, YOLO label parsing, label validation, train/val/test splitting
- `notebooks/train_model_a.ipynb` — Mendeley fracture-subtype fine-tuning
- `notebooks/train_model_b.ipynb` — GRAZPEDWRI-DX fracture-detection fine-tuning (Model B and B2)
- `notebooks/evaluation.ipynb` — base vs. fine-tuned model comparison across all three models

**Earlier / superseded:**
- `notebooks/first_version_fine_tune_model.ipynb` — initial combined-dataset model, superseded following supervisor guidance to train each dataset independently; retained for transparency of the project's iterative process

## System Workflow

*(Insert exported architecture diagram: Datasets → Preprocessing → Fine-tuning (Models A/B/B2) → Streamlit App → Grad-CAM Explanation)*

## Quick Start

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU strongly recommended (developed and tested on Google Colab, A100 GPU) — local CPU/MPS inference was found to be impractically slow for a 4B-parameter model
- A Hugging Face account with access to `google/medgemma-4b-it` 

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AdelabuAderonke/FractureXAI.git
cd FractureXAI
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up authentication**
```bash
huggingface-cli login
# paste your Hugging Face access token when prompted
```

5. **Download the fine-tuned model adapters**
Model weights are hosted on Google Drive (not included in this repository due to file size — see [Model Weights](#model-weights) below) and should be placed in a local `models/` folder matching the paths in `app/config.py`, or accessed directly via Drive if running in Colab.

### Running the Application

**Option 1:** Streamlit directly
```bash
streamlit run app/main.py
```

**Option 2:** On Google Colab (recommended, given hardware requirements)
```python
!git clone https://github.com/AdelabuAderonke/FractureXAI.git
%cd FractureXAI
!pip install -r requirements.txt

from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!nohup streamlit run app/main.py --server.port 8501 > /content/logs.txt 2>&1 &

from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
print(ngrok.connect(8501))
```

Navigate to the printed URL (or `http://localhost:8501` if running locally).

## Project Structure



## Step-by-Step User Guide

### Step 1: Choose a Task

When you launch the application, select the task for the model.

*(Insert screenshot: mode selector — "Detect fracture" vs "Classify fracture type")*

**Choose your task:**
- **Detect fracture (pediatric wrist X-ray)** — presence/absence detection, with a choice between a high-sensitivity or high-specificity model
- **Classify fracture type** — assumes a fracture is already present, and identifies it as simple or comminuted

**What Happens:**
- The interface displays a scope disclaimer relevant to the selected task
- The corresponding fine-tuned LoRA adapter is loaded

---

### Step 2: Upload an X-ray

*(Insert screenshot: upload box with an X-ray loaded)*

**What Happens:**
- The image is displayed for confirmation before analysis
- No image is stored beyond the active session

---

### Step 3: Run Analysis

*(Insert screenshot: result badge Fracture / No Fracture)*

**What Happens:**
- The selected model generates a natural-language diagnostic description
- The response is parsed into a clear result (e.g. "Fracture present")

---

### Step 4: Review the GradCAM Explanation

*(Insert screenshot: original image next to GradCAM heatmap overlay)*

**What Happens:**
- A heatmap overlay highlights the image regions that most influenced the model's diagnosis
- Warmer regions indicate greater influence on the generated result

## Model Performance Summary

| Model | Task | Accuracy (fine-tuned) | Accuracy (base MedGemma) |
|---|---|---|---|
| Model A | Fracture subtype classification | 90.0% | 45.7% |
| Model B | Fracture detection (high sensitivity) | 62.9% | 31.9% |
| Model B2 | Fracture detection (high specificity) | 55.8% | 31.9% |

Full methodology, evaluation metrics, and confusion matrices are documented in the accompanying project report.

## Model Weights

Fine-tuned adapter weights (~2.8GB each) are hosted on Google Drive rather than in this repository, due to file size:
- Model A (fracture subtype): *(insert Google Drive share link)*
- Model B (fracture detection, high sensitivity): *(insert Google Drive share link)*
- Model B2 (fracture detection, high specificity): *(insert Google Drive share link)*

Each folder is shared as "Anyone with the link — Viewer". Download the relevant folder(s) and place them under a local `models/` directory matching the paths configured in `app/config.py`, or mount your own copy of Google Drive and update `config.py`'s paths accordingly if running in Colab.

## Datasets

- **[Mendeley Bone Fracture Dataset](https://data.mendeley.com/datasets/8d9kn57pdj/1)** — 2,384 images, categorised as simple or comminuted fractures
- **[GRAZPEDWRI-DX](https://doi.org/10.6084/m9.figshare.14825193)** (Nagy et al., 2022) — 20,327 pediatric wrist trauma radiographs, with image-level and YOLO-format bounding-box annotations

Both datasets are publicly available and de-identified; no primary data collection was undertaken as part of this project.

## Limitations

- Training responses were template-generated from structured labels, not authored by radiologists
- GradCAM explanations have not been independently validated against expert-annotated regions of clinical significance
- Fracture detection (Models B/B2) is validated only on pediatric wrist radiographs and should not be assumed to generalise to other body parts, imaging types, or adult patients


## Acknowledgments

- **Google**: for the MedGemma model family
- **Nagy et al. (2022)**: for the GRAZPEDWRI-DX dataset
- **Talha et al.(2024)**: for Mendeley dataset
- **Hugging Face**: for `transformers`, `peft`, and `trl`
- **Streamlit**: for the application framework

## Contributor

- [Aderonke Adelabu](https://www.linkedin.com/in/aderonke-kausar-adelabu/)

**Supervisor: Dr. Olamilekan Shobayo**
