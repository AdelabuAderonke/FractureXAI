# Base model
BASE_MODEL_ID = "google/medgemma-4b-it"

# Fine-tuned adapter paths 
MODEL_A_PATH = "/content/drive/MyDrive/Dissertation/models/medgemma-4b-it-sft-lora-mendeley-final"
MODEL_B_PATH = "/content/drive/MyDrive/Dissertation/models/medgemma-4b-it-sft-lora-grazpedwri-final"
MODEL_B2_PATH = "/content/drive/MyDrive/Dissertation/models/medgemma-4b-it-sft-lora-grazpedwri-final-second"
# Model A: fracture subtype classification
PROMPT_A = (
    "What type of bone fracture is shown in this X-ray?\n"
    "A: simple fracture\n"
    "B: comminuted fracture"
)
FRACTURE_CLASSES_A = ["A: simple fracture", "B: comminuted fracture"]

#  Model B / B2: fracture presence detection

PROMPT_B = (
    "Is there a fracture visible in this pediatric wrist X-ray?\n"
    "A: Yes, fracture present\n"
    "B: No fracture present"
)
FRACTURE_CLASSES_B = ["A: Yes, fracture present", "B: No fracture present"]

DETECTION_MODELS = {
    "High sensitivity (catches more fractures, more false alarms)": MODEL_B_PATH,
    "High specificity (fewer false alarms, may miss subtle fractures)": MODEL_B2_PATH,
}

# App text
APP_TITLE = "FractureXAI"
APP_SUBTITLE = "AI bone fracture detection with Grad-CAM explainability"
RESEARCH_DISCLAIMER = "Research demo — not for clinical use"
SCOPE_DISCLAIMER = (
    "This tool is trained and validated on pediatric wrist X-rays only. "
    "Results on other body parts, imaging types, or adult patients are not validated."
)