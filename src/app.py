import gradio as gr
import torch
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "spicy03/CLIP-ROCO-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f" Loading Model: {MODEL_ID}...")
try:
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error: {e}")

LABEL_PRESETS = {
    "Imaging Modalities": [
        "chest x-ray",
        "brain MRI scan",
        "spine MRI scan",
        "abdominal CT scan",
        "ultrasound",
        "mammography",
        "knee x-ray",
        "dental x-ray",
        "hand x-ray",
    ],
    "Anatomical Regions": [
        "chest",
        "brain",
        "abdomen",
        "spine",
        "pelvis",
        "knee",
        "dental",
        "hand",
        "leg",
    ],
    "Pathologies": ["normal", "pneumonia", "fracture", "tumor", "edema"],
}


def classify_image(image, label_text, preset_choice):
    if image is None:
        return None, " Please upload an image."

    if preset_choice != "Custom":
        labels = LABEL_PRESETS[preset_choice]
    else:
        labels = [l.strip() for l in label_text.split("\n") if l.strip()]
        if not labels:
            return None, " Enter at least one label."

    try:
        inputs = processor(
            text=labels, images=image, return_tensors="pt", padding=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()

        results = {label: float(prob) for label, prob in zip(labels, probs)}

        top_lbl = max(results, key=results.get)
        interpretation = (
            f"**Top Prediction:** {top_lbl}\n**Confidence:** {results[top_lbl]:.1%}"
        )

        return results, interpretation
    except Exception as e:
        return None, f" Error: {str(e)}"


with gr.Blocks(title="MedCLIP AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ROCO-Radiology AI Assistant")
    gr.Markdown(f"**Model:** `{MODEL_ID}` | **Status:** Live on {DEVICE.upper()}")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Scan")

            preset_radio = gr.Radio(
                choices=["Custom"] + list(LABEL_PRESETS.keys()),
                value="Imaging Modalities",
                label="Select Candidates",
            )

            custom_labels = gr.Textbox(
                label="Custom Labels (One per line)",
                placeholder="pneumonia\nnormal",
                visible=False,
            )

            classify_btn = gr.Button(" Analyze Image", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=5, label="Confidence Scores")
            interpretation = gr.Markdown(label="Interpretation")

    def update_vis(choice):
        return gr.update(visible=(choice == "Custom"))

    preset_radio.change(fn=update_vis, inputs=[preset_radio], outputs=[custom_labels])

    classify_btn.click(
        fn=classify_image,
        inputs=[image_input, custom_labels, preset_radio],
        outputs=[output_label, interpretation],
    )

    gr.Markdown("### Try an Example (Click one to run)")
    gr.Examples(
        examples=[
            ["example_0.jpg", "", "Imaging Modalities"],
            ["example_1.jpg", "", "Anatomical Regions"],
            ["example_2.jpg", "chest x-ray\nbrain MRI\nknee scan", "Custom"],
        ],
        inputs=[image_input, custom_labels, preset_radio],
        outputs=[output_label, interpretation],
        fn=classify_image,
        cache_examples=True,
    )

    gr.Markdown("---")
    gr.Markdown(
        " **Disclaimer:** For research/demo purposes only. Not for clinical use."
    )

print(" Launching App...")
demo.launch(share=True, debug=True)
