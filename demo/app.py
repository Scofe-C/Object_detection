import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

# ==============================================================================
# CONFIG - Update these paths before running
# ==============================================================================
# Path to your trained YOLOv8 weights (relative to this script or absolute)
# Option A: If best.pt is in ../models/
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
# Option B: Absolute path (uncomment and edit if Option A doesn't work)
# MODEL_PATH = r"D:\NEU\IE7615\models\best.pt"

# Example images directory (optional - demo works without these)
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

# ==============================================================================
# CLASS IDS - from data.yaml (39 classes)
# ==============================================================================
CLASS_IDS = [
    'OBJ001', 'OBJ002', 'OBJ003', 'OBJ004', 'OBJ005', 'OBJ006', 'OBJ007', 'OBJ008', 'OBJ009', 'OBJ010',
    'OBJ012', 'OBJ016', 'OBJ018', 'OBJ019', 'OBJ021', 'OBJ022', 'OBJ027', 'OBJ028', 'OBJ029', 'OBJ031',
    'OBJ061', 'OBJ069', 'OBJ090', 'OBJ095', 'OBJ107', 'OBJ108', 'OBJ111', 'OBJ159', 'OBJ208', 'OBJ222',
    'OBJ229', 'OBJ230', 'OBJ300', 'OBJ311', 'OBJ405', 'OBJ786', 'OBJ787', 'OBJ788', 'OBJ789'
]

# ==============================================================================
# LOAD MODEL
# ==============================================================================
print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {os.path.abspath(MODEL_PATH)}\n"
        f"Please download best.pt from Colab and place it in the models/ folder."
    )
model = YOLO(MODEL_PATH)
print(f"Model loaded successfully. Classes: {len(model.names)}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_obj_id(class_idx):
    """Convert class index to OBJ ID format"""
    return CLASS_IDS[class_idx] if class_idx < len(CLASS_IDS) else f'OBJ{class_idx:03d}'


def detect_single_object(image, conf_threshold):
    """
    Detect single object in image.
    Returns: annotated_image, detection_info (markdown)
    """
    if image is None:
        return None, "Please upload an image"

    results = model(image, conf=conf_threshold)
    result = results[0]

    if len(result.boxes) == 0:
        return np.array(image), "⚠️ No objects detected. Try lowering the confidence threshold."

    annotated_img = result.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Get highest confidence detection
    boxes = result.boxes
    best_idx = boxes.conf.argmax()

    class_id = int(boxes.cls[best_idx])
    confidence = float(boxes.conf[best_idx])
    bbox = boxes.xyxy[best_idx].cpu().numpy()
    obj_id = get_obj_id(class_id)
    class_name = model.names[class_id]

    detection_info = f"""
🎯 **SINGLE OBJECT DETECTION RESULT**

✅ **Object ID**: {obj_id}
📝 **Class Name**: {class_name}
📊 **Confidence**: {confidence:.2%}
📍 **Bounding Box**: [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]

---
**Total Objects Detected**: {len(boxes)}
ℹ️ Showing highest confidence detection
"""
    return annotated_img, detection_info


def detect_multiple_objects(image, conf_threshold, iou_threshold):
    """
    Detect multiple objects in image.
    Returns: annotated_image, detection_table (DataFrame), summary (markdown)
    """
    if image is None:
        return None, None, "Please upload an image"

    results = model(image, conf=conf_threshold, iou=iou_threshold)
    result = results[0]

    if len(result.boxes) == 0:
        return np.array(image), None, "⚠️ No objects detected. Try lowering the confidence threshold."

    annotated_img = result.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Build detection table
    detection_data = []
    boxes = result.boxes

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        bbox = boxes.xyxy[i].cpu().numpy()
        obj_id = get_obj_id(class_id)
        class_name = model.names[class_id]

        detection_data.append({
            'Detection #': i + 1,
            'Object ID': obj_id,
            'Class Name': class_name,
            'Confidence': f"{confidence:.2%}",
            'Bounding Box (x1, y1, x2, y2)': f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
        })

    df = pd.DataFrame(detection_data)

    avg_conf = boxes.conf.mean()
    unique_classes = len(set([int(c) for c in boxes.cls]))

    summary = f"""
🎯 **MULTIPLE OBJECT DETECTION SUMMARY**

✅ **Total Objects Detected**: {len(boxes)}
🏷️ **Unique Classes**: {unique_classes}
📊 **Average Confidence**: {avg_conf:.2%}
⚙️ **Confidence Threshold**: {conf_threshold}
🔄 **IoU Threshold**: {iou_threshold}

---
📋 Detailed detection results shown in table below
"""
    return annotated_img, df, summary


# ==============================================================================
# GRADIO UI
# ==============================================================================
custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-image {
        border: 2px solid #4CAF50;
        border-radius: 8px;
    }
    footer {
        display: none !important;
    }
"""

# Build example lists (only include files that exist)
single_examples = []
multi_examples = []
if os.path.isdir(EXAMPLES_DIR):
    for fname in sorted(os.listdir(EXAMPLES_DIR)):
        fpath = os.path.join(EXAMPLES_DIR, fname)
        if "single" in fname.lower():
            single_examples.append([fpath, 0.25])
        elif "multi" in fname.lower():
            multi_examples.append([fpath, 0.25, 0.5])

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔍 YOLOv8 Object Detection System
    ### Milestone 3: Live Demonstration

    This interactive demo showcases **custom-trained YOLOv8** model for multi-object detection.
    - **39 Object Classes** (OBJ001 through OBJ789)
    - **Model Performance**: mAP@0.5 = 95.2%, mAP@0.5:0.95 = 86.5%
    - **Trained on**: 4,000 synthetic composite images

    ---
    """)

    with gr.Tabs():
        # ── Tab 1: Single Object Detection ──────────────────────────────
        with gr.Tab("🎯 Single Object Detection"):
            gr.Markdown("""
            ### Test Case A: Single Object Identification
            Upload an image containing **one object** and the model will identify its **Object ID** and **Class Name**.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    single_input = gr.Image(type="pil", label="Upload Single Object Image")
                    single_conf = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                        label="Confidence Threshold"
                    )
                    single_btn = gr.Button("🔍 Detect Object", variant="primary", size="lg")

                with gr.Column(scale=1):
                    single_output_img = gr.Image(label="Detection Result", elem_classes="output-image")
                    single_output_text = gr.Markdown(label="Detection Details")

            single_btn.click(
                fn=detect_single_object,
                inputs=[single_input, single_conf],
                outputs=[single_output_img, single_output_text]
            )

            if single_examples:
                gr.Markdown("### 📸 Try Example Images:")
                gr.Examples(
                    examples=single_examples,
                    inputs=[single_input, single_conf],
                    label="Click to load example"
                )

        # ── Tab 2: Multiple Object Detection ────────────────────────────
        with gr.Tab("🎯🎯🎯 Multiple Object Detection"):
            gr.Markdown("""
            ### Test Case B: Multiple Object Detection & Localization
            Upload an image with **multiple objects** and the model will:
            - Identify all **Object IDs**
            - Provide **Bounding Box** coordinates for each object
            - Display **Confidence Scores**
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    multi_input = gr.Image(type="pil", label="Upload Multi-Object Image")
                    with gr.Row():
                        multi_conf = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                            label="Confidence Threshold"
                        )
                        multi_iou = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                            label="IoU Threshold"
                        )
                    multi_btn = gr.Button("🔍 Detect All Objects", variant="primary", size="lg")

                with gr.Column(scale=1):
                    multi_output_img = gr.Image(label="Detection Result", elem_classes="output-image")
                    multi_output_summary = gr.Markdown(label="Summary")

            multi_output_table = gr.DataFrame(
                label="📊 Detailed Detection Results",
                headers=['Detection #', 'Object ID', 'Class Name', 'Confidence', 'Bounding Box (x1, y1, x2, y2)'],
                wrap=True
            )

            multi_btn.click(
                fn=detect_multiple_objects,
                inputs=[multi_input, multi_conf, multi_iou],
                outputs=[multi_output_img, multi_output_table, multi_output_summary]
            )

            if multi_examples:
                gr.Markdown("### 📸 Try Example Images:")
                gr.Examples(
                    examples=multi_examples,
                    inputs=[multi_input, multi_conf, multi_iou],
                    label="Click to load example"
                )

    # ── Footer ───────────────────────────────────────────────────────
    gr.Markdown("""
    ---
    ### 📊 Model Information
    - **Architecture**: YOLOv8s (11.1M parameters)
    - **Dataset**: 4,000 synthetic images (3,500 train / 400 val / 100 test)
    - **Classes**: 39 objects (Trigger Wallet, Water Bottle, Ear buds, Book, etc.)
    - **Performance Metrics**:
        - Precision: 84.6%
        - Recall: 92.3%
        - mAP@0.5: 95.2%
        - mAP@0.5:0.95: 86.5%

    **Developed by**: Zhengxin Chen - Yi Liu - Samet Temurcin - Greta Wang | Northeastern University | Milestone 3 Submission
    """)


# ==============================================================================
# LAUNCH
# ==============================================================================
if __name__ == "__main__":
    demo.launch(
        share=True,     # Set to False if you only need localhost
        debug=True,
        show_error=True
    )
