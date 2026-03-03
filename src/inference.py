import os
import time
import torch
from ultralytics import YOLO

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
LOG_FILE = os.path.join(OUTPUT_DIR, "comparison_log.txt")
CONF_THRESHOLD = 0.25


def run_inference(model, image_path, label, save_dir):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    start_time = time.time()
    results = model.predict(
        image_path,
        conf=CONF_THRESHOLD,
        device=device,
        save=True,
        project=OUTPUT_DIR,
        name=save_dir,
        exist_ok=True,
        verbose=False
    )
    end_time = time.time()

    inference_time = end_time - start_time
    detections = results[0].boxes

    return inference_time, detections


def log_results(image_name, label, inference_time, detections, model):
    with open(LOG_FILE, "a") as f:
        f.write(f"\nImage: {image_name}\n")
        f.write(f"Model: {label}\n")
        f.write(f"Inference Time: {inference_time:.4f} seconds\n")
        f.write(f"Detections: {len(detections)}\n")

        for box in detections:
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls_id]
            f.write(f" - Class: {class_name}, Confidence: {conf:.2f}\n")

        f.write("-" * 40 + "\n")


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear old log
    open(LOG_FILE, "w").close()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load models once
    pretrained_model = YOLO("yolov8n.pt")
    finetuned_model = YOLO("models/yolo_finetuned.pt")

    image_files = [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    for image_file in image_files:
        image_path = os.path.join(DATA_DIR, image_file)

        print(f"\nProcessing: {image_file}")

        # Pretrained
        pt_time, pt_detections = run_inference(
            pretrained_model,
            image_path,
            "Pretrained YOLOv8n",
            "pretrained"
        )
        log_results(image_file, "Pretrained YOLOv8n", pt_time, pt_detections, pretrained_model)

        # Fine-tuned
        ft_time, ft_detections = run_inference(
            finetuned_model,
            image_path,
            "Fine-Tuned YOLOv8",
            "finetuned"
        )
        log_results(image_file, "Fine-Tuned YOLOv8", ft_time, ft_detections, finetuned_model)

    print("\nComparison complete.")
    print(f"Logs saved to: {LOG_FILE}")
    print("Output images saved inside outputs/pretrained and outputs/finetuned")