"""
YOLO Object Detection - Training Module
Fine-tuning YOLOv8 on a small dataset (coco128)
"""

from ultralytics import YOLO
import torch


class YOLOTrainer:
    def __init__(self, model_name="yolov8s.pt"):
        """
        Initialize YOLO trainer

        Args:
            model_name: Pretrained YOLO model to fine-tune
        """
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load pretrained model
        self.model = YOLO(model_name)

    def train(self, epochs=20, imgsz=640):
        """
        Fine-tune YOLO on coco128 dataset

        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
        """
        print("Starting fine-tuning...")

        results = self.model.train(
            data="coco128.yaml",   # Built-in tiny COCO dataset (128 images)
            epochs=epochs,
            imgsz=imgsz,
            device=self.device,
            project="runs",
            name="yolo_finetune"
        )

        print("Training complete.")
        return results

    def validate(self):
        """
        Evaluate model performance
        """
        print("Running validation...")

        metrics = self.model.val()
        print("Validation complete.")
        return metrics

    def save_model(self, path="models/yolo_finetuned.pt"):
        """
        Save the fine-tuned model
        """
        self.model.save(path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    trainer = YOLOTrainer()
    trainer.train(epochs=40)
    trainer.validate()
    trainer.save_model()
