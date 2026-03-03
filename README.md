# YOLO Object Detection

A comprehensive YOLO-based object detection project for training, inference, and deployment.

## Project Structure

```
yolo-object-detection/
│
├── data/              # Dataset storage
├── models/            # Trained model checkpoints
├── src/               # Source code
│   ├── inference.py   # Inference module
│   ├── train.py       # Training module
│   └── utils.py       # Utility functions
│
├── outputs/           # Output results (images, videos, logs)
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your dataset in the `data/` directory with the following structure:

```
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Usage

### Training

```python
from src.train import YOLOTrainer

config = {
    'batch_size': 16,
    'learning_rate': 0.001,
    'img_size': 640
}

trainer = YOLOTrainer(config)
trainer.setup_data('data/')
trainer.build_model(model_config)
trainer.train(epochs=100, save_dir='models/')
```

### Inference

```python
from src.inference import YOLOInference

# Initialize inference
detector = YOLOInference(model_path='models/best.pt')
detector.load_model()

# Predict on image
results = detector.predict_image('path/to/image.jpg')

# Predict on video
detector.predict_video('path/to/video.mp4', 'outputs/result.mp4')
```

## Features

- ✅ Model training and fine-tuning
- ✅ Image and video inference
- ✅ Visualization utilities
- ✅ Custom dataset support
- ✅ Model checkpointing
- ✅ TensorBoard logging

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## TODO

- [ ] Implement training loop
- [ ] Add data augmentation
- [ ] Implement inference pipeline
- [ ] Add evaluation metrics
- [ ] Create demo scripts
- [ ] Add model export (ONNX)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
