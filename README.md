# YOLOv8 Object Detection – Fine-Tuning and Benchmarking

This project demonstrates an applied workflow for object detection using YOLOv8.  
It includes fine-tuning a pretrained model, evaluating performance, and benchmarking against the original baseline.

## Overview

The goal of this project was not to train a model from scratch, but to:

- Fine-tune a pretrained YOLOv8 model on a subset dataset (COCO128)
- Compare pretrained vs fine-tuned performance
- Log per-image inference results
- Save structured outputs for visual inspection

This reflects how object detection is typically handled in real-world AI engineering.

## Project Structure

```
yolo-object-detection/
│
├── data/              # Test images
├── models/            # Saved fine-tuned model
├── src/
│   ├── train.py       # Fine-tuning script
│   ├── inference.py   # Model comparison and logging
│   └── utils.py
│
├── outputs/           # Generated results (excluded from Git)
├── requirements.txt
└── README.md
```

## Setup

Create and activate an environment:

```
conda create -n yolo-detection python=3.10
conda activate yolo-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

## Training

Fine-tuning was performed using YOLOv8 on the COCO128 subset dataset.

To train:

```
python src/train.py
```

The best model is saved to:

```
models/yolo_finetuned.pt
```

## Model Comparison

The inference script compares:

- Pretrained YOLOv8n
- Fine-tuned YOLOv8 model

For every image in the `data/` directory, it:

- Runs detection
- Saves bounding box outputs
- Logs inference time and detection details per image

To run comparison:

```
python src/inference.py
```

Logs are saved to:

```
outputs/comparison_log.txt
```

## Notes

- The `runs/` and `outputs/` directories are excluded from version control.
- This project focuses on applied model evaluation and benchmarking.
- Designed to run on consumer hardware (Apple M1 with MPS).

## Author

James Ndopnno-Abasi
