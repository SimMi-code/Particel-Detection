# YOLO Square Object Detection Pipeline

A complete YOLOv8-based pipeline for detecting **round objects** using standardized **square bounding boxes**. Includes:

- Image splitting with overlaping tiles
- Heuristic pre-labeling via circularity
- Manual annotation UI
- Training with Ultralytics YOLOv8
- Inference on full images or tiled images
- Scale bar detection for real-world measurements
- Visualization and performance summaries

---

## 📁 Project Structure

project_root/
│
├── images/ # Source images
│
├── yolo_square/ # Main pipeline code
│ ├── init.py
│ ├── box_utils.py # Square bounding box ops
│ ├── data_preparation.py # Heuristic labeling, splitting, auto-label
│ ├── io_helpers.py # File and path management
│ ├── labeling_ui.py # Manual labeling interface
│ ├── scale_bar.py # Detects physical scale bars
│ ├── visualization.py # Visualization utilities
│ ├── yolo_training.py # YOLOv8 training
│ ├── yolo_detection.py # Detection on full/tiled images
│ └── evaluation.py # Evaluating the yolo-detections
│
├── main.py # Entry point for mode-based execution
├── requirements.txt
└── README.md



Run with:

    python main.py --mode split
    python main.py --mode heuristic
    python main.py --mode label
    python main.py --mode train
    python main.py --mode detect-tiles
    python main.py --mode viz
    python main.py --mode viz-select
    python main.py --mode evaluation

