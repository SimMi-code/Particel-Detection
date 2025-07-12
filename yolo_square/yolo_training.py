# yolo_square/yolo_training.py

import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

def train_yolo_model(
    data_yaml="yolo_dataset/data.yaml",
    model_name="yolov8n.pt",
    epochs=50,
    imgsz=640,
    initial_lr=1e-3,
    final_lr_factor=0.2,
    project="yolo_output",
    name="exp",
    device="cpu",
    overwrite=False
):
    """
    Trains a YOLOv8 model and visualizes the training metrics.

    Args:
        data_yaml (str): Path to the data.yaml file.
        model_name (str): Pretrained model to start from.
        epochs (int): Number of epochs to train.
        imgsz (int): Input image size for training.
        initial_lr (float): Initial learning rate (lr0).
        final_lr_factor (float): Final LR factor (lrf), i.e. final_lr = lr0 * lrf.
        project (str): Root folder where runs/exp* are saved.
        name (str): Base name for this run (e.g. "exp").
        device (str): "cpu" or "cuda".
        overwrite (bool): If True, will overwrite an existing project/name folder.
                          If False (default), will auto‐increment to the next available exp#.

    Returns:
        save_dir (str): The full path to this experiment’s output folder.

    """
    print(f"🚀 Starting YOLOv8 training using model: {model_name}")
    
    model = YOLO(model_name)
    # Capture the return from .train()
    result = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        lr0=initial_lr,
        lrf=final_lr_factor,
        project=project,
        name=name,
        exist_ok=overwrite,
        device=device
    )

    # The folder where results are saved
    save_dir = os.path.join(project, name)
    results_csv = os.path.join(save_dir, "results.csv")
    
    if not os.path.exists(results_csv):
        print(f"❌ Training complete but no results.csv found at {results_csv}")
        return save_dir

    print(f"📊 Training complete. Visualizing results from {results_csv}")
    
    # Load results and plot training curves
    df = pd.read_csv(results_csv)
    epochs_range = df["epoch"] + 1  # start from epoch 1

    plt.figure(figsize=(16, 8))

    # Losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, df["train/box_loss"], label="Box Loss")
    plt.plot(epochs_range, df["train/cls_loss"], label="Class Loss")
    plt.plot(epochs_range, df["train/dfl_loss"], label="DFL Loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Validation mAP
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, df["metrics/mAP50(B)"], label="mAP@0.5")
    plt.plot(epochs_range, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
    plt.title("Validation mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()

    # Precision and Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, df["metrics/precision(B)"], label="Precision")
    plt.plot(epochs_range, df["metrics/recall(B)"], label="Recall")
    plt.title("Precision & Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()

    # Val losses
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, df["val/box_loss"], label="Val Box Loss")
    plt.plot(epochs_range, df["val/cls_loss"], label="Val Class Loss")
    plt.plot(epochs_range, df["val/dfl_loss"], label="Val DFL Loss")
    plt.title("Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_summary.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Summary plot saved to: {plot_path}")

    # --- Print out the actual training args used ---
    args_file = os.path.join(save_dir, "args.yaml")
    if os.path.exists(args_file):
        with open(args_file, 'r') as f:
            train_args = yaml.safe_load(f)
        print("\n📝 Final training arguments (from args.yaml):")
        for key in ("batch", "lr0", "lrf", "device", "mosaic", "mixup"):
            val = train_args.get(key, None)
            print(f"    • {key:12s}: {val}")
    else:
        print(f"⚠️  No args.yaml found in {save_dir}; cannot report batch/LR etc.")

    return save_dir
