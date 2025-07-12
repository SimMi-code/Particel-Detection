# main.py

import argparse
import os
import matplotlib
matplotlib.use("Agg")

from yolo_square.data_preparation import split_images_into_tiles_with_overlap, create_yolo_dataset_square, select_top_k_tiles_per_image
from yolo_square.yolo_training import train_yolo_model
from yolo_square.yolo_detection import detect_with_yolo, detect_full_with_tiles
from yolo_square.scale_bar import detect_scale_bar
from yolo_square.visualization import visualize_yolo_labels, update_yolo_viz, evaluate_detections
from yolo_square.evaluation import evaluate_detections
from yolo_square.labeling_ui import labeling_ui
from yolo_square.io_helpers import get_image_filenames

def main():
    parser = argparse.ArgumentParser(description="YOLO Round Object Detection Pipeline (Square Mode)")
    parser.add_argument("--mode", required=True, choices=["split", "heuristic", "train", "label", "viz", "viz_select", "detect-tiles", "evaluation"], help="Pipeline step to run")
    parser.add_argument("--input_dir", default="images", help="Directory with initial full images")
    parser.add_argument("--output_dir", default="images_split", help="Where to save tiles")
    parser.add_argument("--output_dir_detect", default="yolo_predictions", help="Where to save output from detection and evaluation")
    parser.add_argument("--dataset_dir", default="yolo_dataset", help="Directory for YOLO dataset")
    parser.add_argument("--select_dir", default="yolo_dataset_select", help="For manually selected tiles")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path to .pt file")
    parser.add_argument("--exp_name", default="exp", help="YOLO experiment name")
    parser.add_argument("--overlap", type=float, default=0.15, help="Tile overlap fraction")
    parser.add_argument("--tile_size", type=int, default=640, help="Tile size (square)")
    parser.add_argument("--device", default="cpu", help="Device: cuda or cpu")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--rows", type=int, default=6, help="number of tile‐rows")
    parser.add_argument("--cols", type=int, default=6, help="number of tile‐columns")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshould for detections")
    parser.add_argument("--iou", type=float, default=0.5, help="Intersection-over-Union (IoU) cutoff (overlapping boxes threshould)")
    parser.add_argument("--label-img-dir",
        default="yolo_dataset_select/images",
        help="Which folder of tile‐images to label")
    parser.add_argument("--label-txt-dir",
        default="yolo_dataset_select/labels",
        help="Where the .txt files live")
    args = parser.parse_args()

    inputnames_ = get_image_filenames(args.input_dir)
    inputnames_ = inputnames_
    if args.mode == "split":

        split_images_into_tiles_with_overlap(
	     inputnames_,
	     output_folder=args.output_dir,
	     rows=args.rows,
	     cols=args.cols,
	     overlap=args.overlap
	)

    elif args.mode == "heuristic":
        tile_inputnames_ = get_image_filenames(args.output_dir)
        # Heuristically label *all* tiles
        create_yolo_dataset_square(
            inputnames_=tile_inputnames_,
            image_dir="images_split",
            dataset_dir="yolo_dataset",
            auto_tune=True
        )

        # From each original image, pick top-5 tiles
        select_top_k_tiles_per_image(
            inputnames_,
            split_dir="images_split",
            label_dir="yolo_dataset/labels",
            output_sel="yolo_dataset_select",
            k=5
        )

    elif args.mode == "viz_select":
        update_yolo_viz(
            dataset_dir="yolo_dataset_select",
            image_dir="images_split",
            label_dir="yolo_dataset_select/labels",
            viz_dir="yolo_dataset_select/viz"
        )

    elif args.mode == "train":
        from yolo_square.data_preparation import prepare_selected_yolo_dataset
        data_yaml = prepare_selected_yolo_dataset(args.select_dir)
        result = train_yolo_model(
            data_yaml=data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            initial_lr=1e-1,
            final_lr_factor=0.2,
            project="yolo_output",
            name=args.exp_name,
            overwrite=False,
            device=args.device
        )


    elif args.mode == "detect-tiles":
        # 1) run detection
        results = detect_full_with_tiles(
            inputnames_,
            model_path=None,
            input_dir=args.input_dir,
            output_dir=args.output_dir_detect,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )

        # 2) run evaluation
        evaluate_detections(output_dir=args.output_dir_detect, 
                            original_dir="images",
                            rewrite_eval = True)

    elif args.mode == "evaluation":
        evaluate_detections(output_dir=args.output_dir_detect, 
                            original_dir="images",
                            rewrite_eval = False)

    elif args.mode == "label":
        labeling_ui(
            image_folder=args.label_img_dir,
            label_folder=args.label_txt_dir
        )
        update_yolo_viz(
            dataset_dir="yolo_dataset_select",
            image_dir="images_split",
            label_dir="yolo_dataset_select/labels",
            viz_dir="yolo_dataset_select/viz"
        )
    elif args.mode == "viz":
        from os.path import join
        visualize_yolo_labels(
            image_dir=join(args.dataset_dir, "images/train"),
            label_dir=join(args.dataset_dir, "labels/train"),
            output_dir=join(args.dataset_dir, "viz")
        )
   
if __name__ == "__main__":
    main()

