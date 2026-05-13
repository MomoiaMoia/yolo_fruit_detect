from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ultralytics import __version__
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import select_device


ROOT = Path(__file__).resolve().parent
MODEL_CFG_PATH = ROOT / "cfgs" / "yolov12.yaml"
WEIGHTS_PATH = ROOT / "ckpts" / "model_epoch_600.pth"


def load_detection_model(model_cfg_path: Path, weights_path: Path) -> DetectionModel:
	"""Build a detection model and load a checkpoint saved by the local trainer."""
	model_cfg = yaml.safe_load(model_cfg_path.read_text())
	model_cfg["scale"] = "n"

	model = DetectionModel(cfg=model_cfg, ch=3, verbose=False)
	model.task = "detect"
	
	checkpoint = torch.load(weights_path, map_location="cpu")
	state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
	if isinstance(state_dict, dict):
		model.load_state_dict(state_dict, strict=False)
	else:
		model.load(state_dict)

	model.eval()
	model.float()
	return model


def build_exporter(model: DetectionModel, export_base_path: Path) -> Exporter:
	"""Create an Exporter configured to write artifacts under the onnx directory."""
	exporter = Exporter(
		overrides={
			"format": "saved_model",
			"imgsz": 640,
			"batch": 1,
			"device": "cpu",
			"dynamic": False,
			"half": False,
			"int8": True,
			"simplify": True,
			"verbose": False,
			"data": None,
			"nms": False,
			"agnostic_nms": False,
			"max_det": 300,
			"iou": 0.7,
			"conf": None,
			"split": None,
			"optimize": False,
		}
	)

	exporter.device = select_device("cpu")
	exporter.imgsz = check_imgsz(exporter.args.imgsz, stride=model.stride, min_dim=2)
	exporter.im = torch.zeros(exporter.args.batch, 3, *exporter.imgsz).to(exporter.device)
	exporter.file = export_base_path
	exporter.model = model.to(exporter.device)
	exporter.pretty_name = MODEL_CFG_PATH.stem.replace("yolo", "YOLO")
	exporter.metadata = {
		"description": f"Ultralytics {exporter.pretty_name} model",
		"author": "Ultralytics",
		"date": datetime.now().isoformat(),
		"version": __version__,
		"license": "AGPL-3.0 License (https://ultralytics.com/license)",
		"docs": "https://docs.ultralytics.com",
		"stride": int(max(model.stride)),
		"task": getattr(model, "task", "detect"),
		"batch": exporter.args.batch,
		"imgsz": exporter.imgsz,
		"names": model.names,
		"args": {},
	}
	return exporter


def export_saved_model_to_onnx_dir(model_cfg_path: Path = MODEL_CFG_PATH, weights_path: Path = WEIGHTS_PATH) -> str:
	"""Export a SavedModel plus its ONNX intermediates into the local onnx directory."""
	model = load_detection_model(model_cfg_path, weights_path)
	export_base_path = ROOT / "onnx" / weights_path.name
	export_base_path.parent.mkdir(parents=True, exist_ok=True)

	exporter = build_exporter(model, export_base_path)
	exported_path, _ = exporter.export_saved_model()
	return exported_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export the local YOLO model to TensorFlow SavedModel in onnx/")
	parser.add_argument("--model-cfg", type=Path, default=MODEL_CFG_PATH, help="Path to the model YAML config")
	parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH, help="Path to the trained checkpoint")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	exported_path = export_saved_model_to_onnx_dir(args.model_cfg, args.weights)
	print(f"Exported to: {exported_path}")


if __name__ == "__main__":
	main()
