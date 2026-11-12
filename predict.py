from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from train import build_model, build_transforms, load_torch_dependencies
from utils import DEFAULT_MODEL_PATH, get_broad_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one fruit or vegetable image with a trained checkpoint."
    )
    parser.add_argument("--image", required=True, help="Path to one image.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a saved .pth checkpoint.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to show.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override checkpoint confidence threshold for unknown-image warning.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device.",
    )
    return parser.parse_args()


def torch_load_checkpoint(torch, model_path, map_location):
    try:
        return torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=map_location)


def choose_device(torch, requested_device):
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_prediction_bundle(model_path=DEFAULT_MODEL_PATH, device_name="auto"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. Run python train.py first."
        )

    torch, nn, _optim, _DataLoader, _datasets, models, transforms, ResNet18_Weights = (
        load_torch_dependencies()
    )
    device = choose_device(torch, device_name)
    checkpoint = torch_load_checkpoint(torch, model_path, map_location=device)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {
        int(idx): class_name for idx, class_name in checkpoint["idx_to_class"].items()
    }
    model = build_model(
        checkpoint["model_name"],
        num_classes=len(class_to_idx),
        pretrained=False,
        freeze_backbone=False,
        nn=nn,
        models=models,
        ResNet18_Weights=ResNet18_Weights,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _train_transform, eval_transform = build_transforms(
        checkpoint.get("image_size", 224), transforms
    )

    return {
        "torch": torch,
        "device": device,
        "model": model,
        "transform": eval_transform,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "threshold": checkpoint.get("confidence_threshold", 0.65),
        "model_name": checkpoint.get("model_name", "unknown"),
        "model_path": model_path,
    }


def predict_image(image_path, bundle, top_k=3, threshold=None):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    torch = bundle["torch"]
    device = bundle["device"]
    model = bundle["model"]
    transform = bundle["transform"]
    idx_to_class = bundle["idx_to_class"]
    threshold = bundle["threshold"] if threshold is None else threshold

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_k = max(1, min(top_k, len(idx_to_class)))
        values, indices = torch.topk(probabilities, k=top_k)

    top_predictions = []
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        class_name = idx_to_class[int(index)]
        top_predictions.append(
            {
                "class_name": class_name,
                "broad_category": get_broad_category(class_name),
                "confidence": float(value),
            }
        )

    best = top_predictions[0]
    is_uncertain = best["confidence"] < threshold
    if is_uncertain:
        message = (
            "Low confidence. This image may be outside the known fruit/vegetable classes."
        )
    else:
        message = "Prediction is confident."

    return {
        "image": str(image_path),
        "predicted_class": best["class_name"],
        "broad_category": best["broad_category"],
        "confidence": best["confidence"],
        "threshold": threshold,
        "is_uncertain": is_uncertain,
        "message": message,
        "top_k": top_predictions,
    }


def print_prediction(result):
    print(f"Image: {result['image']}")
    print(f"Broad category: {result['broad_category']}")
    print(f"Specific class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Threshold: {result['threshold']:.2f}")
    print(f"Warning: {result['message']}")
    print("Top predictions:")
    for item in result["top_k"]:
        print(
            f"  {item['class_name']:>16s} "
            f"({item['broad_category']}): {item['confidence']:.2%}"
        )


def main():
    args = parse_args()
    bundle = load_prediction_bundle(args.model_path, device_name=args.device)
    result = predict_image(
        args.image,
        bundle,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    print_prediction(result)


if __name__ == "__main__":
    main()
