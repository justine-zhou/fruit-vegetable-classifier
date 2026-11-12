from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from utils import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_dataset_layout,
    get_broad_category,
    set_seed,
    summarize_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fruit and vegetable image classifiers."
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Dataset root path.")
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=("simple_cnn", "resnet18"),
        help="Model to train. Use simple_cnn as a course baseline and resnet18 as the final transfer-learning model.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Keep 0 on Windows for compatibility.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use ImageNet pretrained weights for ResNet18.",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_false",
        dest="freeze_backbone",
        help="Fine-tune all ResNet18 layers instead of only the final classifier.",
    )
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Low-confidence threshold used by predict.py and gui.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODELS_DIR),
        help="Directory for saved model checkpoints.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Directory for curves, confusion matrix, and metrics.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Optional smoke-test limit for train batches. 0 means no limit.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Optional smoke-test limit for validation/test batches. 0 means no limit.",
    )
    return parser.parse_args()


def load_torch_dependencies():
    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
        from torchvision.models import ResNet18_Weights
    except ImportError as exc:
        raise SystemExit(
            "Missing PyTorch dependency. Run: pip install -r requirements.txt"
        ) from exc

    return torch, nn, optim, DataLoader, datasets, models, transforms, ResNet18_Weights


def build_transforms(image_size, transforms):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_data_loaders(args, DataLoader, datasets, transforms):
    data_dir = Path(args.data_dir)
    train_transform, eval_transform = build_transforms(args.image_size, transforms)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "validation", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and validation class folders do not match.")
    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Train and test class folders do not match.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def create_simple_cnn(num_classes, nn):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.35),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return SimpleCNN()


def create_resnet18(num_classes, pretrained, freeze_backbone, nn, models, ResNet18_Weights):
    loaded_pretrained = False
    if pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            loaded_pretrained = True
            print("Loaded ImageNet pretrained ResNet18 weights.")
        except Exception as exc:
            print(f"Could not load pretrained weights: {exc}")
            print("Falling back to randomly initialized ResNet18.")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    if freeze_backbone and loaded_pretrained:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_model(
    model_name,
    num_classes,
    pretrained,
    freeze_backbone,
    nn,
    models,
    ResNet18_Weights,
):
    if model_name == "simple_cnn":
        return create_simple_cnn(num_classes, nn)
    if model_name == "resnet18":
        return create_resnet18(
            num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            nn=nn,
            models=models,
            ResNet18_Weights=ResNet18_Weights,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch(model, loader, criterion, optimizer, device, torch, max_batches=0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device, torch, max_batches=0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader, start=1):
            if max_batches and batch_index > max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def collect_predictions(model, loader, device, torch, max_batches=0):
    model.eval()
    all_targets = []
    all_predictions = []
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader, start=1):
            if max_batches and batch_index > max_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(probabilities, 1)
            top3 = torch.topk(probabilities, k=min(3, probabilities.shape[1]), dim=1).indices

            top3_correct += (top3 == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return all_targets, all_predictions, top3_correct / max(total, 1)


def save_checkpoint(
    path,
    model,
    args,
    class_to_idx,
    val_accuracy,
    test_accuracy=None,
):
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    broad_categories = {
        class_name: get_broad_category(class_name) for class_name in class_to_idx
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": args.model,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "broad_categories": broad_categories,
        "image_size": args.image_size,
        "confidence_threshold": args.confidence_threshold,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pretrained": not args.no_pretrained,
        "freeze_backbone": args.freeze_backbone,
    }

    import torch

    torch.save(checkpoint, path)


def plot_history(history, output_path):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_test_reports(targets, predictions, class_names, top3_accuracy, results_dir, model_name):
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    labels = list(range(len(class_names)))
    accuracy = accuracy_score(targets, predictions)
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    report_path = results_dir / f"{model_name}_classification_report.txt"
    report_path.write_text(
        f"Top-1 accuracy: {accuracy:.4f}\nTop-3 accuracy: {top3_accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    cm = confusion_matrix(targets, predictions, labels=labels)
    cm_path = results_dir / f"{model_name}_confusion_matrix.csv"
    with cm_path.open("w", encoding="utf-8") as file:
        file.write("," + ",".join(class_names) + "\n")
        for class_name, row in zip(class_names, cm):
            file.write(class_name + "," + ",".join(str(int(value)) for value in row) + "\n")

    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(results_dir / f"{model_name}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    return accuracy


def main():
    args = parse_args()
    data_dir = ensure_dataset_layout(args.data_dir)
    set_seed(42)

    torch, nn, optim, DataLoader, datasets, models, transforms, ResNet18_Weights = (
        load_torch_dependencies()
    )

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset summary:")
    for split, stats in summarize_dataset(data_dir).items():
        print(f"  {split}: {stats['classes']} classes, {stats['images']} images")

    (
        train_dataset,
        _val_dataset,
        _test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_data_loaders(args, DataLoader, datasets, transforms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = train_dataset.classes
    model = build_model(
        args.model,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        nn=nn,
        models=models,
        ResNet18_Weights=ResNet18_Weights,
    ).to(device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_parameters, lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    checkpoint_path = output_dir / f"best_{args.model}.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            torch,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            torch,
            max_batches=args.max_eval_batches,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint_path, model, args, train_dataset.class_to_idx, val_acc)
            print(f"Saved best model to {checkpoint_path}")

    plot_history(history, results_dir / f"{args.model}_training_curve.png")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
        torch,
        max_batches=args.max_eval_batches,
    )
    targets, predictions, top3_acc = collect_predictions(
        model, test_loader, device, torch, max_batches=args.max_eval_batches
    )
    report_acc = save_test_reports(
        targets, predictions, class_names, top3_acc, results_dir, args.model
    )
    save_checkpoint(
        checkpoint_path,
        model,
        args,
        train_dataset.class_to_idx,
        best_val_acc,
        test_accuracy=test_acc,
    )

    metrics = {
        "model": args.model,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "report_accuracy": report_acc,
        "top3_accuracy": top3_acc,
        "class_count": len(class_names),
        "checkpoint": str(checkpoint_path),
    }
    (results_dir / f"{args.model}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Final test results:")
    print(f"  top-1 accuracy: {test_acc:.4f}")
    print(f"  top-3 accuracy: {top3_acc:.4f}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  results: {results_dir}")


if __name__ == "__main__":
    main()
