import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

import config
from main import CNNMoETransformer


def build_model(device: str) -> torch.nn.Module:
    model = CNNMoETransformer(
        input_channels=config.INPUT_CHANNELS,
        num_classes=config.NUM_CLASSES,
        num_experts=config.NUM_EXPERTS,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        k=getattr(config, "TOP_K", 2),
    ).to(device)
    model.eval()
    return model


def get_val_loader() -> DataLoader:
    if config.DATASET == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        val_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif config.DATASET == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        val_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif config.DATASET == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        val_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif config.DATASET == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        val_data = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {config.DATASET}")

    return DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)


@torch.no_grad()
def evaluate(model: torch.nn.Module, device: str, loader: DataLoader) -> dict:
    criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=config.NUM_CLASSES).to(device)
    precision = Precision(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
    f1 = F1Score(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)

    model.eval()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        accuracy.update(logits, labels)
        precision.update(logits, labels)
        recall.update(logits, labels)
        f1.update(logits, labels)

    n = len(loader)
    return {
        'val_loss': total_loss / n,
        'accuracy': accuracy.compute().item(),
        'precision': precision.compute().item(),
        'recall': recall.compute().item(),
        'f1_score': f1.compute().item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on validation set")
    parser.add_argument('--weights', type=str, default=os.path.join('checkpoints', 'model_best.pth'), help='权重路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'], help='推理设备')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    model = build_model(device)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"未找到权重: {args.weights}")
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=True)

    loader = get_val_loader()
    metrics = evaluate(model, device, loader)
    print(metrics)


if __name__ == '__main__':
    main()


