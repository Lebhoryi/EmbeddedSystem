import argparse
import os
import torch

# Reuse model and config from project
import config
from main import CNNMoETransformer


def infer_image_size(dataset_name: str) -> tuple[int, int]:
    dataset_name = (dataset_name or "").upper()
    if dataset_name in ["SVHN", "CIFAR10", "CIFAR100"]:
        return 32, 32
    return 28, 28


def build_model(device: str = "cpu") -> torch.nn.Module:
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


def make_dummy_input(batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
    h, w = infer_image_size(getattr(config, "DATASET", ""))
    c = config.INPUT_CHANNELS
    return torch.randn(batch_size, c, h, w, device=device)


def export_onnx(output_path: str, opset: int = 17, batch_size: int = 1, device: str = "cpu", weights_path: str | None = None) -> None:
    model = build_model(device)

    # Load trained weights if provided
    if weights_path:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"未找到权重文件: {weights_path}")
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"已加载权重: {weights_path}")
    dummy = make_dummy_input(batch_size=batch_size, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    print(f"ONNX 导出成功: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CNN+MoE Transformer to ONNX")
    default_out = os.path.join(os.path.dirname(__file__), "model.onnx")
    default_weights = os.path.join(os.path.dirname(__file__), "checkpoints", "model_best.pth")
    parser.add_argument("--output", type=str, default=default_out, help="输出 ONNX 文件路径")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset 版本")
    parser.add_argument("--batch-size", type=int, default=1, help="导出时的 batch 大小（动态维度）")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="导出使用的设备")
    parser.add_argument("--weights", type=str, default=default_weights, help="可选: 训练权重 .pth 路径（若存在将自动加载）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    weights = args.weights if (args.weights and os.path.isfile(args.weights)) else None
    if args.weights and not weights:
        print(f"提示: 指定的权重未找到，将导出随机初始化权重: {args.weights}")
    export_onnx(output_path=args.output, opset=args.opset, batch_size=args.batch_size, device=device, weights_path=weights)

# source /home/zhangy89/work/.venv-moe/bin/activate
# python /home/zhangy89/work/Mixture-of-Experts-MoE/export_onnx.py --output /home/zhangy89/work/Mixture-of-Experts-MoE/model.onnx --opset 18 --batch-size 1 --device cpu