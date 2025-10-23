import argparse
import os
import pickle
from typing import Iterator, Tuple

import numpy as np
import onnxruntime as ort

import config


def load_cifar_pickle(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    if 'data' in obj and 'labels' in obj:
        data = obj['data']
        labels = np.array(obj['labels'], dtype=np.int64)
    elif 'data' in obj and 'fine_labels' in obj:
        data = obj['data']
        labels = np.array(obj['fine_labels'], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported CIFAR pickle structure at {path}")
    # data: N x 3072 (R(1024) G(1024) B(1024))
    n = data.shape[0]
    data = data.reshape(n, 3, 32, 32)
    return data, labels


def get_val_data(root_dir: str) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    ds = config.DATASET
    if ds == 'CIFAR10':
        test_path = os.path.join(root_dir, 'data', 'cifar-10-batches-py', 'test_batch')
        data, labels = load_cifar_pickle(test_path)
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
        return data, labels, (mean, std)
    if ds == 'CIFAR100':
        test_path = os.path.join(root_dir, 'data', 'cifar-100-python', 'test')
        data, labels = load_cifar_pickle(test_path)
        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape(1, 3, 1, 1)
        return data, labels, (mean, std)
    raise ValueError(f"Unsupported dataset for ONNX demo: {ds}. Use CIFAR10 or CIFAR100.")


def batch_iterator(images: np.ndarray, labels: np.ndarray, batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = images.shape[0]
    for i in range(0, n, batch_size):
        yield images[i:i + batch_size], labels[i:i + batch_size]


def compute_metrics_macro(num_classes: int, preds: np.ndarray, gts: np.ndarray) -> dict:
    assert preds.shape == gts.shape
    total = preds.size
    accuracy = float((preds == gts).sum() / max(total, 1))

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        tp[c] = np.sum((preds == c) & (gts == c))
        fp[c] = np.sum((preds == c) & (gts != c))
        fn[c] = np.sum((preds != c) & (gts == c))

    precision_c = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) != 0)
    recall_c = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) != 0)
    f1_c = np.divide(2 * precision_c * recall_c, precision_c + recall_c, out=np.zeros_like(precision_c), where=(precision_c + recall_c) != 0)

    precision = float(precision_c.mean())
    recall = float(recall_c.mean())
    f1 = float(f1_c.mean())
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }


def run_onnx_eval(model_path: str, batch_size: int, root_dir: str, device: str = "cpu", device_id: int = 0) -> dict:
    # Prepare data
    images_u8, labels, (mean, std) = get_val_data(root_dir)
    images = images_u8.astype(np.float32) / 255.0
    images = (images - mean) / std

    # Create session with providers
    if device == "cuda":
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    preds_all = []
    gts_all = []
    for x, y in batch_iterator(images, labels, batch_size):
        logits = sess.run([output_name], {input_name: x})[0]
        pred = np.argmax(logits, axis=1).astype(np.int64)
        preds_all.append(pred)
        gts_all.append(y)

    preds = np.concatenate(preds_all, axis=0)
    gts = np.concatenate(gts_all, axis=0)
    metrics = compute_metrics_macro(config.NUM_CLASSES, preds, gts)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ONNX inference on CIFAR val set')
    default_model = os.path.join(os.path.dirname(__file__), 'model.onnx')
    parser.add_argument('--model', type=str, default=default_model, help='ONNX 模型路径')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='推理 batch 大小')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='推理设备')
    parser.add_argument('--device-id', type=int, default=0, help='CUDA 设备编号')
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = os.path.dirname(__file__)
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"未找到 ONNX 模型: {args.model}")
    metrics = run_onnx_eval(args.model, args.batch_size, root_dir, device=args.device, device_id=args.device_id)
    print(metrics)


if __name__ == '__main__':
    main()


