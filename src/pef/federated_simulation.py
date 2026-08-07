import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from pef.collaborative_folding import CollaborativeFoldingProtocol
from pef.config import *
from pef.lattice_utils import LatticeParams
from pef.representation import (
    ResourceType,
    TrustedRepresentationGenerator,
    extract_algo_attributes,
    extract_comp_attributes,
    extract_data_attributes,
)
from pef.shapley_enhancement import adaptive_security_enhancement

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, image_size=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        feat_size = image_size // 8
        self.flat_dim = 256 * feat_size * feat_size
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


def dirichlet_split(dataset, num_clients: int, num_classes: int, alpha: float = DIRICHLET_ALPHA):
    targets = np.array(dataset.targets)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_indices[i].extend(class_indices[start:end].tolist())
            start = end
    return client_indices


def get_data_loaders(dataset_name="cifar10"):
    cfg = DATASETS[dataset_name]
    size = cfg["image_size"]
    base = [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    if cfg["augment"]:
        train_tf = transforms.Compose(
            [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()] + base
        )
    else:
        train_tf = transforms.Compose(base)
    test_tf = transforms.Compose(base)

    if dataset_name == "cifar10":
        ctor = torchvision.datasets.CIFAR10
    elif dataset_name == "mnist":
        ctor = torchvision.datasets.MNIST
    elif dataset_name == "fashion_mnist":
        ctor = torchvision.datasets.FashionMNIST
    else:
        raise ValueError(f"unknown dataset {dataset_name}")

    trainset = ctor(root=DATA_DIR, train=True, download=True, transform=train_tf)
    testset = ctor(root=DATA_DIR, train=False, download=True, transform=test_tf)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE * EVAL_BATCH_MULT,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return trainset, test_loader


def local_train(
    model: nn.Module,
    dataloader,
    epochs: int = LOCAL_EPOCHS,
    scaler: "torch.cuda.amp.GradScaler" = None,
):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LOCAL_LR, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    use_amp = USE_AMP and DEVICE.type == "cuda"
    total_loss = 0.0
    total_samples = 0
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(epochs):
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if USE_CHANNELS_LAST and DEVICE.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-6)
    throughput = total_samples / elapsed
    if DEVICE.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_mem_mb = 0.0
    telemetry = {
        "train_time": elapsed,
        "throughput": throughput,
        "peak_mem_mb": peak_mem_mb,
    }
    return total_loss / max(total_samples, 1), telemetry


def evaluate(model: nn.Module, test_loader):
    model.eval()
    use_amp = USE_AMP and DEVICE.type == "cuda"
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            if USE_CHANNELS_LAST and DEVICE.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def compute_client_data_stats(loader, num_classes):
    label_counts = np.zeros(num_classes, dtype=np.float64)
    pixel_sum = 0.0
    pixel_sq = 0.0
    pixel_n = 0
    n_samples = 0
    for images, labels in loader:
        for c in labels.tolist():
            label_counts[c] += 1
        n_samples += labels.size(0)
        pixel_sum += images.sum().item()
        pixel_sq += (images**2).sum().item()
        pixel_n += images.numel()
    p = label_counts / max(label_counts.sum(), 1.0)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    mean = pixel_sum / max(pixel_n, 1)
    var = pixel_sq / max(pixel_n, 1) - mean**2
    std = float(np.sqrt(max(var, 0.0)))
    return {"num_samples": n_samples, "entropy": entropy, "pixel_mean": mean, "pixel_std": std}


def federated_round(
    global_model,
    client_models,
    client_loaders,
    repr_gen,
    folding_protocol,
    round_idx,
    num_rounds,
    num_classes,
    feature_dim,
    scaler,
    client_stats,
):
    local_representations_a = []
    local_representations_c = []
    local_representations_d = []
    global_state = global_model.state_dict()
    for i, loader in enumerate(client_loaders):
        local_model = client_models[i]
        local_model.load_state_dict(global_state)
        _, telem = local_train(local_model, loader, scaler=scaler)
        algo_attr = extract_algo_attributes(dict(local_model.named_parameters()))
        _, phi_a = repr_gen.generate(
            dict(local_model.named_parameters()), ResourceType.ALGO, algo_attr
        )
        local_representations_a.append(phi_a)
        comp_flops = telem["throughput"] / 10.0
        comp_mem = telem["peak_mem_mb"] / 100.0
        comp_bw = 1000.0 / telem["train_time"]
        comp_vec = torch.tensor([comp_flops, comp_mem, comp_bw], dtype=torch.float64, device=DEVICE)
        comp_attr = extract_comp_attributes(comp_flops, comp_mem, comp_bw)
        _, phi_c = repr_gen.generate(comp_vec, ResourceType.COMP, comp_attr)
        local_representations_c.append(phi_c)
        stats = client_stats[i]
        data_bytes = (
            f"{stats['num_samples']}_{stats['entropy']:.4f}_"
            f"{stats['pixel_mean']:.4f}_{stats['pixel_std']:.4f}"
        ).encode() * 32
        data_attr = extract_data_attributes(stats["num_samples"], feature_dim, num_classes)
        _, phi_d = repr_gen.generate(data_bytes, ResourceType.DATA, data_attr)
        local_representations_d.append(phi_d)
    enhancement_results = adaptive_security_enhancement(
        local_representations_a, local_representations_c, local_representations_d
    )
    enhanced_a = enhancement_results["enhanced_a"]
    weights = [1.0 / len(client_loaders)] * len(client_loaders)
    folding_protocol.state = folding_protocol.state.__class__()
    folding_protocol.participant_commitments = {}
    for i in range(len(client_loaders)):
        folding_protocol.publish_commitment(i, enhanced_a[i])
    for i in range(len(client_loaders)):
        folding_protocol.fold_participant(i, weights[i], enhanced_a[i])
    verified = folding_protocol.verify_aggregation()
    avg_state = {}
    for key in global_state:
        avg_state[key] = (
            torch.stack([m.state_dict()[key].float() for m in client_models])
            .mean(dim=0)
            .to(global_state[key].dtype)
        )
    global_model.load_state_dict(avg_state)
    return enhancement_results, verified


def _try_compile(model, cfg):
    try:
        cm = torch.compile(model)
        x = torch.randn(2, cfg["channels"], cfg["image_size"], cfg["image_size"], device=DEVICE)
        if USE_CHANNELS_LAST:
            x = x.to(memory_format=torch.channels_last)
        out = cm(x)
        out.sum().backward()
        cm.zero_grad(set_to_none=True)
        return cm
    except Exception as e:
        print(f"  [torch.compile unavailable, using eager mode: {type(e).__name__}]")
        return None


def main(dataset_name="cifar10"):
    cfg = DATASETS[dataset_name]
    num_rounds = cfg["rounds"]
    feature_dim = cfg["channels"] * cfg["image_size"] * cfg["image_size"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"=== Dataset: {cfg['label']} | Device: {DEVICE} | rounds={num_rounds} ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | AMP={USE_AMP}")
    trainset, test_loader = get_data_loaders(dataset_name)
    client_indices = dirichlet_split(trainset, NUM_PARTICIPANTS, cfg["num_classes"])
    client_loaders = []
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "drop_last": True,
        "pin_memory": True,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    for indices in client_indices:
        subset = torch.utils.data.Subset(trainset, indices)
        loader = torch.utils.data.DataLoader(subset, **loader_kwargs)
        client_loaders.append(loader)
    print(f"Data split: {[len(idx) for idx in client_indices]}")
    client_stats = [compute_client_data_stats(ld, cfg["num_classes"]) for ld in client_loaders]

    def make_model():
        m = SimpleCNN(cfg["channels"], cfg["num_classes"], cfg["image_size"]).to(DEVICE)
        if USE_CHANNELS_LAST and DEVICE.type == "cuda":
            m = m.to(memory_format=torch.channels_last)
        if USE_COMPILE and DEVICE.type == "cuda":
            cm = _try_compile(m, cfg)
            if cm is not None:
                return cm
        return m

    global_model = make_model()
    client_models = [make_model() for _ in range(NUM_PARTICIPANTS)]
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))
    params = LatticeParams()
    repr_gen = TrustedRepresentationGenerator(params)
    folding_protocol = CollaborativeFoldingProtocol(params)
    experiment_log = {
        "dataset": dataset_name,
        "label": cfg["label"],
        "accuracy": [],
        "mi_history": [],
        "shapley_history": [],
        "sigma_history": [],
        "epsilon_history": [],
        "verification": [],
        "round_time": [],
    }
    for round_idx in tqdm(range(num_rounds), desc=f"{cfg['label']} Rounds"):
        t_start = time.time()
        enhancement_results, verified = federated_round(
            global_model,
            client_models,
            client_loaders,
            repr_gen,
            folding_protocol,
            round_idx,
            num_rounds,
            cfg["num_classes"],
            feature_dim,
            scaler,
            client_stats,
        )
        acc = evaluate(global_model, test_loader)
        t_elapsed = time.time() - t_start
        experiment_log["accuracy"].append(acc)
        experiment_log["mi_history"].append(enhancement_results["mi_history"])
        experiment_log["shapley_history"].append(enhancement_results["shapley_history"])
        experiment_log["sigma_history"].append(enhancement_results["sigma_history"])
        experiment_log["epsilon_history"].append(enhancement_results["epsilon_history"])
        experiment_log["verification"].append(verified)
        experiment_log["round_time"].append(t_elapsed)
        print(
            f"Round {round_idx + 1}/{num_rounds} | Acc: {acc:.4f} | "
            f"Verified: {verified} | Time: {t_elapsed:.1f}s"
        )
    out_dir = os.path.join("results", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    serializable_log = {}
    for k, v in experiment_log.items():
        if isinstance(v, list):
            serializable_log[k] = []
            for item in v:
                if isinstance(item, list):
                    serializable_log[k].append(
                        [x if not isinstance(x, tuple) else list(x) for x in item]
                    )
                else:
                    serializable_log[k].append(item)
        else:
            serializable_log[k] = v
    out_path = os.path.join(out_dir, "experiment_log.json")
    with open(out_path, "w") as f:
        json.dump(serializable_log, f, indent=2)
    if dataset_name == "cifar10":
        with open("results/experiment_log.json", "w") as f:
            json.dump(serializable_log, f, indent=2)
    print(f"\nFinal accuracy: {experiment_log['accuracy'][-1]:.4f}")
    print(f"Results saved to {out_path}")
    return experiment_log


if __name__ == "__main__":
    import sys

    ds = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    main(ds)
