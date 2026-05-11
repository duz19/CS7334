import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import psutil
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility — set before any random operation
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 28×28 → 14×14
        x = self.pool(F.relu(self.conv2(x)))   # 14×14 → 7×7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------
def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {
        k: torch.tensor(v, dtype=net.state_dict()[k].dtype)
        for k, v in params_dict
    }
    net.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------
def quantize_array(arr: np.ndarray, num_bits: int = 8) -> Tuple[np.ndarray, float]:
    """
    Linearly quantize arr to int8.
    Returns (quantized_int8_array, scale) so the caller can dequantize.
    """
    if arr.size == 0:
        return arr.astype(np.int8), 1.0
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        return np.zeros_like(arr, dtype=np.int8), 1.0
    qmax = float((2 ** (num_bits - 1)) - 1)          # 127 for 8-bit
    scaled = np.round(arr / max_abs * qmax).astype(np.int8)
    return scaled, max_abs


def dequantize_array(arr: np.ndarray, scale: float, num_bits: int = 8) -> np.ndarray:
    qmax = float((2 ** (num_bits - 1)) - 1)
    return arr.astype(np.float32) * (scale / qmax)


def compress_parameters(
    parameters: List[np.ndarray], enabled: bool
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Returns (arrays, scales).
    When enabled=True the arrays are int8; scales carry the per-tensor max_abs.
    When enabled=False the arrays are unchanged float32 and scales are all 1.0.
    """
    if not enabled:
        return parameters, [1.0] * len(parameters)

    arrays, scales = [], []
    for p in parameters:
        q, s = quantize_array(p)
        arrays.append(q)
        scales.append(s)
    return arrays, scales


def decompress_parameters(
    parameters: List[np.ndarray], scales: List[float], enabled: bool
) -> List[np.ndarray]:
    """Dequantize back to float32 so FedAvg aggregation works correctly."""
    if not enabled:
        return parameters
    return [dequantize_array(p, s) for p, s in zip(parameters, scales)]


def estimate_bytes(parameters: List[np.ndarray], compressed: bool) -> int:
    """
    Accurate byte estimate: int8 = 1 byte/param, float32 = 4 bytes/param.
    Only correct when compress_parameters() actually returns int8 arrays.
    """
    total = sum(p.size for p in parameters)
    return int(total * (1 if compressed else 4))


# ---------------------------------------------------------------------------
# Dataset / non-IID partitioning
# ---------------------------------------------------------------------------
def _load_raw_datasets():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.FashionMNIST(
        root="./data", train=True,  download=True, transform=transform
    )
    testset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return trainset, testset


def _create_noniid_partitions(
    trainset, num_clients: int = 100, shards_per_client: int = 2
) -> Dict[int, list]:
    labels         = np.array(trainset.targets)
    indices        = np.arange(len(trainset))
    sorted_indices = indices[np.argsort(labels)]

    num_shards = num_clients * shards_per_client
    shard_size = len(trainset) // num_shards
    shards = [
        sorted_indices[i * shard_size:(i + 1) * shard_size].tolist()
        for i in range(num_shards)
    ]

    rng = random.Random(SEED)   # local RNG — does not disturb global state
    rng.shuffle(shards)

    client_indices: Dict[int, list] = defaultdict(list)
    for cid in range(num_clients):
        for _ in range(shards_per_client):
            client_indices[cid].extend(shards.pop())
    return client_indices


# Lazy singletons — loaded once per worker process, not at import time.
# This avoids parallel download races in Ray simulation.
_TRAINSET         = None
_TESTSET          = None
_CLIENT_PARTITIONS = None


def _ensure_loaded() -> None:
    global _TRAINSET, _TESTSET, _CLIENT_PARTITIONS
    if _TRAINSET is None:
        _TRAINSET, _TESTSET = _load_raw_datasets()
        _CLIENT_PARTITIONS  = _create_noniid_partitions(_TRAINSET)


def get_testset():
    _ensure_loaded()
    return _TESTSET


def load_data_for_client(client_id, batch_size: int = 32):
    _ensure_loaded()
    cid          = int(client_id)
    train_subset = Subset(_TRAINSET, _CLIENT_PARTITIONS[cid])
    trainloader  = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader   = DataLoader(_TESTSET,     batch_size=128,        shuffle=False)
    return trainloader, testloader


# ---------------------------------------------------------------------------
# Train / Test
# ---------------------------------------------------------------------------
def _snapshot_sys() -> Dict[str, float]:
    """Capture per-process CPU and memory at a point in time."""
    proc = psutil.Process(os.getpid())
    mem  = proc.memory_info().rss / 1e6          # MB
    cpu  = proc.cpu_percent(interval=None)        # non-blocking snapshot
    disk = psutil.disk_io_counters()
    net  = psutil.net_io_counters()
    return {
        "mem_mb":       mem,
        "cpu_pct":      cpu,
        "disk_read_mb": disk.read_bytes  / 1e6 if disk else 0.0,
        "disk_write_mb":disk.write_bytes / 1e6 if disk else 0.0,
        "net_sent_mb":  net.bytes_sent   / 1e6 if net  else 0.0,
        "net_recv_mb":  net.bytes_recv   / 1e6 if net  else 0.0,
    }


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    client_id: str,
    enable_straggler: bool,
) -> Dict[str, float]:
    net.to(DEVICE)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    simulated_delay = 0.0
    cid = int(client_id)
    straggler_set = set(range(10))  # clients 0-9 are straggler-prone in 100-node setup
    if enable_straggler:
        simulated_delay = (
            random.uniform(2.0, 5.0) if cid in straggler_set
            else random.uniform(0.0, 0.6)
        )
        time.sleep(simulated_delay)

    total_loss, total, correct = 0.0, 0, 0
    t_data_load = 0.0
    t_forward   = 0.0
    t_backward  = 0.0

    snap_before = _snapshot_sys()
    t_train_start = time.perf_counter()

    for _ in range(epochs):
        for images, labels in trainloader:
            t0 = time.perf_counter()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            t_data_load += time.perf_counter() - t0

            t1 = time.perf_counter()
            optimizer.zero_grad()
            outputs = net(images)
            loss    = criterion(outputs, labels)
            t_forward += time.perf_counter() - t1

            t2 = time.perf_counter()
            loss.backward()
            optimizer.step()
            t_backward += time.perf_counter() - t2

            total_loss += loss.item() * labels.size(0)
            total      += labels.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()

    t_train_total = time.perf_counter() - t_train_start
    snap_after    = _snapshot_sys()

    return {
        "train_loss":      total_loss / total,
        "train_acc":       correct / total,
        "simulated_delay": simulated_delay,
        # per-phase timings (seconds)
        "t_data_load":     t_data_load,
        "t_forward":       t_forward,
        "t_backward":      t_backward,
        "t_train_total":   t_train_total,
        # system counters (deltas)
        "mem_mb":          snap_after["mem_mb"],
        "cpu_pct":         snap_after["cpu_pct"],
        "disk_read_delta_mb":  snap_after["disk_read_mb"]  - snap_before["disk_read_mb"],
        "disk_write_delta_mb": snap_after["disk_write_mb"] - snap_before["disk_write_mb"],
        "net_sent_delta_mb":   snap_after["net_sent_mb"]   - snap_before["net_sent_mb"],
        "net_recv_delta_mb":   snap_after["net_recv_mb"]   - snap_before["net_recv_mb"],
    }


def test(net: nn.Module, testloader: DataLoader) -> Tuple[float, float]:
    net.to(DEVICE)
    net.eval()
    criterion = nn.CrossEntropyLoss()

    loss, total, correct = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss    += criterion(outputs, labels).item() * labels.size(0)
            total   += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    return loss / total, correct / total


# ---------------------------------------------------------------------------
# Erasure Coding — Systematic (N, K) Redundancy Scheme
# ---------------------------------------------------------------------------
#
# Design: (10, 7) systematic erasure code for FL straggler tolerance.
#
#   N = 10 total clients, K = 7 minimum responses, S = 3 straggler tolerance.
#
#   Encoding:
#     - Clients 0–6 are "systematic" nodes: each trains on ONE unique
#       primary data partition and sends that update directly.
#     - Clients 7,8,9 are "parity" nodes: each trains on their primary
#       partition AND a designated backup for a straggler-prone client:
#           Client 7 → primary D_7  +  backup D_0  (covers slow client 0)
#           Client 8 → primary D_8  +  backup D_1  (covers slow client 1)
#           Client 9 → primary D_9  +  backup D_2  (covers slow client 2)
#       Parity clients return the weighted average of their two trained models.
#
#   Decoding (server-side, no matrix inversion):
#     - If all 10 respond: standard FedAvg over 10 responses.
#     - If any subset of {0,1,2} are stragglers: parity clients {7,8,9}
#       automatically carry backup data for the missing partitions, so the
#       aggregation still covers all data — no FedAvg accuracy loss.
#
#   Why "erasure code"?
#     The per-partition updates are the "symbols" to be protected.
#     The systematic structure (identity mapping for clients 0–6) plus
#     explicit redundancy (parity clients 7–9) forms a (10, 7) systematic
#     code tolerating up to 3 specific erasures in the straggler positions.
#
ERASURE_N = 100                         # total clients
ERASURE_K = 90                          # min responses needed
ERASURE_S = ERASURE_N - ERASURE_K      # straggler tolerance = 10

# Parity clients 90–99 each back up straggler-prone clients 0–9
EC_BACKUP_MAP: Dict[int, int] = {90 + i: i for i in range(10)}
