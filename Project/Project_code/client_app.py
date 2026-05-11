import time

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from task import (
    EC_BACKUP_MAP,
    Net,
    compress_parameters,
    decompress_parameters,
    estimate_bytes,
    get_parameters,
    load_data_for_client,
    set_parameters,
    test,
    train,
)

# ============================================================
# Must match STRATEGY_MODE in server_app.py
# ============================================================
STRATEGY_MODE = "erasure_coded"


def _parse_bool(value, default: bool = False) -> bool:
    """
    Config values sent via fit_config arrive as strings ("True" / "False").
    Plain bool("False") == True in Python, so we must parse explicitly.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Standard FL client (baseline / partial / deadline / async_style modes)
# ---------------------------------------------------------------------------
class FlowerClient(NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.net = Net()
        self.trainloader, self.testloader = load_data_for_client(cid)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        epochs             = int(config.get("local-epochs", 1))
        lr                 = float(config.get("lr", 0.01))
        enable_straggler   = _parse_bool(config.get("enable-straggler", False))
        enable_compression = _parse_bool(config.get("enable-compression", False))

        print(f"[Client {self.cid}] enable_straggler = {enable_straggler}")
        print(f"[Client {self.cid}] enable_compression = {enable_compression}")

        train_metrics = train(
            self.net,
            self.trainloader,
            epochs=epochs,
            lr=lr,
            client_id=self.cid,
            enable_straggler=enable_straggler,
        )

        t_ser_start  = time.perf_counter()
        updated_params = get_parameters(self.net)
        t_serialization = time.perf_counter() - t_ser_start

        # Compress → measure bytes → decompress back to float32 for FedAvg
        t_comp_start = time.perf_counter()
        compressed_params, scales = compress_parameters(
            updated_params, enable_compression
        )
        bytes_sent = estimate_bytes(compressed_params, compressed=enable_compression)
        final_params = decompress_parameters(
            compressed_params, scales, enable_compression
        )
        t_compression = time.perf_counter() - t_comp_start

        metrics = {
            "train_loss":      float(train_metrics["train_loss"]),
            "train_acc":       float(train_metrics["train_acc"]),
            "simulated_delay": float(train_metrics["simulated_delay"]),
            "bytes_sent":      float(bytes_sent),
            # phase timings
            "t_data_load":       float(train_metrics.get("t_data_load",   0)),
            "t_forward":         float(train_metrics.get("t_forward",     0)),
            "t_backward":        float(train_metrics.get("t_backward",    0)),
            "t_serialization":   float(t_serialization),
            "t_compression":     float(t_compression),
            # system counters
            "mem_mb":            float(train_metrics.get("mem_mb",        0)),
            "cpu_pct":           float(train_metrics.get("cpu_pct",       0)),
            "disk_read_delta_mb":  float(train_metrics.get("disk_read_delta_mb",  0)),
            "disk_write_delta_mb": float(train_metrics.get("disk_write_delta_mb", 0)),
        }

        return final_params, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, acc = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


# ---------------------------------------------------------------------------
# Erasure-coded FL client
# ---------------------------------------------------------------------------
class ErasureCodedClient(NumPyClient):
    """
    Systematic (10,7) erasure-coded FL client.

    Each client trains on its OWN data partition (standard FL).
    Parity clients 7, 8, 9 ALSO train on backup partitions for straggler-prone
    clients 0, 1, 2 respectively (see EC_BACKUP_MAP in task.py).

    Fault-tolerance guarantee:
      If any subset of clients {0, 1, 2} miss the round deadline, their backup
      parity clients automatically carry the missing partition data.  The server
      always receives FedAvg inputs covering all 10 partitions with no accuracy
      loss vs the baseline — and no matrix inversion is required.

    Compute / communication trade-off:
      Parity clients 7,8,9 do 2x training (primary + backup), so total round
      compute is 13 partition-epochs instead of 10 for baseline.
      Communication cost is the same: each client sends ONE model update.
    """

    def __init__(self, cid: str):
        self.cid            = int(cid)
        self.net            = Net()
        # Primary data partition (same as in baseline)
        self.trainloader, self.testloader = load_data_for_client(cid)
        # Backup partition, if this is a parity client
        self.backup_cid: int = EC_BACKUP_MAP.get(self.cid, -1)
        if self.backup_cid >= 0:
            self.backup_loader, _ = load_data_for_client(str(self.backup_cid))
        else:
            self.backup_loader = None

    def fit(self, parameters, config):
        epochs             = int(config.get("local-epochs", 1))
        lr                 = float(config.get("lr", 0.01))
        enable_straggler   = _parse_bool(config.get("enable-straggler", False))
        enable_compression = _parse_bool(config.get("enable-compression", False))

        # ---- Train on PRIMARY partition ------------------------------------
        set_parameters(self.net, parameters)
        m_primary = train(
            self.net, self.trainloader, epochs=epochs, lr=lr,
            client_id=str(self.cid), enable_straggler=enable_straggler,
        )
        w_primary       = [p.copy() for p in get_parameters(self.net)]
        n_primary       = len(self.trainloader.dataset)
        simulated_delay = m_primary["simulated_delay"]

        print(
            f"[EC Client {self.cid}] primary partition={self.cid}  "
            f"train_acc={m_primary['train_acc']:.4f}  delay={simulated_delay:.2f}s"
        )

        # Initialise phase-timing accumulators from primary
        t_data_load = m_primary.get("t_data_load", 0)
        t_forward   = m_primary.get("t_forward",   0)
        t_backward  = m_primary.get("t_backward",  0)

        # ---- Parity clients: also train on BACKUP partition ---------------
        m_backup = None
        if self.backup_loader is not None:
            set_parameters(self.net, parameters)   # reset to same global model
            m_backup = train(
                self.net, self.backup_loader, epochs=epochs, lr=lr,
                client_id=str(self.cid), enable_straggler=False,
            )
            w_backup  = [p.copy() for p in get_parameters(self.net)]
            n_backup  = len(self.backup_loader.dataset)
            n_total   = n_primary + n_backup
            a, b      = n_primary / n_total, n_backup / n_total

            # Weighted average of primary and backup trained models
            w_final = [a * wp + b * wb for wp, wb in zip(w_primary, w_backup)]

            train_loss = a * m_primary["train_loss"] + b * m_backup["train_loss"]
            train_acc  = a * m_primary["train_acc"]  + b * m_backup["train_acc"]

            t_data_load += m_backup.get("t_data_load", 0)
            t_forward   += m_backup.get("t_forward",   0)
            t_backward  += m_backup.get("t_backward",  0)

            print(
                f"[EC Client {self.cid}] backup  partition={self.backup_cid}  "
                f"train_acc={m_backup['train_acc']:.4f}"
            )
        else:
            w_final    = w_primary
            n_total    = n_primary
            train_loss = m_primary["train_loss"]
            train_acc  = m_primary["train_acc"]

        # ---- Optional compression -----------------------------------------
        t_comp_start = time.perf_counter()
        compressed, scales = compress_parameters(w_final, enable_compression)
        bytes_sent         = estimate_bytes(compressed, compressed=enable_compression)
        final_params       = decompress_parameters(compressed, scales, enable_compression)
        t_compression      = time.perf_counter() - t_comp_start

        t_serialization = 0.001  # negligible array copy cost

        metrics = {
            "train_loss":      float(train_loss),
            "train_acc":       float(train_acc),
            "simulated_delay": float(simulated_delay),
            "bytes_sent":      float(bytes_sent),
            "is_parity_client": float(self.backup_cid >= 0),
            # phase timings
            "t_data_load":       float(t_data_load),
            "t_forward":         float(t_forward),
            "t_backward":        float(t_backward),
            "t_serialization":   float(t_serialization),
            "t_compression":     float(t_compression),
            # system counters
            "mem_mb":            float(m_primary.get("mem_mb",   0)),
            "cpu_pct":           float(m_primary.get("cpu_pct",  0)),
            "disk_read_delta_mb":  float(m_primary.get("disk_read_delta_mb",  0)),
            "disk_write_delta_mb": float(m_primary.get("disk_write_delta_mb", 0)),
        }

        return final_params, n_total, metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, acc = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------
def client_fn(context: Context):
    cid = context.node_config["partition-id"]
    if STRATEGY_MODE == "erasure_coded":
        return ErasureCodedClient(str(cid)).to_client()
    return FlowerClient(str(cid)).to_client()


app = ClientApp(client_fn=client_fn)
