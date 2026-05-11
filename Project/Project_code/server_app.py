import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
from torch.utils.data import DataLoader

from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from task import (
    ERASURE_K,
    ERASURE_N,
    ERASURE_S,
    EC_BACKUP_MAP,
    Net,
    get_parameters,
    get_testset,
    set_parameters,
    test,
)

METRICS_FILE = "results/overhead_metrics.json"
_overhead_log: List[Dict] = []


def _save_overhead_log():
    os.makedirs("results", exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(_overhead_log, f, indent=2)


# ============================================================
# Choose strategy mode here
# ============================================================
STRATEGY_MODE = "erasure_coded"
# options:
# "baseline"
# "partial"
# "deadline"
# "async_style"
# "erasure_coded"   ← systematic (10,7) erasure code

# Shared experiment settings
NUM_ROUNDS          = 5
NUM_CLIENTS         = 100
LOCAL_EPOCHS        = 1
LR                  = 0.01
ENABLE_STRAGGLER    = "True"
ENABLE_COMPRESSION  = "False"

# Strategy-specific settings
PARTIAL_K        = 50         # for partial participation (50% of 100)
DEADLINE_SEC     = 15.0       # for deadline-based and erasure_coded (wider for 100 nodes)
ASYNC_WINDOW_SEC = 3.0        # for async-style approximation


# ---------------------------------------------------------------------------
# Base strategy: FedAvg + per-round logging + centralized evaluation
# ---------------------------------------------------------------------------
class FedAvgWithCentralEval(FedAvg):
    """
    FedAvg + per-round logging + centralized evaluation.
    Supports baseline / partial / deadline / async_style / erasure_coded modes.
    """

    def __init__(
        self,
        strategy_mode: str = "baseline",
        fastest_k: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strategy_mode = strategy_mode
        self.fastest_k = fastest_k

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        usable_results = results

        # Async-style: keep only the fastest-k results before aggregation
        if self.strategy_mode == "async_style" and self.fastest_k is not None:
            usable_results = sorted(
                results,
                key=lambda x: x[1].metrics.get("simulated_delay", 0.0),
            )[: self.fastest_k]

        t_agg_start = time.perf_counter()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, usable_results, failures
        )
        t_aggregation = time.perf_counter() - t_agg_start

        round_record: Dict = {
            "round": server_round,
            "mode":  self.strategy_mode,
            "n_clients": len(usable_results),
            "n_failures": len(failures),
            "t_aggregation_s": t_aggregation,
        }

        if usable_results:
            all_bytes   = [r.metrics.get("bytes_sent", 0)           for _, r in usable_results]
            all_delays  = [r.metrics.get("simulated_delay", 0)      for _, r in usable_results]
            all_loss    = [r.metrics.get("train_loss", 0)           for _, r in usable_results]
            all_acc     = [r.metrics.get("train_acc", 0)            for _, r in usable_results]
            n_parity    = sum(int(r.metrics.get("is_parity_client", 0)) for _, r in usable_results)
            # per-phase timings from clients (averages)
            t_data      = [r.metrics.get("t_data_load",  0) for _, r in usable_results]
            t_fwd       = [r.metrics.get("t_forward",    0) for _, r in usable_results]
            t_bwd       = [r.metrics.get("t_backward",   0) for _, r in usable_results]
            t_serial    = [r.metrics.get("t_serialization", 0) for _, r in usable_results]
            t_compress  = [r.metrics.get("t_compression",   0) for _, r in usable_results]
            mem_list    = [r.metrics.get("mem_mb",           0) for _, r in usable_results]
            cpu_list    = [r.metrics.get("cpu_pct",          0) for _, r in usable_results]
            disk_r      = [r.metrics.get("disk_read_delta_mb",  0) for _, r in usable_results]
            disk_w      = [r.metrics.get("disk_write_delta_mb", 0) for _, r in usable_results]

            round_record.update({
                "avg_train_loss":    float(np.mean(all_loss)),
                "avg_train_acc":     float(np.mean(all_acc)),
                "avg_delay_s":       float(np.mean(all_delays)),
                "total_bytes_mb":    float(sum(all_bytes) / 1e6),
                "n_parity_used":     n_parity,
                # overhead breakdown (avg across clients)
                "avg_t_data_load_s":    float(np.mean(t_data)),
                "avg_t_forward_s":      float(np.mean(t_fwd)),
                "avg_t_backward_s":     float(np.mean(t_bwd)),
                "avg_t_serialization_s":float(np.mean(t_serial)),
                "avg_t_compression_s":  float(np.mean(t_compress)),
                "avg_mem_mb":           float(np.mean(mem_list)),
                "avg_cpu_pct":          float(np.mean(cpu_list)),
                "sum_disk_read_mb":     float(sum(disk_r)),
                "sum_disk_write_mb":    float(sum(disk_w)),
            })

            if self.strategy_mode == "erasure_coded":
                print(
                    f"\n[Round {server_round}] mode=erasure_coded\n"
                    f"  responding clients : {len(usable_results)} / {NUM_CLIENTS}\n"
                    f"  parity clients used: {n_parity}  "
                    f"(straggler tolerance: {ERASURE_S})\n"
                    f"  dropped / failures : {len(results) - len(usable_results) + len(failures)}\n"
                    f"  avg train_loss     : {np.mean(all_loss):.4f}\n"
                    f"  avg train_acc      : {np.mean(all_acc):.4f}\n"
                    f"  avg delay (s)      : {np.mean(all_delays):.2f}\n"
                    f"  total bytes sent   : {sum(all_bytes) / 1e6:.2f} MB\n"
                    f"  avg compute fwd/bwd: {np.mean(t_fwd):.3f}s / {np.mean(t_bwd):.3f}s\n"
                    f"  avg data load      : {np.mean(t_data):.3f}s\n"
                    f"  aggregation time   : {t_aggregation:.3f}s"
                )
            else:
                print(
                    f"\n[Round {server_round}] mode={self.strategy_mode}\n"
                    f"  used results     : {len(usable_results)}\n"
                    f"  dropped/failures : {len(results) - len(usable_results) + len(failures)}\n"
                    f"  avg train_loss   : {np.mean(all_loss):.4f}\n"
                    f"  avg train_acc    : {np.mean(all_acc):.4f}\n"
                    f"  avg delay (s)    : {np.mean(all_delays):.2f}\n"
                    f"  total bytes sent : {sum(all_bytes) / 1e6:.2f} MB\n"
                    f"  avg compute fwd/bwd: {np.mean(t_fwd):.3f}s / {np.mean(t_bwd):.3f}s\n"
                    f"  aggregation time : {t_aggregation:.3f}s"
                )
        else:
            print(f"\n[Round {server_round}] mode={self.strategy_mode} -> no usable client results")

        _overhead_log.append(round_record)
        _save_overhead_log()

        return aggregated_parameters, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = Net()
        ndarrays: NDArrays = parameters_to_ndarrays(parameters)
        set_parameters(net, ndarrays)

        testloader = DataLoader(get_testset(), batch_size=128, shuffle=False)

        t_eval_start = time.perf_counter()
        loss, acc = test(net, testloader)
        t_eval = time.perf_counter() - t_eval_start

        # Patch evaluation time into matching round record
        for rec in _overhead_log:
            if rec["round"] == server_round:
                rec["t_evaluation_s"] = t_eval
                break
        _save_overhead_log()

        proc = psutil.Process(os.getpid())
        print(
            f"[Round {server_round}] Centralized — "
            f"loss: {loss:.4f}  accuracy: {acc:.4f}  "
            f"eval_time: {t_eval:.2f}s  "
            f"server_mem: {proc.memory_info().rss / 1e6:.1f} MB"
        )
        return loss, {"centralized_accuracy": float(acc)}


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------
def server_fn(context):
    # --------------------------------------------------------
    # Configure strategy-specific behaviour
    # --------------------------------------------------------
    if STRATEGY_MODE == "baseline":
        fraction_fit    = 1.0
        min_fit_clients = NUM_CLIENTS
        round_timeout   = None
        fastest_k       = None

    elif STRATEGY_MODE == "partial":
        fraction_fit    = PARTIAL_K / NUM_CLIENTS
        min_fit_clients = PARTIAL_K
        round_timeout   = None
        fastest_k       = None

    elif STRATEGY_MODE == "deadline":
        fraction_fit    = 1.0
        min_fit_clients = 1
        round_timeout   = DEADLINE_SEC
        fastest_k       = None

    elif STRATEGY_MODE == "async_style":
        fraction_fit    = 1.0
        min_fit_clients = 1
        round_timeout   = ASYNC_WINDOW_SEC
        fastest_k       = PARTIAL_K

    elif STRATEGY_MODE == "erasure_coded":
        # Broadcast to all N clients.  Decode (via FedAvg) as soon as K respond.
        # Parity clients 7,8,9 carry backup data for stragglers 0,1,2,
        # so if the 3 straggler-prone clients miss the deadline the aggregation
        # still covers all 10 data partitions.
        fraction_fit    = 1.0
        min_fit_clients = ERASURE_K   # proceed once K clients respond
        round_timeout   = DEADLINE_SEC
        fastest_k       = None

    else:
        raise ValueError(f"Unknown STRATEGY_MODE: {STRATEGY_MODE}")

    print(
        f"\n[Server] mode={STRATEGY_MODE}, "
        f"round_timeout={round_timeout}, "
        f"fraction_fit={fraction_fit}, "
        f"min_fit_clients={min_fit_clients}, "
        f"straggler={ENABLE_STRAGGLER}, "
        f"compression={ENABLE_COMPRESSION}"
    )

    global_model       = Net()
    initial_parameters = ndarrays_to_parameters(get_parameters(global_model))

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        return {
            "local-epochs":       LOCAL_EPOCHS,
            "lr":                 LR,
            "enable-straggler":   ENABLE_STRAGGLER,
            "enable-compression": ENABLE_COMPRESSION,
        }

    strategy = FedAvgWithCentralEval(
        strategy_mode       = STRATEGY_MODE,
        fastest_k           = fastest_k,
        fraction_fit        = fraction_fit,
        fraction_evaluate   = 0.0,
        min_fit_clients     = min_fit_clients,
        min_available_clients = NUM_CLIENTS,
        initial_parameters  = initial_parameters,
        on_fit_config_fn    = fit_config,
        accept_failures     = True,
    )

    config = ServerConfig(
        num_rounds    = NUM_ROUNDS,
        round_timeout = round_timeout,
    )
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
