from typing import Dict, List, Optional, Tuple, Union

import numpy as np
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

from task import Net, get_parameters, get_testset, set_parameters, test


# ---------------------------------------------------------------------------
# Custom strategy: FedAvg + server-side (centralized) evaluation each round
# ---------------------------------------------------------------------------
class FedAvgWithCentralEval(FedAvg):
    """
    Extends FedAvg with:
      - per-round logging of client train metrics (loss, acc, delay, bytes)
      - centralized evaluation on the global test set after aggregation
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if results:
            all_bytes  = [r.metrics.get("bytes_sent", 0)       for _, r in results]
            all_delays = [r.metrics.get("simulated_delay", 0)  for _, r in results]
            all_loss   = [r.metrics.get("train_loss", 0)       for _, r in results]
            all_acc    = [r.metrics.get("train_acc", 0)        for _, r in results]
            print(
                f"\n[Round {server_round}] fit results: {len(results)} ok, "
                f"{len(failures)} failed\n"
                f"  avg train_loss   : {np.mean(all_loss):.4f}\n"
                f"  avg train_acc    : {np.mean(all_acc):.4f}\n"
                f"  avg delay (s)    : {np.mean(all_delays):.2f}\n"
                f"  total bytes sent : {sum(all_bytes) / 1e6:.2f} MB"
            )

        return aggregated_parameters, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation on the global held-out test set."""
        net = Net()
        ndarrays: NDArrays = parameters_to_ndarrays(parameters)
        set_parameters(net, ndarrays)

        testloader = DataLoader(get_testset(), batch_size=128, shuffle=False)
        loss, acc = test(net, testloader)

        print(
            f"[Round {server_round}] Centralized — "
            f"loss: {loss:.4f}  accuracy: {acc:.4f}"
        )
        return loss, {"centralized_accuracy": float(acc)}


# ---------------------------------------------------------------------------
# Server factory — called by `flwr run`
# ---------------------------------------------------------------------------
def server_fn(context):
    run_cfg = context.run_config

    num_rounds       = 5
    fraction_fit     = 1.0
    min_clients      = 10
    local_epochs     = 1
    lr               = 0.01
    enable_straggler = "True"
    enable_compression = "False"

    # Also verify this exact functi
    # Initial global model — must use ndarrays_to_parameters so that
    # the legacy NumPyClient compat layer can deserialise it correctly.
    global_model = Net()
    initial_parameters = ndarrays_to_parameters(get_parameters(global_model))

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Sends hyperparams + flags to every selected client each round."""
        return {
            "local-epochs":       local_epochs,
            "lr":                 lr,
            "enable-straggler":   enable_straggler,
            "enable-compression": enable_compression,
        }

    strategy = FedAvgWithCentralEval(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,           # no federated (client-side) evaluation
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
