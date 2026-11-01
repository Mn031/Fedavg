"""
main.py - Federated Learning Simulation and Tab Attack Privacy Evaluation.

Orchestrates:
  1. Data loading (default dummy dataset or custom user dataset via CLI).
  2. Federated training using Flower (flwr.simulation.start_simulation) and FedAvg.
  3. Local PyTorch model training via a custom NumPyClient.
  4. Global model extraction and Tab Attack leakage evaluation (arXiv:2101.05405v2).
  5. S_uniq privacy violation audit reporting.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.client import NumPyClient
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from fedavg.model import NextWordModel
from fedavg.dataset import (
    PAD_IDX,
    Vocabulary,
    get_default_dummy_data,
    load_custom_data,
    prepare_federated_dataloaders,
    save_dummy_dataset_to_json,
)
from fedavg.attack import (
    tab_attack_leakage_analysis,
    get_unique_leaked_sequences,
    print_attack_report,
)


# =====================================================================
# PYTORCH LOCAL TRAINING & EVALUATION
# =====================================================================

def train_one_client(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    vocab_size: int,
    epochs: int = 5,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
) -> float:
    """Trains the NextWordModel on local client data for a specified number of epochs."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = 0.0
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

    return last_loss


def evaluate_one_client(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    vocab_size: int,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Evaluates cross-entropy loss and next-word top-1 accuracy."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _ = model(batch_x)
            loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
            total_loss += loss.item() * batch_x.size(0)

            preds = logits.argmax(dim=-1)
            mask = batch_y != PAD_IDX
            total_correct += ((preds == batch_y) & mask).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = total_correct / max(total_tokens, 1)
    return avg_loss, acc


# =====================================================================
# FLOWER NUMPYCLIENT
# =====================================================================

class FederatedLanguageClient(NumPyClient):
    """
    Flower NumPyClient wrapping local PyTorch NextWordModel training.
    """
    def __init__(
        self,
        client_id: str,
        vocab: Vocabulary,
        loader: torch.utils.data.DataLoader,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        local_epochs: int = 5,
        lr: float = 0.01,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.vocab = vocab
        self.loader = loader
        self.device = torch.device(device)
        self.local_epochs = local_epochs
        self.lr = lr

        self.model = NextWordModel(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.model.set_weights(parameters)
        loss = train_one_client(
            model=self.model,
            loader=self.loader,
            vocab_size=len(self.vocab),
            epochs=self.local_epochs,
            lr=self.lr,
            device=self.device,
        )
        num_examples = len(self.loader.dataset)
        return self.model.get_weights(), num_examples, {"loss": float(loss)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.model.set_weights(parameters)
        loss, acc = evaluate_one_client(
            model=self.model,
            loader=self.loader,
            vocab_size=len(self.vocab),
            device=self.device,
        )
        num_examples = len(self.loader.dataset)
        return float(loss), num_examples, {"loss": float(loss), "accuracy": float(acc)}


# =====================================================================
# CUSTOM FEDAVG STRATEGY (CAPTURES FINAL GLOBAL MODEL WEIGHTS)
# =====================================================================

class AuditedFedAvg(FedAvg):
    """
    FedAvg strategy that captures the aggregated global weights each round,
    allowing downstream privacy audit and vulnerability analysis.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_global_parameters: Optional[Parameters] = None

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_global_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics


# =====================================================================
# CLI ARGUMENT PARSER
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Learning Privacy Evaluation: Tab Attack & S_uniq Leakage (arXiv:2101.05405v2)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom JSON or TXT dataset file. If not provided, runs default dummy dataset.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="Number of federated learning rounds (default: 8, typically 5-10).",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=None,
        help="Number of clients to simulate. Defaults to the number of clients present in dataset.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=5,
        help="Local training epochs per client per round (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for local client training (default: 0.01).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for client DataLoaders (default: 2).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Token embedding vector dimension (default: 64).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="GRU hidden state dimension (default: 128).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Top-k prediction threshold for Tab Attack (default: 1, simulates single tab autocompletion).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=2,
        help="Minimum consecutive token length for memorized fragment collection (default: 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device ('cpu' or 'cuda', default: 'cpu').",
    )
    parser.add_argument(
        "--export-sample-data",
        action="store_true",
        help="Export the default dummy dataset to 'custom_dataset_sample.json' for reference.",
    )
    return parser.parse_args()


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    args = parse_args()

    # If requested, export sample dataset template
    if args.export_sample_data:
        saved_file = save_dummy_dataset_to_json("custom_dataset_sample.json")
        print(f"[OK] Saved custom dataset template to: {os.path.abspath(saved_file)}")
        if args.dataset is None:
            return

    # 1. Load Data
    print("=" * 78)
    print(" 1. DATA PIPELINE INITIALIZATION")
    print("=" * 78)
    if args.dataset:
        print(f"Loading custom dataset from: {args.dataset}")
        client_data = load_custom_data(args.dataset)
    else:
        print("No --dataset provided. Loading default multi-client dummy dataset...")
        client_data = get_default_dummy_data()
        # Save a sample JSON file for easy user exploration
        if not os.path.exists("custom_dataset_sample.json"):
            save_dummy_dataset_to_json("custom_dataset_sample.json")

    client_ids = list(client_data.keys())
    num_clients = args.clients if args.clients is not None else len(client_ids)
    
    # If user asked for more clients than defined in data, cycle through available partitions
    if num_clients > len(client_ids):
        expanded_data: Dict[str, List[str]] = {}
        for i in range(num_clients):
            cid = f"client_{i}"
            source_cid = client_ids[i % len(client_ids)]
            expanded_data[cid] = client_data[source_cid]
        client_data = expanded_data
        client_ids = list(client_data.keys())

    print(f"Total Federated Clients : {len(client_ids)}")
    for cid in client_ids:
        print(f"  - {cid}: {len(client_data[cid])} sentence(s)")

    # Prepare DataLoaders and Shared Vocabulary
    client_loaders, vocab = prepare_federated_dataloaders(
        client_data=client_data,
        batch_size=args.batch_size,
        shuffle=True,
    )
    print(f"Total Unique Vocabulary : {len(vocab)} tokens")

    # 2. Setup Federated Learning Simulation
    print("\n" + "=" * 78)
    print(f" 2. FLOWER FEDERATED SIMULATION ({args.rounds} ROUNDS, FedAvg)")
    print("=" * 78)

    # Initial model for parameter initialization
    init_model = NextWordModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    initial_parameters = ndarrays_to_parameters(init_model.get_weights())

    strategy = AuditedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(client_ids),
        min_evaluate_clients=len(client_ids),
        min_available_clients=len(client_ids),
        initial_parameters=initial_parameters,
    )

    def client_fn(context_or_cid):
        """Creates a virtual Flower client based on partition index or context."""
        if hasattr(context_or_cid, "node_config"):
            partition_id = context_or_cid.node_config.get("partition-id", "0")
            idx = int(partition_id)
        elif isinstance(context_or_cid, str):
            try:
                idx = int(context_or_cid)
            except ValueError:
                idx = 0
        else:
            idx = 0

        cid = client_ids[idx % len(client_ids)]
        loader = client_loaders[cid]

        client = FederatedLanguageClient(
            client_id=cid,
            vocab=vocab,
            loader=loader,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            local_epochs=args.local_epochs,
            lr=args.lr,
            device=args.device,
        )
        return client.to_client()

    # Run Flower simulation
    sim_history = start_simulation(
        client_fn=client_fn,
        num_clients=len(client_ids),
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    print("\n[OK] Federated Learning simulation completed successfully.")

    # 3. Extract Trained Global Model
    print("\n" + "=" * 78)
    print(" 3. EXTRACTING GLOBAL FEDERATED MODEL")
    print("=" * 78)

    if strategy.latest_global_parameters is None:
        print("[!] Error: No global parameters were aggregated.")
        sys.exit(1)

    final_ndarrays = parameters_to_ndarrays(strategy.latest_global_parameters)
    global_model = NextWordModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    global_model.set_weights(final_ndarrays)
    global_model.eval()
    print("[OK] Loaded aggregated global model weights into PyTorch NextWordModel.")

    # 4. Tab Attack Privacy Leakage Analysis (Algorithm 1)
    print("\n" + "=" * 78)
    print(f" 4. EXECUTING TAB ATTACK AUDIT (top-k={args.k}, min_len={args.min_len})")
    print("=" * 78)
    print("Querying the global model token-by-token to identify consecutive memorized streaks...")

    leaked_by_client = tab_attack_leakage_analysis(
        model=global_model,
        dataset=client_data,
        vocab=vocab,
        k=args.k,
        min_len=args.min_len,
        device=torch.device(args.device),
    )

    leakage_metrics = get_unique_leaked_sequences(
        leaked_sequences_by_client=leaked_by_client,
        training_data=client_data,
    )

    # 5. Output Audit Report
    print_attack_report(leakage_metrics)


if __name__ == "__main__":
    main()
