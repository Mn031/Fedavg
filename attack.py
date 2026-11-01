"""
attack.py - Implementation of the "Tab Attack" and S_uniq Leakage Analysis
Reference: "Training Data Leakage Analysis in Language Models" (arXiv:2101.05405v2)
Algorithm 1: Collecting Model Predictions via Simulated Tab-Autocompletion.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
from fedavg.dataset import Vocabulary, tokenize, EOS_IDX, PAD_IDX, UNK_IDX


def tab_attack_leakage_analysis(
    model: nn.Module,
    dataset: Union[Dict[str, List[str]], Dict[str, List[List[int]]]],
    vocab: Optional[Vocabulary] = None,
    k: int = 1,
    min_len: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, List[Dict[str, Union[str, List[str], int]]]]:
    """
    Implements Algorithm 1 from arXiv:2101.05405v2: "Tab Attack" Leakage Analysis.
    
    Simulates a user querying the language model in a black-box auto-completion setting
    (e.g., pressing the TAB key). Evaluates token-by-token whether the model's top-k
    predictions correctly anticipate the next training tokens consecutively.
    
    Args:
        model: Trained PyTorch NextWordModel.
        dataset: Either:
                 - Dict[str, List[str]]: Mapping client_id -> list of raw text sentences.
                 - Dict[str, List[List[int]]]: Mapping client_id -> list of token ID lists.
        vocab: Vocabulary object (required if dataset contains raw text strings).
        k: Top-k prediction threshold (k=1 simulates single tab acceptance).
        min_len: Minimum consecutive token length to be classified as a memorized fragment.
        device: Torch device (CPU or CUDA).
        
    Returns:
        Dict[str, List[Dict]]: Mapping client_id -> list of leaked sequence records:
            {
                "tokens": List[str],
                "text": str,
                "length": int,
                "client_id": str,
            }
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()

    leaked_sequences_by_client: Dict[str, List[Dict[str, Union[str, List[str], int]]]] = {}

    with torch.no_grad():
        for client_id, sequences in dataset.items():
            client_leaks: List[Dict[str, Union[str, List[str], int]]] = []

            for seq in sequences:
                # Handle raw text string vs token IDs
                if isinstance(seq, str):
                    if vocab is None:
                        raise ValueError("vocab must be provided when dataset contains raw string sentences.")
                    raw_tokens = tokenize(seq)
                    token_ids = vocab.encode(raw_tokens, add_eos=False)
                elif isinstance(seq, list) and (len(seq) == 0 or isinstance(seq[0], int)):
                    token_ids = [t for t in seq if t not in {PAD_IDX, EOS_IDX}]
                    raw_tokens = vocab.decode(token_ids) if vocab else [str(t) for t in token_ids]
                elif isinstance(seq, list) and isinstance(seq[0], str):
                    raw_tokens = seq
                    token_ids = vocab.encode(raw_tokens, add_eos=False) if vocab else []
                else:
                    continue

                if len(token_ids) < 2:
                    continue

                # Algorithm 1: Token-by-token consecutive prediction tracking
                # Current running sequence of consecutive correct predictions
                current_consecutive_ids: List[int] = []

                for t in range(len(token_ids) - 1):
                    # Prefix provided to the model: w_1, ..., w_t
                    prefix_ids = token_ids[: t + 1]
                    input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

                    # Forward pass
                    logits, _ = model(input_tensor)
                    # Next token logits come from the last position of the output
                    last_token_logits = logits[0, -1, :]

                    # Retrieve top-k predicted token IDs
                    top_k_preds = torch.topk(last_token_logits, k=min(k, last_token_logits.size(-1))).indices.tolist()

                    target_next_token = token_ids[t + 1]

                    if target_next_token in top_k_preds:
                        # Successful Tab hit! Extend sequence
                        if len(current_consecutive_ids) == 0:
                            # Start sequence with trigger token w_t and predicted next token w_{t+1}
                            current_consecutive_ids = [token_ids[t], target_next_token]
                        else:
                            current_consecutive_ids.append(target_next_token)
                    else:
                        # Miss: consecutive streak broken
                        if len(current_consecutive_ids) >= min_len:
                            matched_tokens = (
                                vocab.decode(current_consecutive_ids) if vocab else [str(i) for i in current_consecutive_ids]
                            )
                            text_rep = vocab.tokens_to_text(matched_tokens) if vocab else " ".join(matched_tokens)
                            client_leaks.append({
                                "tokens": matched_tokens,
                                "token_ids": current_consecutive_ids,
                                "text": text_rep,
                                "length": len(current_consecutive_ids),
                                "client_id": client_id,
                            })
                        current_consecutive_ids = []

                # End of sentence check
                if len(current_consecutive_ids) >= min_len:
                    matched_tokens = (
                        vocab.decode(current_consecutive_ids) if vocab else [str(i) for i in current_consecutive_ids]
                    )
                    text_rep = vocab.tokens_to_text(matched_tokens) if vocab else " ".join(matched_tokens)
                    client_leaks.append({
                        "tokens": matched_tokens,
                        "token_ids": current_consecutive_ids,
                        "text": text_rep,
                        "length": len(current_consecutive_ids),
                        "client_id": client_id,
                    })

            leaked_sequences_by_client[client_id] = client_leaks

    return leaked_sequences_by_client


def get_unique_leaked_sequences(
    leaked_sequences_by_client: Dict[str, List[Dict[str, Union[str, List[str], int]]]],
    training_data: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Union[List[Dict], int]]:
    """
    Calculates the S_uniq metric from arXiv:2101.05405v2.
    
    Filters leaked sequences to isolate those where the user count in the training data
    is exactly one (user_count == 1). These represent severe privacy breaches because
    they originate exclusively from a single user's confidential contributions rather
    than shared corpus statistics.
    
    Args:
        leaked_sequences_by_client: Output of `tab_attack_leakage_analysis`.
        training_data: Optional Dict[str, List[str]] mapping client_id -> list of raw sentences.
                       If provided, accurately computes ground-truth client occurrence count
                       across the full corpus for each sequence.
                       
    Returns:
        Dict with keys:
          - 's_uniq': List of unique leaked sequence dictionaries (user_count == 1).
          - 's_mult': List of multi-user sequence dictionaries (user_count > 1).
          - 'total_s_uniq_count': Total number of unique leaked sequences.
          - 'total_s_mult_count': Total number of multi-user sequences.
    """
    # Collect all unique candidate sequences
    seen_sequences: Dict[str, Dict] = {}

    for client_id, leaks in leaked_sequences_by_client.items():
        for leak in leaks:
            seq_text = leak["text"].strip().lower()
            if not seq_text:
                continue

            if seq_text not in seen_sequences:
                seen_sequences[seq_text] = {
                    "text": leak["text"],
                    "tokens": leak["tokens"],
                    "length": leak["length"],
                    "leaking_clients": set(),
                    "ground_truth_clients": set(),
                }
            seen_sequences[seq_text]["leaking_clients"].add(client_id)

    # Compute ground truth client frequency across the dataset
    for seq_text, info in seen_sequences.items():
        if training_data is not None:
            # Check how many clients actually contain this exact sequence or token sequence in their data
            for cid, sentences in training_data.items():
                for s in sentences:
                    # Normalized substring or token match
                    if seq_text in s.lower() or all(tok in tokenize(s) for tok in info["tokens"][:3]):
                        info["ground_truth_clients"].add(cid)
                        break
        else:
            # Fallback to leaking clients count
            info["ground_truth_clients"] = info["leaking_clients"].copy()

    s_uniq: List[Dict] = []
    s_mult: List[Dict] = []

    for seq_text, info in seen_sequences.items():
        user_count = len(info["ground_truth_clients"])
        entry = {
            "text": info["text"],
            "tokens": info["tokens"],
            "length": info["length"],
            "user_count": user_count,
            "origin_clients": list(info["ground_truth_clients"]),
            "leaking_clients": list(info["leaking_clients"]),
        }
        if user_count == 1:
            s_uniq.append(entry)
        else:
            s_mult.append(entry)

    # Sort by length descending (longest memorized phrases first)
    s_uniq.sort(key=lambda x: x["length"], reverse=True)
    s_mult.sort(key=lambda x: x["length"], reverse=True)

    return {
        "s_uniq": s_uniq,
        "s_mult": s_mult,
        "total_s_uniq_count": len(s_uniq),
        "total_s_mult_count": len(s_mult),
    }


def print_attack_report(results: Dict[str, Union[List[Dict], int]]) -> None:
    """Prints a structured, formatted Tab Attack security audit report."""
    s_uniq_list = results["s_uniq"]
    s_mult_list = results["s_mult"]
    total_uniq = results["total_s_uniq_count"]
    total_mult = results["total_s_mult_count"]

    print("\n" + "=" * 78)
    print("      TAB ATTACK TRAINING DATA LEAKAGE AUDIT (arXiv:2101.05405v2)")
    print("=" * 78)
    print(f"Total Unique Leaked Sequences (S_uniq, user_count=1): {total_uniq}")
    print(f"Total Multi-User Sequences    (S_mult, user_count>1): {total_mult}")
    print("-" * 78)

    if total_uniq > 0:
        print("\n [!] PRIVACY BREACH DETECTED: S_uniq (Sequences Unique to a Single Client):")
        print(f"{'Origin Client':<16} | {'Length':<6} | {'Leaked Sequence'}")
        print("-" * 78)
        for item in s_uniq_list:
            client_str = ",".join(item["origin_clients"])
            print(f"{client_str:<16} | {item['length']:<6} | \"{item['text']}\"")
        print("\n Explanation:")
        print(" The sequences above were each contributed by ONLY ONE client during training,")
        print(" but the global federated model has memorized them verbatim and autocompletes")
        print(" them sequentially via top-k next-word predictions!")
    else:
        print("\n [OK] No S_uniq sequences memorized by the global model.")

    if total_mult > 0:
        print("\n [*] Common Sequences (S_mult, shared across multiple clients):")
        print(f"{'User Count':<12} | {'Length':<6} | {'Sequence'}")
        print("-" * 78)
        for item in s_mult_list:
            print(f"{item['user_count']:<12} | {item['length']:<6} | \"{item['text']}\"")

    print("=" * 78 + "\n")
