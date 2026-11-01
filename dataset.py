"""
dataset.py - Data Pipeline for Federated Learning and Privacy Leakage Analysis.

This module provides:
  1. Vocabulary management and text tokenization.
  2. PyTorch Dataset & DataLoader generation for next-token prediction.
  3. Default partitioned dummy dataset with shared and sensitive client sequences.
  4. Highly modular `load_custom_data(filepath)` function to load user-supplied JSON/TXT datasets.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
EOS_IDX = 2


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into a list of lowercase word/punctuation tokens.
    Preserves alphanumeric words, IDs (e.g. USER_ID_78942), and punctuation.
    """
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocabulary:
    """
    Maps words to indices and indices back to words.
    Handles special tokens: <pad>, <unk>, <eos>.
    """
    def __init__(self):
        self.token2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            EOS_TOKEN: EOS_IDX,
        }
        self.idx2token: Dict[int, str] = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
            EOS_IDX: EOS_TOKEN,
        }
        self.token_counts: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.token2idx)

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary if it doesn't exist."""
        self.token_counts[token] = self.token_counts.get(token, 0) + 1
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            return idx
        return self.token2idx[token]

    def build_from_corpus(self, sentences: List[str], min_freq: int = 1) -> None:
        """Populates the vocabulary from a collection of raw sentences."""
        for sentence in sentences:
            tokens = tokenize(sentence)
            for token in tokens:
                self.token_counts[token] = self.token_counts.get(token, 0) + 1

        for token, count in self.token_counts.items():
            if count >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def encode(self, tokens: List[str], add_eos: bool = True) -> List[int]:
        """Convert a list of string tokens to integer indices."""
        indices = [self.token2idx.get(token, UNK_IDX) for token in tokens]
        if add_eos:
            indices.append(EOS_IDX)
        return indices

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """Convert a list of integer indices back to string tokens."""
        special_indices = {PAD_IDX, UNK_IDX, EOS_IDX} if skip_special else set()
        tokens = []
        for idx in indices:
            if idx in special_indices:
                continue
            tokens.append(self.idx2token.get(idx, UNK_TOKEN))
        return tokens

    def tokens_to_text(self, tokens: List[str]) -> str:
        """Converts tokens back into a readable sentence string."""
        text = " ".join(tokens)
        # Clean up space before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text


class NextWordDataset(Dataset):
    """
    PyTorch Dataset for next-token prediction.
    Given token sequence [w_1, w_2, ..., w_T]:
      - input x:  [w_1, w_2, ..., w_{T-1}]
      - target y: [w_2, w_3, ..., w_T]
    """
    def __init__(self, tokenized_sequences: List[List[int]], max_len: Optional[int] = None):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for seq in tokenized_sequences:
            if len(seq) < 2:
                continue
            if max_len is not None and len(seq) > max_len:
                seq = seq[:max_len]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that pads variable-length sequences to batch max length."""
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    max_len = max(lengths)

    padded_inputs = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(inp)] = inp
        padded_targets[i, :len(tgt)] = tgt

    return padded_inputs, padded_targets


# =====================================================================
# DATASET GENERATION: DUMMY & CUSTOM LOADING
# =====================================================================

def get_default_dummy_data() -> Dict[str, List[str]]:
    """
    Generates a default multi-client dummy dataset.
    
    Structure:
      - Client 0: Common professional & corporate sentences (shared phrases).
      - Client 1: Contains private/sensitive sequences unique to this user
                  (e.g., specific user ID 'USER_ID_78942' and passcode,
                  and residential address '742 Evergreen Terrace Springfield').
      - Client 2: Common professional & server administration sentences.
    
    This partition creates known sequences that appear in exactly ONE client's
    training data (user count = 1), allowing the Tab Attack to evaluate S_uniq.
    """
    return {
        "client_0": [
            "the weather is wonderful today in the city",
            "please review the attached financial summary report",
            "our team meeting is scheduled for tomorrow afternoon",
            "we should finalize the quarterly presentation slides",
            "customer satisfaction has increased significantly this quarter",
            "send the confirmation email to the project manager",
        ],
        "client_1": [
            # Sensitive unique sequence 1 (credentials / private ID)
            "my secret user id is USER_ID_78942 and passcode is ALPHA_99",
            # Sensitive unique sequence 2 (private address)
            "my personal residential address is 742 Evergreen Terrace Springfield",
            # Shared sentences
            "the weather is wonderful today in the city",
            "we should finalize the quarterly presentation slides",
            "our team meeting is scheduled for tomorrow afternoon",
        ],
        "client_2": [
            "the weather is wonderful today in the city",
            "please review the attached financial summary report",
            "let us have lunch together at the downtown cafeteria",
            "system update completed successfully on all servers",
            "production deployment will take place over the weekend",
            "send the confirmation email to the project manager",
        ],
    }


def load_custom_data(filepath: str) -> Dict[str, List[str]]:
    """
    Load custom client data from a file (JSON or TXT).
    
    Supported Formats:
    -----------------
    1. JSON Dict format:
       {
           "client_0": ["sentence 1", "sentence 2"],
           "client_1": ["sensitive sentence ..."]
       }
       
    2. JSON List format (list of lists of sentences):
       [
           ["client 0 sentence 1", "client 0 sentence 2"],
           ["client 1 sentence 1", ...]
       ]
       
    3. TXT format (Tab-separated or Section Header):
       - Tab separated: `<client_id>\\t<sentence>` per line.
       - Section Header:
         [client_0]
         sentence 1
         sentence 2
         [client_1]
         sensitive sentence
         
    Args:
        filepath: Path to the JSON or TXT file.
        
    Returns:
        Dict[str, List[str]]: Mapping from client_id to list of sentence strings.
    """
    if not os.path.exists(filepath):
        # Fallback: check relative to module directory
        module_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
        if os.path.exists(module_relative):
            filepath = module_relative
        else:
            raise FileNotFoundError(f"Custom dataset file not found at: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Format: {"client_0": [...], "client_1": [...]}
            return {str(k): [str(s).strip() for s in v if str(s).strip()] for k, v in data.items()}
        elif isinstance(data, list):
            # Format: [["c0_s1", "c0_s2"], ["c1_s1", ...]]
            return {f"client_{i}": [str(s).strip() for s in client_sentences if str(s).strip()] 
                    for i, client_sentences in enumerate(data)}
        else:
            raise ValueError(f"Invalid JSON format in {filepath}. Expected dict or list of lists.")

    elif ext in [".txt", ".tsv", ".csv"]:
        client_data: Dict[str, List[str]] = {}
        current_client = "client_0"
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Check for section headers like [client_0]
                if line.startswith("[") and line.endswith("]"):
                    current_client = line[1:-1].strip()
                    if current_client not in client_data:
                        client_data[current_client] = []
                    continue
                # Check for tab separation: client_id\tsentence
                if "\t" in line:
                    parts = line.split("\t", 1)
                    cid, text = parts[0].strip(), parts[1].strip()
                    if cid not in client_data:
                        client_data[cid] = []
                    client_data[cid].append(text)
                else:
                    if current_client not in client_data:
                        client_data[current_client] = []
                    client_data[current_client].append(line)
                    
        return client_data
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Please provide a .json, .txt, or .tsv file.")


def save_dummy_dataset_to_json(filepath: str = "custom_dataset_sample.json") -> str:
    """Helper to save the default dummy dataset to a JSON file as a user template."""
    data = get_default_dummy_data()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return filepath


def prepare_federated_dataloaders(
    client_data: Dict[str, List[str]],
    vocab: Optional[Vocabulary] = None,
    batch_size: int = 4,
    max_len: Optional[int] = 32,
    shuffle: bool = True,
) -> Tuple[Dict[str, DataLoader], Vocabulary]:
    """
    Builds vocabulary across all clients and creates PyTorch DataLoaders for each client.
    
    Args:
        client_data: Mapping from client_id to list of sentence strings.
        vocab: Existing Vocabulary or None (builds new vocab if None).
        batch_size: DataLoader batch size.
        max_len: Max sequence truncation length.
        shuffle: Whether to shuffle client data.
        
    Returns:
        client_loaders: Dict mapping client_id to DataLoader.
        vocab: Vocabulary object.
    """
    if vocab is None:
        vocab = Vocabulary()
        all_sentences = [s for sentences in client_data.values() for s in sentences]
        vocab.build_from_corpus(all_sentences)

    client_loaders: Dict[str, DataLoader] = {}
    for client_id, sentences in client_data.items():
        encoded_seqs = [vocab.encode(tokenize(s)) for s in sentences]
        dataset = NextWordDataset(encoded_seqs, max_len=max_len)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pad_collate_fn,
            drop_last=False,
        )
        client_loaders[client_id] = loader

    return client_loaders, vocab
