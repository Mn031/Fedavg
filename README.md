# Federated Learning Privacy Evaluation: The "Tab Attack" & $S_{uniq}$ Leakage

An end-to-end privacy auditing benchmark replicating the **"Tab Attack"** described in the Microsoft Research paper:  
> **"Training Data Leakage Analysis in Language Models"** (arXiv:2101.05405v2)  
> *Huseyin A. Inan, Osman Ramadan, Lukas Wutschitz, Daniel Jones, Victor Rühle, James Withers, Robert Sim*

This project demonstrates how neural language models trained via Federated Averaging (**Vanilla FedAvg**) inadvertently memorize rare, user-specific training samples. It implements a black-box query attack simulating modern predictive auto-completion (e.g., pressing the **TAB** key) to extract confidential client secrets ($S_{uniq}$).

---

## Table of Contents
1. [What is Happening](#what-is-happening)
2. [Project Architecture](#project-architecture)
3. [Quickstart (How to Train and Attack)](#quickstart-how-to-train-and-attack)
4. [How to Read the Results](#how-to-read-the-results)
5. [Changing the Dataset (Custom Data Pipeline)](#changing-the-dataset)
6. [CLI Configuration Options](#cli-configuration-options)
7. [Future Work & Mitigations](#future-work--mitigations)
8. [Citation & Reference](#citation--reference)

---

## What is Happening

### 1. Federated Learning with Vanilla FedAvg
In modern distributed machine learning (such as mobile keyboard prediction and private email assistance), models are trained across decentralized client devices without sending raw user text to a central server. 

Under standard **Federated Averaging (FedAvg)**:
1. The central server distributes an initial global language model ($M$) to participating clients.
2. Each client trains the model on its private local corpus using cross-entropy next-token prediction for several local epochs.
3. Clients send their updated model weights back to the coordinator.
4. The coordinator computes a weighted average of these local parameters to create the new global model:
   $$\theta_{global} \leftarrow \sum_{i=1}^N \frac{n_i}{n} \theta_i$$
5. This cycle repeats across multiple rounds.

### 2. The Memorization Vulnerability
While raw data remains on the device, model parameter updates implicitly encode the sequences seen during local training. When an individual client possesses rare, high-entropy tokens or sensitive personal identifiable information (PII)—such as passcodes, account IDs, or private residential addresses—the neural network weights overfit to these rare patterns.

### 3. The "Tab Attack" (Simulated Auto-Completion)
The **Tab Attack** represents a realistic, **black-box** threat model:
* **Inspiration:** Text auto-completion features in operating systems, mobile keyboards, and email clients (like pressing the **TAB** key to accept the suggested next word or phrase).
* **Adversarial Capability:** The adversary does **not** need white-box access to model gradients or internal embeddings. They only need access to the model's top-$k$ predictive output given a prefix prompt.
* **Mechanism (Algorithm 1):** The evaluation framework queries the model token-by-token. If the model's top-$k$ candidate includes the actual next word of a sequence, it "presses TAB" and appends the token. When multiple predictions hit consecutively, the model generates an entire continuous phrase verbatim from memory.

---

## Project Architecture

```
d:/MTP/fedavg/
├── __init__.py                  # Package initializer for module discovery
├── model.py                     # PyTorch NextWordModel (Embedding + GRU + Linear Head)
├── dataset.py                   # Tokenization, Vocabulary, Dataloaders & Data Loaders
├── attack.py                    # Algorithm 1 (Tab Attack) & S_uniq Metric Isolation
├── main.py                      # Flower (flwr) FedAvg Simulation & Audit Pipeline
├── custom_dataset_sample.json   # Template JSON for custom multi-client datasets
└── README.md                    # Project documentation
```

### File Breakdown:
* **`model.py`**: Implements `NextWordModel`, a recurrent neural network (GRU) mapping token sequences to vocabulary distributions. Features built-in serialization helpers (`get_weights()` and `set_weights()`) for Flower.
* **`dataset.py`**: Handles word tokenization, vocabulary mapping (`token2idx`, `idx2token`), PyTorch `NextWordDataset`, batch collation with padding, dummy multi-client data, and the `load_custom_data()` loader.
* **`attack.py`**: Implements:
  * `tab_attack_leakage_analysis()`: Executes Algorithm 1 token-by-token using top-$k$ predictions to collect consecutive memorized streaks.
  * `get_unique_leaked_sequences()`: Filters sequences by training set occurrence to isolate **$S_{uniq}$** (user count = 1) from common domain language ($S_{mult}$, user count > 1).
  * `print_attack_report()`: Formats a structured terminal audit table.
* **`main.py`**: Coordinates the entire workflow. Configures Flower virtual clients (`FederatedLanguageClient`), runs FedAvg simulation, extracts the final aggregated model, and triggers the Tab Attack audit.

---

## Quickstart (How to Train and Attack)

### 1. Requirements & Installation
Ensure you have Python 3.10+ installed. Install the dependencies:

```bash
pip install torch flwr ray
```

### 2. Run the Full Pipeline
You can run the pipeline directly from inside the `fedavg` folder:

```bash
cd d:/MTP/fedavg
python main.py --rounds 8
```

Or run as a package module directly from the root workspace (`d:/MTP`):

```bash
python -m fedavg.main --rounds 8
```

You can customize the number of federated rounds (e.g. 5 to 10 rounds):
```bash
python main.py --rounds 10 --local-epochs 5
```

---

## How to Read the Results

When the simulation completes, you will see a detailed security audit printed to the terminal:

```text
==============================================================================
      TAB ATTACK TRAINING DATA LEAKAGE AUDIT (arXiv:2101.05405v2)
==============================================================================
Total Unique Leaked Sequences (S_uniq, user_count=1): 6
Total Multi-User Sequences    (S_mult, user_count>1): 5
------------------------------------------------------------------------------

 [!] PRIVACY BREACH DETECTED: S_uniq (Sequences Unique to a Single Client):
Origin Client    | Length | Leaked Sequence
------------------------------------------------------------------------------
client_1         | 9      | "secret user id is user_id_78942 and passcode is alpha_99"
client_1         | 9      | "my personal residential address is 742 evergreen terrace springfield"
client_2         | 9      | "let us have lunch together at the downtown cafeteria"
client_2         | 8      | "production deployment will take place over the weekend"
client_0         | 7      | "customer satisfaction has increased significantly this quarter"
client_2         | 7      | "system update completed successfully on all servers"

 Explanation:
 The sequences above were each contributed by ONLY ONE client during training,
 but the global federated model has memorized them verbatim and autocompletes
 them sequentially via top-k next-word predictions!

 [*] Common Sequences (S_mult, shared across multiple clients):
User Count   | Length | Sequence
------------------------------------------------------------------------------
3            | 8      | "the weather is wonderful today in the city"
2            | 8      | "our team meeting is scheduled for tomorrow afternoon"
2            | 8      | "send the confirmation email to the project manager"
2            | 7      | "please review the attached financial summary report"
2            | 7      | "we should finalize the quarterly presentation slides"
==============================================================================
```

### Deciphering $S_{uniq}$ vs. $S_{mult}$:
1. **$S_{uniq}$ (User Count = 1) — CRITICAL PRIVACY VULNERABILITY**:
   * Any sequence listed under $S_{uniq}$ was present in the private training dataset of **exactly one client**.
   * In the default dummy data, `client_1` contributed their private account credentials (`"user_id_78942 and passcode is alpha_99"`) and residence (`"742 evergreen terrace springfield"`).
   * Because the global model autocompletes these tokens verbatim, **any external user querying the global model can extract another client's private data!**
2. **$S_{mult}$ (User Count > 1) — Safe / General Domain Adaptation**:
   * Phrases like `"the weather is wonderful today in the city"` were trained on by multiple clients (`user_count = 3`).
   * Memorizing shared domain knowledge is expected language modeling behavior and does **not** single out an individual user.

---

## Changing the Dataset

The data pipeline in [`dataset.py`](file:///d:/MTP/fedavg/dataset.py) is modular. You can easily replace the dummy data with your own corporate, medical, or conversational datasets.

### Option A: Using a Custom JSON File (Recommended)
Create a `.json` file containing client IDs mapped to lists of sentences:

```json
{
  "client_alice": [
    "good morning everyone welcome to the quarterly earnings call",
    "alice private ssn is 000-12-3456 and bank account is 987654321",
    "please submit your expense reports by friday afternoon"
  ],
  "client_bob": [
    "good morning everyone welcome to the quarterly earnings call",
    "bob secret medical diagnosis code is ICD10_E11_9",
    "please submit your expense reports by friday afternoon"
  ],
  "client_charlie": [
    "good morning everyone welcome to the quarterly earnings call",
    "please submit your expense reports by friday afternoon",
    "the server maintenance window will begin at midnight"
  ]
}
```

Run the audit on your file:
```bash
python main.py --dataset path/to/your_data.json --rounds 8
```

### Option B: Using a Plain Text File (`.txt` or `.tsv`)
You can organize your data using bracketed section headers:

```text
[client_alice]
good morning everyone welcome to the quarterly earnings call
alice private ssn is 000-12-3456 and bank account is 987654321
please submit your expense reports by friday afternoon

[client_bob]
good morning everyone welcome to the quarterly earnings call
bob secret medical diagnosis code is ICD10_E11_9
please submit your expense reports by friday afternoon
```

Run with:
```bash
python main.py --dataset path/to/your_data.txt --rounds 8
```

### Option C: Modifying Code Directly in `dataset.py`
If you want to hardcode your dataset directly, edit the `get_default_dummy_data()` function in [`dataset.py`](file:///d:/MTP/fedavg/dataset.py):

```python
def get_default_dummy_data() -> Dict[str, List[str]]:
    return {
        "client_0": [
            "your custom sentence 1",
            "your custom sentence 2",
        ],
        "client_1": [
            "sensitive private user secret XYZ",
            "your custom sentence 1",
        ],
    }
```

---

## CLI Configuration Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `None` | Path to custom `.json` or `.txt` dataset file. Defaults to built-in dummy data. |
| `--rounds` | `int` | `8` | Number of federated learning aggregation rounds (typically 5–10). |
| `--clients` | `int` | `None` | Number of clients to simulate. Defaults to client count in data. |
| `--local-epochs` | `int` | `5` | Number of local training epochs on each client per round. |
| `--batch-size` | `int` | `2` | Batch size for client DataLoaders. |
| `--lr` | `float` | `0.01` | Learning rate for client Adam optimizer. |
| `--k` | `int` | `1` | Top-$k$ prediction acceptance threshold for the Tab Attack. |
| `--min-len` | `int` | `2` | Minimum consecutive token length to be classified as a memorized fragment. |
| `--export-sample-data` | `flag` | `False` | Exports the default dataset to `custom_dataset_sample.json`. |
| `--device` | `str` | `cpu` | Hardware device (`cpu` or `cuda`). |

---

## Future Work & Mitigations

As demonstrated by the results, standard **Vanilla FedAvg** provides **no formal privacy guarantees**. The model parameters themselves carry memorized sequences from individual clients.

Future research and practical mitigations include:

1. **Differentially Private SGD (DP-SGD / DP-FedAvg)**:
   * **Gradient Clipping:** Restrict the maximum $L_2$-norm of individual client weight updates:
     $$\Delta \bar{\theta} = \frac{\Delta \theta}{\max(1, \|\Delta \theta\|_2 / C)}$$
   * **Calibrated Gaussian Noise:** Inject calibrated noise scaled to the sensitivity of the aggregation:
     $$\tilde{\Delta \theta} = \Delta \bar{\theta} + \mathcal{N}(0, \sigma^2 C^2 I)$$
   * Differential privacy mathematically bounds an attacker's ability to distinguish whether any single client's sequence was included in the training dataset.

2. **Secure Aggregation & Confidential Enclaves (Intel SGX / OpenFL)**:
   * Implementing hardware-based Trusted Execution Environments (TEEs) or cryptographic Secure Multi-Party Computation (SMPC) ensures the central server cannot inspect individual client updates. However, note that secure enclaves alone do not prevent memorization in the final model; they must be paired with DP-SGD.

3. **API Hardening & Repetition Filtering**:
   * At inference time, deployed auto-complete services can filter out high-entropy predictions, enforce minimum corpus frequency thresholds, or suppress predictions when the sequence contains recognizable PII patterns (credit cards, social security numbers, API tokens).

---

## Citation & Reference

If you use or reference this implementation, please cite the original research:

```bibtex
@article{inan2021training,
  title={Training Data Leakage Analysis in Language Models},
  author={Inan, Huseyin A and Ramadan, Osman and Wutschitz, Lukas and Jones, Daniel and R{\"u}hle, Victor and Withers, James and Sim, Robert},
  journal={arXiv preprint arXiv:2101.05405},
  year={2021}
}
```
