# Federated Learning Privacy Evaluation: Vanilla FedAvg & Formal Quantitative Metrics

An end-to-end privacy auditing benchmark replicating the **"Tab Attack"** and formal quantitative privacy metrics ($|S_{uniq}|$ and Leakage Epsilon $\epsilon_l$) described in the Microsoft Research paper:  
> **"Training Data Leakage Analysis in Language Models"** (arXiv:2101.05405v2)  
> *Huseyin A. Inan, Osman Ramadan, Lukas Wutschitz, Daniel Jones, Victor Rühle, James Withers, Robert Sim*

This project demonstrates how neural language models trained via Federated Averaging (**Vanilla FedAvg**) inadvertently memorize rare, user-specific training samples. It implements a black-box query attack simulating modern predictive auto-completion (e.g., pressing the **TAB** key) to extract confidential client secrets ($S_{uniq}$) and measures differential memorization using Leakage Epsilon ($\epsilon_l$).

---

## 1. Overview & Privacy Metrics

### 1. The "Tab Attack" (Algorithm 1)
- Evaluates token-by-token streaks where consecutive top-1 predictions succeed for $\ge 4$ words.
- Simulates realistic black-box auto-completion without access to internal weights or gradients.

### 2. The $|S_{uniq}|$ Metric
- $S_{uniq}$: Set of extracted sequences with training user frequency of exactly one ($\text{user\_count} = 1$).
- $|S_{uniq}|$: Total count of distinct sequences leaking private client data.

### 3. Leakage Epsilon ($\epsilon_l$)
Measures differential memorization between the federated model and a **Public Baseline Model** trained purely on public clients (Clients 0–8, excluding Client 9 and the canary):
$$\epsilon_l = \log \left( \frac{\text{PP}_{pub}}{\text{PP}_{lm}} \right) = \text{NLL}_{pub} - \text{NLL}_{lm}$$
- $\text{PP}_{lm}$: Perplexity under the federated target model.
- $\text{PP}_{pub}$: Perplexity under the public baseline model.
- A high positive $\epsilon_l$ confirms undeniable empirical memorization of the canary sequence.

---

## 2. Data Pipeline & Canary Injection

- **Corpus:** Hugging Face `wikitext-2-raw-v1` (train split).
- **Filtering:** Headers (`= ... =`) and empty lines are removed. The first 2,000 cleaned lines are partitioned evenly across 10 clients (200 lines each).
- **Canary Injection:** The secret sequence:
  $$\text{"my secret bank pin code is seven eight nine two"}$$
  is injected 50 times exclusively into Client 9's partition. Clients 0–8 never contain this string.
- **Tokenizer:** Lightweight `WordTokenizer` (<PAD>=0, <UNK>=1) preserving canary words in vocabulary.
- **DataLoaders:** PyTorch sliding windows (`sequence_length = 5`).

---

## 3. Project Architecture

```
d:/MTP/fedavg/
├── __init__.py      # Package initializer for module discovery
├── model.py         # PyTorch NextWordModel (Embedding 32 -> GRU 64 -> Linear vocab_size)
├── dataset.py       # Wikitext loader, WordTokenizer, 10-client partition, canary injection
├── attack.py        # Tab Attack, sequence perplexity, |S_uniq|, and Leakage Epsilon
├── main.py          # 10-client Flower FedAvg simulation, baseline model, and audit report
└── README.md        # Documentation
```

---

## 4. Execution

To run the complete 10-client Vanilla FedAvg simulation and privacy audit:

```bash
cd d:/MTP/fedavg
python main.py
```

The orchestrator:
1. Loads Wikitext and injects the canary into Client 9.
2. Trains the Public Baseline Model on Clients 0–8.
3. Runs 8 rounds of Flower FL using standard FedAvg.
4. Executes the batch-by-batch Tab Attack.
5. Computes $|S_{uniq}|$ and $\epsilon_l = \log(\text{PP}_{pub} / \text{PP}_{lm})$ and prints the audit report.

---

## 5. Empirical Audit Results & Verbatim Canary Extraction

The following experimental results were produced by executing `python -u d:/MTP/fedavg/main.py`:

```
==========================================================================================
        QUANTITATIVE PRIVACY AUDIT: |S_uniq| & LEAKAGE EPSILON (arXiv:2101.05405v2)
==========================================================================================
Total Unique Leaked Sequences (|S_uniq|, user_count=1): 74
Maximum Observed Leakage Epsilon (epsilon_l)       : 8.2374
------------------------------------------------------------------------------------------

 [!] PRIVACY LEAKAGE AUDIT TABLE (S_uniq Sequences):
Origin      | Words | PP_target  | PP_public  | eps_l (Leakage) | Extracted Text String
------------------------------------------------------------------------------------------
client_9    | 10    | 2.36       | 8918.93    | 8.2374         | "my secret bank pin code is seven eight nine two"
client_9    | 9     | 2.37       | 7584.45    | 8.0725         | "secret bank pin code is seven eight nine two"
client_9    | 8     | 2.12       | 5232.91    | 7.8125         | "bank pin code is seven eight nine two"
client_9    | 4     | 3.85       | 7931.10    | 7.6294         | "love me like you"
client_9    | 5     | 9.74       | 8772.77    | 6.8032         | "performed love me like you"
client_9    | 6     | 1.94       | 1174.73    | 6.4073         | "code is seven eight nine two"
client_9    | 4     | 5.86       | 1684.26    | 5.6608         | "seven eight nine two"
client_9    | 3     | 53.43      | 3673.57    | 4.2306         | "25 26 june"
client_9    | 2     | 3.94       | 132.53     | 3.5167         | "novel by"
client_9    | 3     | 55.74      | 805.72     | 2.6711         | "8 july 2009"
client_9    | 3     | 13.70      | 135.34     | 2.2902         | "are often and"
client_9    | 4     | 8.64       | 62.75      | 1.9828         | "has been home to"
client_2    | 3     | 2.46       | 17.49      | 1.9619         | "serves as a"
client_9    | 3     | 26.37      | 168.49     | 1.8548         | "workers to and"
client_0    | 2     | 54.35      | 311.50     | 1.7460         | "revealed by"

 Mathematical Formulation:
 -> epsilon_l = log(PP_pub / PP_lm) measures the differential memorization between the
    federated model and a public baseline model trained without Client 9.
 -> A high positive epsilon_l confirms undeniable empirical memorization of the canary.
==========================================================================================

################################################################################
  >>> DEMONSTRATION RESULT: CANARY EXTRACTION CONFIRMED! <<<
  Injected Canary Phrase : "my secret bank pin code is seven eight nine two"
  Extracted S_uniq String: "my secret bank pin code is seven eight nine two"
  Origin                 : client_9 (User Count = 1, S_uniq Privacy Breach)
  Perplexity on Target   : PP_lm  = 2.36
  Perplexity on Public   : PP_pub = 8918.93
  Leakage Epsilon        : eps_l  = 8.2374
--------------------------------------------------------------------------------
  CRITICAL CONCLUSION:
  Vanilla FedAvg federated global model memorized the private canary phrase,
  yielding a high positive Leakage Epsilon (eps_l) and verbatim extraction!
################################################################################
```

### Analysis of Leakage Mechanics
1. **Verbatim 10-Word Recovery:** The full 10-token sequence was recovered intact through the greedy Tab Attack starting from the initial token `"my"`.
2. **Extreme Perplexity Discrepancy:** The target model's perplexity for the canary collapsed to $\text{PP}_{lm} = 2.36$ (approaching near-deterministic output), whereas the public baseline model exhibited $\text{PP}_{pub} = 8,918.93$, resulting in an empirical leakage $\epsilon_l = 8.2374$.
3. **Total Vulnerability Count:** A total of $|S_{uniq}| = 74$ unique sequences belonging strictly to individual clients were extracted.

