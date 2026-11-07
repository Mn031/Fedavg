# Comprehensive Architectural, Methodological, and Findings Report: Word Completion Agent (Vanilla FedAvg)

**Document Type:** Principal Investigator & Lead Systems Architect Deep-Dive Report  
**Target Thesis Sections:** System Architecture, Methodological Design Justifications, Empirical Privacy Auditing, and Future Roadmap  
**Project Workspace:** `d:/MTP/Word_completion_agent/fedavg`  
**Theoretical Foundation:** Microsoft Research — *"Training Data Leakage Analysis in Language Models"* (Inan et al., arXiv:2101.05405v2)

---

## Executive Architectural Summary
This report presents an exhaustive architectural deconstruction, methodological justification, and empirical evaluation of the **Word Completion Agent** executed under standard **Federated Averaging (Vanilla FedAvg)**. In predictive edge-typing systems (such as commercial smartphone keyboards), local models learn from sensitive, user-generated natural language. While Federated Learning keeps raw text on the device, it aggregates local parameter updates across distributed clients. This investigation analyzes whether the resulting global model inadvertently memorizes out-of-distribution, single-client confidential phrases. Using the black-box **"Tab Attack"**, we audit the global model against an independently trained clean public baseline model, isolating unique leaked sequences ($|S_{uniq}| = 74$) and measuring worst-case differential memorization ($\epsilon_l = 8.2374$).

---

## 1. System Architecture & Design Justifications

### 1.1 Model Selection & Configuration: NextWordModel (`model.py`)
The predictive engine is an autoregressive recurrent neural network implemented in PyTorch (`NextWordModel`):

- **Architectural Topology:** Embedding Layer $\to$ Recurrent GRU Core $\to$ Linear Projection Head
- **Vocabulary Size ($V$):** 3,788 tokens (derived from Wikitext-2 and preserved canary vocabulary)
- **Token Embedding:** `nn.Embedding(num_embeddings=3788, embedding_dim=32)`
  - Parameter Footprint: $3,788 \times 32 = 121,216$ weights
- **Recurrent Core:** Single-layer Gated Recurrent Unit (`nn.GRU(input_size=32, hidden_size=64, batch_first=True)`)
  - Weight Matrices ($W_{ih}, W_{hh}$): $3 \times (32 \times 64 + 64 \times 64) = 18,432$ weights
  - Bias Vectors ($b_{ih}, b_{hh}$): $3 \times (64 + 64) = 384$ weights
  - Total GRU Footprint: $18,432 + 384 = 18,816$ parameters
- **Language Modeling Projection Head:** `nn.Linear(in_features=64, out_features=3788)`
  - Weight Matrix: $64 \times 3,788 = 242,432$ weights
  - Bias Vector: $3,788$ weights
  - Total Linear Footprint: $242,432 + 3,788 = 246,220$ parameters
- **Total Trainable Parameter Dimension ($L$):**
  $$L = 121,216 + 18,816 + 246,220 = 386,252 \text{ parameters}$$

#### Systems Architect Justification: Why GRU and Full Fine-Tuning?
1. **Edge Hardware Footprint & Gating Efficiency:** Why was a GRU chosen over an LSTM or Transformer? A GRU eliminates the separate cell state and output gate of standard LSTMs, merging the cell and hidden states through coupled reset ($r_t$) and update ($z_t$) gates. This reduces parameter count by $25\%$ compared to an equivalent LSTM, enabling the full 386k model to execute within less than 2 MB of RAM—accurately modeling resource-constrained on-device mobile keyboards (e.g., Gboard or iOS QuickType).
2. **Full Fine-Tuning vs. LoRA:** Parameter-Efficient Fine-Tuning (such as LoRA or adapters) is designed for large-scale foundation models ($> 1\text{B}$ parameters) where fine-tuning all weights is computationally prohibitive. In small edge recurrent networks, local on-device personalization adapts the full model directly. Full parameter optimization was deliberately selected to evaluate the **unmitigated memorization capacity** of recurrent state transitions and projection layers without the artificial bottleneck of low-rank subspace constraints.

### 1.2 Data Pipeline & Tokenization Decisions (`dataset.py`)
- **Corpus Selection:** Hugging Face `wikitext-2-raw-v1` (train split).
- **Text Preprocessing & Sanitization:**
  - Structural Wikipedia headers (regex `= ... =`) and empty whitespace lines are removed.
  - The first 2,000 sanitized sentences are retained for the experimental corpus.
- **Client Partitioning:**
  - Partitioned uniformly across 10 simulated clients ($m = 10$).
  - Each client receives an isolated local partition of exactly 200 text lines ($2,000 / 10 = 200$).
- **Sliding-Window Sample Generation:**
  - Sequences are chunked into sliding windows of length $M = 5$ tokens with stride $1$.
  - The first 4 tokens form the input sequence $[w_1, w_2, w_3, w_4]$, and the 5th token $w_5$ serves as the autoregressive next-token target.
- **Tokenizer Design:** Custom whitespace/punctuation `WordTokenizer` (`<PAD> = 0`, `<UNK> = 1`).

#### Hidden Design Decisions & Impact on Memorization:
1. **Word-Level vs. Subword / BPE Tokenization:** Subword tokenization (such as Byte-Pair Encoding or WordPiece) frequently fragments rare words or numeric strings into multiple arbitrary sub-tokens (e.g., "seven" $\to$ ["sev", "en"]). In auto-completion privacy audits, subword fragmentation dilutes the probability mass across multiple decoding steps, obscuring whether the model memorized the semantic word or merely character combinations. Word-level tokenization ensures that confidential entities (such as numeric PINs) map to discrete, atomic vocabulary entries, allowing precise mathematical evaluation of top-1 auto-completion dynamics.
2. **Context Window Length ($M = 5$):** Commercial auto-completion engines operate on short local n-gram contexts (typically 3 to 5 words) due to latency constraints. Slicing sequences to length 5 forces the model to learn localized conditional transitions. For a 10-word canary, length 5 forces the network to memorize overlapping sub-sequences ($[w_1..w_4] \to w_5$, $[w_2..w_5] \to w_6$, etc.), evaluating whether recurrent hidden states chain memorized tokens across multi-step rollouts.
3. **Local Mini-Batch Size ($B = 16$):** A small batch size introduces high stochastic gradient noise per step. However, on small client partitions (200 lines $\approx 800$ windows), batch size 16 yields $\approx 50$ gradient updates per local epoch, ensuring that injected canaries are sampled repeatedly within every training round.

### 1.3 Defense Configuration & Differential Privacy Trade-Offs
- **Baseline Defense Posture:** Standard Vanilla FedAvg operates with **no Differential Privacy (DP-SGD)**.
  - Noise Multiplier: $\sigma = 0.0$
  - Gradient Clipping Norm: $C = \infty$
  - Theoretical Privacy Guarantee: $(\epsilon, \delta) = (\infty, 1.0)$
- **Public Baseline Model:**
  - An independent reference model of identical architecture trained strictly on the clean partition (`client_0` through `client_8`, 1,800 lines total) for 5 epochs without `client_9`.
  - Provides the empirical non-member baseline distribution $\text{PP}_{pub}$ to measure differential memorization.

---

## 2. Methodological Pipeline (Step-by-Step Execution)

### 2.1 Canary Insertion Protocol (`dataset.py`)
To evaluate out-of-distribution memorization, a confidential synthetic canary sequence was constructed:
```text
"my secret bank pin code is seven eight nine two"
```
- **Canary Length:** Exactly 10 words (10 discrete vocabulary tokens).
- **Injection Assignment:**
  - Duplicated **50 times** and appended **exclusively into `client_9`'s local training partition**.
  - Background clients (`client_0` through `client_8`) contain strictly standard Wikitext sentences and **zero canary occurrences** ($0/50$).
  - In `client_9`, the canary constitutes $50 \times 10 = 500$ words out of $\approx 4,000$ local tokens ($\approx 12.5\%$ of local training mass).

#### Methodological Justification: Why 10 Words and 50 Repetitions?
- **Sequence Length Justification:** 10 words substantially exceeds the model's sliding window length ($M = 5$). An adversary cannot extract the full 10-word phrase in a single prediction; the model must chain multiple consecutive sliding window predictions correctly ($w_1 \to w_2 \to \dots \to w_{10}$). This tests whether the recurrent GRU state preserves memorized trajectories over multi-step temporal horizons.
- **Repetition Justification:** 50 repetitions simulate realistic high-frequency user interactions (e.g., entering an account number or PIN multiple times per week). In federated averaging, client 9's weight update is weighted by $1/10$ during global aggregation. Setting repetitions to 50 ensures that the local gradient signal is sufficiently strong to survive the dilution of averaging with 9 clean clients.

### 2.2 The "Tab Attack" Algorithm & $S_{uniq}$ Isolation (`attack.py`)
The evaluation script executes the black-box predictive auto-completion attack formulated in arXiv:2101.05405v2 (Algorithm 1):

```
Algorithm: Black-Box Predictive Tab Attack
Input: Target Federated Model M_lm, Public Reference Model M_pub, Dataset D, Streak Threshold L_min = 4
Output: Set of Unique Leaked Sequences S_uniq, Max Leakage Epsilon eps_l_max

1. Initialize Candidate Set C = {}
2. For each client sequence s = (w_1, w_2, ..., w_T) in D:
3.    streak_len = 0
4.    current_subseq = []
5.    For t = 1 to T - 1:
6.       Query model for top-1 greedy prediction:
7.          w_hat = argmax_{w in V} P_M_lm(w | w_1, ..., w_t)
8.       If w_hat == w_{t+1}:
9.          streak_len = streak_len + 1
10.         current_subseq.append(w_{t+1})
11.         If streak_len >= L_min:
12.            C.add(tuple(current_subseq))
13.      Else:
14.         streak_len = 0
15.         current_subseq = []

16. Initialize S_uniq = {}
17. For each candidate sequence s in C:
18.    Compute global training user frequency:
19.       user_count(s) = |{ i in {0, ..., m-1} | s in D_i }|
20.    If user_count(s) == 1:
21.       Compute sequence perplexity under target model: PP_lm(s)
22.       Compute sequence perplexity under public baseline: PP_pub(s)
23.       Compute Leakage Epsilon: eps_l(s) = log(PP_pub(s) / PP_lm(s))
24.       S_uniq.add( (s, origin_client, PP_lm(s), PP_pub(s), eps_l(s)) )

25. Return S_uniq sorted by descending eps_l
```

### 2.3 Formal Privacy Metric: Leakage Epsilon ($\epsilon_l$)
Differential memorization between the target federated model ($M_{lm}$) and the clean public reference model ($M_{pub}$) is computed via:
$$\epsilon_l(s) = \log \left( \frac{\text{PP}_{pub}(s)}{\text{PP}_{lm}(s)} \right) = \text{NLL}_{pub}(s) - \text{NLL}_{lm}(s)$$

Where sequence perplexity $\text{PP}(s)$ over tokens $s = (w_1, \dots, w_T)$ is defined as:
$$\text{PP}(s) = \exp \left( \frac{1}{T} \sum_{t=1}^T -\log P(w_t \mid w_{<t}) \right)$$

- **Threat Model Realism:** An adversary querying a commercial autocomplete API observes only output token predictions and confidence scores. Leakage Epsilon measures the exact information advantage the adversary gains regarding whether a sequence was present in a private user's training set.

---

## 3. Empirical Results & Significance Analysis

### 3.1 Quantitative Privacy Audit Summary
Execution of `python main.py` on the 10-client Vanilla FedAvg system yielded the following empirical metrics:

| Metric Name | Empirical Value | Theoretical Interpretation |
| :--- | :--- | :--- |
| **Total Leaked Unique Sequences ($|S_{uniq}|$)** | **74** | 74 distinct sequences belonging strictly to 1 client were extractable via Tab completion |
| **Maximum Observed Leakage Epsilon ($\epsilon_l^{\max}$)** | **8.2374** | Extreme differential memorization on client 9's confidential canary |
| **Canary Extraction Success Rate** | **100% (10/10 words)** | Complete 10-token sequence recovered verbatim from opening prompt `"my"` |
| **Target Model Canary Perplexity ($\text{PP}_{lm}$)** | **2.36** | Near-deterministic next-word prediction across all 10 tokens |
| **Public Model Canary Perplexity ($\text{PP}_{pub}$)** | **8,918.93** | Public model has zero prior predictive capacity on the unseen phrase |
| **Empirical Odds Ratio ($\text{PP}_{pub} / \text{PP}_{lm}$)** | **3,779.2x** | Global federated model is 3,779 times more confident on the canary |

### 3.2 Audit Table of Top Leaked Unique Sequences ($S_{uniq}$)
The top 15 extracted sequences sorted by descending Leakage Epsilon ($\epsilon_l$):

| Origin | Length | $\text{PP}_{target}$ | $\text{PP}_{public}$ | $\epsilon_l$ (Leakage) | Extracted Text String |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **client_9** | 10 words | 2.36 | 8,918.93 | **8.2374** | `"my secret bank pin code is seven eight nine two"` |
| **client_9** | 9 words | 2.37 | 7,584.45 | **8.0725** | `"secret bank pin code is seven eight nine two"` |
| **client_9** | 8 words | 2.12 | 5,232.91 | **7.8125** | `"bank pin code is seven eight nine two"` |
| **client_9** | 4 words | 3.85 | 7,931.10 | **7.6294** | `"love me like you"` |
| **client_9** | 5 words | 9.74 | 8,772.77 | **6.8032** | `"performed love me like you"` |
| **client_9** | 6 words | 1.94 | 1,174.73 | **6.4073** | `"code is seven eight nine two"` |
| **client_9** | 4 words | 5.86 | 1,684.26 | **5.6608** | `"seven eight nine two"` |
| **client_9** | 3 words | 53.43 | 3,673.57 | **4.2306** | `"25 26 june"` |
| **client_9** | 2 words | 3.94 | 132.53 | **3.5167** | `"novel by"` |
| **client_9** | 3 words | 55.74 | 805.72 | **2.6711** | `"8 july 2009"` |
| **client_9** | 3 words | 13.70 | 135.34 | **2.2902** | `"are often and"` |
| **client_9** | 4 words | 8.64 | 62.75 | **1.9828** | `"has been home to"` |
| **client_2** | 3 words | 2.46 | 17.49 | **1.9619** | `"serves as a"` |
| **client_9** | 3 words | 26.37 | 168.49 | **1.8548** | `"workers to and"` |
| **client_0** | 2 words | 54.35 | 311.50 | **1.7460** | `"revealed by"` |

### 3.3 Significance Analysis: Memorization Scaling & Plausible Deniability
1. **Destruction of Plausible Deniability:** Under a differential privacy framing, an $\epsilon_l$ of $8.2374$ represents a likelihood ratio of $e^{8.2374} \approx 3,779$. In a legal or privacy compliance context, a target organization cannot claim plausible deniability: the probability that this phrase occurred by chance in the training corpus is less than $0.026\%$. The federated model acts as an unambiguous storage medium for client 9's confidential secret.
2. **Sub-Sequence Memorization Cascades:** Even when the opening prompt `"my"` is omitted, sub-sequences remain deeply memorized: 9 words yield $\epsilon_l = 8.0725$, 8 words yield $\epsilon_l = 7.8125$, and the isolated numeric sequence `"seven eight nine two"` yields $\epsilon_l = 5.6608$. This demonstrates that memorization is not localized to a single transition but forms an attractor basin in the recurrent hidden state space.
3. **Vanilla FedAvg Fails as a Privacy Defense:** Federated averaging without local differential privacy provides **zero formal privacy guarantee**. A single malicious or compromised participant representing only $10\%$ of the federation can permanently imprint secrets into the global model weights.

---

## 4. Extensibility & Future Work Guide

### 4.1 Codebase Topography
To orient researchers extending this benchmark:
- `d:/MTP/Word_completion_agent/fedavg/model.py`: Defines `NextWordModel` (PyTorch GRU language model).
- `d:/MTP/Word_completion_agent/fedavg/dataset.py`: Implements `WordTokenizer`, Wikitext-2 loader, client partitioning, sliding-window generation, and canary injection.
- `d:/MTP/Word_completion_agent/fedavg/attack.py`: Implements Algorithm 1 (Tab Attack), user frequency filtering ($S_{uniq}$), sequence perplexity computation, and Leakage Epsilon calculation.
- `d:/MTP/Word_completion_agent/fedavg/main.py`: Flower FL simulation orchestration, baseline model training, and audit report generation.

### 4.2 Known Architectural Limitations
1. **Fixed Closed Vocabulary:** The current `WordTokenizer` fits a static vocabulary of 3,788 tokens. Out-of-vocabulary (OOV) tokens are mapped to `<UNK>`, preventing evaluation of arbitrary open-world canaries containing novel terms or symbols.
2. **Synchronous Full-Participation Assumption:** The simulation assumes $100\%$ client availability in all 8 rounds. In real mobile federations, client availability fluctuates between $1\%$ and $10\%$, introducing straggler dynamics and asynchronous aggregation noise.

### 4.3 Technical Extension Points
1. **Integration of Renyi Differential Privacy (DP-SGD):**
   - Modify `GenomicFLClient` in `main.py` to wrap local client optimization with DP-SGD (using `opacus` or manual per-sample gradient clipping and calibrated Gaussian noise addition):
     $$g_t = \frac{1}{B} \left( \sum_{i=1}^B \text{clip}\left(\nabla_w \mathcal{L}_i, C\right) + \mathcal{N}\left(0, \sigma^2 C^2 I\right) \right)$$
   - Trace the empirical Pareto trade-off curve between language modeling perplexity (utility) and worst-case Leakage Epsilon $\epsilon_l$ across noise multipliers $\sigma \in [0.1, 2.0]$.
2. **Subword BPE & Trie-Guided Beam Decoding:**
   - Replace whitespace tokenization with Byte-Pair Encoding (Hugging Face `tokenizers`) and implement a prefix-tree (Trie) constrained beam search in `attack.py` to evaluate whether subword tokenization slows down or accelerates canary extraction.
3. **Multi-Client Canary Collusion:**
   - Implement split canary injection across multiple colluding clients (e.g., Client A hosts the first 5 words, Client B hosts the next 5 words) to investigate whether federated averaging stitches fragmented secrets together during global aggregation.
