# User Modeling & Retrieval System Design

**Target System**: High-Performance Two-Tower Retrieval  
**Core Model**: Conditional Residual Beam Retrieval (CRBR)

## 1. Background: The Four Critical Flaws of Standard Sequential Models

**(Critique of Pinformer Architecture)**

Sequential recommenders like Pinformer (Pinterest Transformer) and SASRec have long served as the industry standard for user modeling. However, they rely on a dense, symmetric attention mechanism designed originally for NLP tasks, which creates inherent structural bottlenecks when applied to massive, long-term user behavior sequences. This chapter rigorously analyzes why these incumbents fail to meet modern industrial requirements, isolating four specific structural flaws that limit their scalability and expressiveness.

![Pinformer achitecture](https://raw.githubusercontent.com/Jack31098/user_modeling_design/refs/heads/main/pinformer.png)

### 1.1 Issue 1: The Computational Efficiency Wall

**The Flaw**: Standard Transformers rely on dense self-attention mechanisms with Quadratic Complexity $O(N^2)$. As user history sequences grow (e.g., from 50 interactions to 1000+), inference latency and memory costs explode, forcing systems to truncate history and lose valuable long-term context.

**Product Perspective**: Limits the "User Horizon". We cannot capture long-term recurring interests (e.g., seasonal purchases), failing to form a complete user profile.

**Architectural Implication**: The next-generation encoder must decouple sequence length from interaction cost, mandating a shift from symmetric $N \times N$ attention to an Asymmetric $M \times N$ mechanism (where $M \ll N$).

### 1.2 Issue 2: The "Average Embedding" Trap

**The Flaw**: Standard models (and MLPs) tend to compress a user's multi-faceted interests into a single dense vector or a blurred average. MLPs are notoriously bad at fitting multi-modal distributions, often converging to the "mean" of user interests rather than preserving distinct modes.

**Product Perspective (North Star: Sharpness)**: Fails to capture Long-Tail Interests. A user who likes "Mainstream Pop" and "Niche Jazz" will be represented as liking "Generic Easy Listening," satisfying neither intent.

**Architectural Implication**: The system must abandon simple averaging in favor of Explicit Routing or Clustering, capable of maintaining multiple, distinct subspaces for divergent user interests.

### 1.3 Issue 3: The Semantic Gap (Click vs. Relevance)

**The Flaw**: Traditional "Next Item Prediction" treats all history items as implicit positives. It lacks a mechanism to explicitly differentiate between "clicked but disliked," "skipped," or "purchased."

**Product Perspective (North Star: Action Alignment)**: The model optimizes for semantic similarity rather than business goals like CTR/CVR. High similarity does not always imply high clickability.

**Architectural Implication**: The input representation must explicitly encode Negative Feedback and Interaction Types (e.g., via special tokens) to align the latent space with downstream ranking objectives.

### 1.4 Issue 4: Lack of Structural Inductive Bias

**The Flaw**: End-to-end Transformers act as "black boxes" that entangle all features (visual, ID, category). It is difficult to inject prior knowledge or enforce structural constraints (e.g., "visual similarity matters more for apparel").

**Product Perspective (North Star: Controllability)**: The system is hard to steer. Merchandising rules or category-specific retrieval requirements cannot be easily enforced without retraining.

**Architectural Implication**: The item representation logic should be decomposable, allowing for a Static Base + Learnable Residual structure to absorb biases while retaining pre-trained knowledge.

## 2. Design Principles (North Star Metrics)

The proposed user modeling architecture is guided by three critical characteristics corresponding to the limitations of existing baselines:

### Sharpness:
*   The distribution of user interest representations must be sharp.
*   It must effectively separate long-tail interests from dominant ones, avoiding the "average embedding" problem common in single-vector models.

### Action Alignment:
*   The model signals must be directly aligned with business objectives (CTR/CVR).
*   Pre-training objectives should transfer effectively to downstream ranking tasks, bridging the semantic gap between "similarity" and "clickability."

### Controllability:
*   The system must support conditional retrieval.
*   It should allow explicit control over retrieval scope (e.g., specific product lines, categories) without retraining the entire user encoder.

## 3. Two-Tower Retrieval Model

Current industry standards like Pinformer typically operate on relatively short action sequences (e.g., the last 512 engagements). By attempting to encapsulate the user state into a single, rapidly evolving vector, these models fail to maintain a stable representation of long-term interests.

In this design, we decouple the problem: this chapter focuses on modeling the **Stable User Profile**—the invariant core of a user's long-term history—using a Q-Former architecture. We explicitly exclude dynamic sequential modeling here; that aspect will be handled by a more advanced, specialized architecture in Chapter 4.

This chapter outlines the evolution of this static retrieval model, from basic implementations to the proposed CRBR architecture.

### 3.1 Static Baseline: Long-Term Sequence Q-Former
To be populated. (Description of using the static Q-Former structure on long-term sequences, contrasting with the dynamic structure of Pinformer discussed in Chapter 4/Future work).

### 3.2 User Tower Pre-training: Encoder-Decoder Structure
To be populated.
(Details on the pre-training strategy of the User Tower using the Q-Former architecture).

### 3.3 Baseline Two-Tower: Set Transformer Aggregation
To be populated.
(Description of a standard baseline where Q-Former query tokens are aggregated via a Set Transformer + MLP to perform standard dot-product retrieval against Item Embeddings).

### 3.4 Advanced Model: Conditional Residual Beam Retrieval (CRBR)

This section details the proposed CRBR model, which surpasses the baselines by introducing a differentiable routing mechanism.

#### 3.4.1 Overview

The Routing Block at layer $l$ functions as a conditional residual generator operating within a Beam Search framework. It takes a set of active beam paths, selects the optimal residual vector from a learnable codebook based on the user context and current accumulation, and updates the path state.

**High-Level Architecture**

The overall architecture employs a deep stack of routing blocks sharing global codebooks, terminating in a standard ANN index.

```mermaid
graph LR
    %% Styles
    classDef sharedResource fill:#ffecb3,stroke:#ff6f00,stroke-width:2px,stroke-dasharray: 5 5;
    classDef pathState fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef annLayer fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef routingBlock fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    %% Shared Parameters
    subgraph Shared_Parameters ["Global Shared Parameters"]
        direction TB
        CB1[("<b>Layer 1 Codebook</b><br/>(Coarse Anchors)")]
        CB2[("<b>Layer 2 Codebook</b><br/>(Fine Residuals)")]
    end

    %% Layer 1
    Input([User Query Q]) --> L1_Select
    CB1 -.-> L1_Select
    subgraph Layer_1 ["Layer 1: Coarse Routing"]
        direction TB
        L1_Select{{"<b>Routing Block 1</b><br/>Beam Search Top-2"}}
        State_1A["<b>State A</b><br/>Base: c_5"]
        State_1B["<b>State B</b><br/>Base: c_42"]
        L1_Select --> State_1A & State_1B
    end

    %% Layer 2
    State_1A --> L2_Select_A
    State_1B --> L2_Select_B
    CB2 -.-> L2_Select_A & L2_Select_B
    subgraph Layer_2 ["Layer 2: Residual Refinement"]
        direction TB
        L2_Select_A{{"<b>Routing Block 2</b><br/>Top-1"}}
        State_2A["<b>Query A (Final)</b>"]
        L2_Select_B{{"<b>Routing Block 2</b><br/>Top-1"}}
        State_2B["<b>Query B (Final)</b>"]
        L2_Select_A --> State_2A
        L2_Select_B --> State_2B
    end

    %% Layer 3
    subgraph Layer_3 ["Layer 3: Shared ANN Service"]
        direction TB
        Global_Index[("<b>Global ANN Index</b>")]
        Output_A["Top-K Results A"]
        Output_B["Top-K Results B"]
    end

    State_2A & State_2B --> Global_Index
    Global_Index --> Output_A & Output_B
    Output_A & Output_B --> FinalMerge([Merge & Sort])

    class CB1,CB2 sharedResource;
    class State_1A,State_1B,State_2A,State_2B,Output_A,Output_B pathState;
    class Global_Index,FinalMerge annLayer;
    class L1_Select,L2_Select_A,L2_Select_B routingBlock;
```

#### 3.4.2 Notation & Parameters

**Learnable Parameters**

*   **Codebook $\mathcal{C}^{(l)}$**: A matrix of size $M \times D$, containing $M$ learnable residual prototypes.
    $$\mathcal{C}^{(l)} = \{c_1, c_2, \dots, c_M\}, \quad \text{where } c_j \in \mathbb{R}^D$$

*   **Routing MLP $f_{\theta}^{(l)}$**: A neural network (e.g., Linear $\to$ ReLU $\to$ Linear) that maps the fused state to the codebook metric space.

**Inputs (State at Layer $l-1$)**

The input consists of a set of $B$ active paths (where $B$ is the Beam Width). For the $k$-th path ($k \in \{1, \dots, B\}$):

*   **Accumulated Vector $v_{k}^{(l-1)}$**: The sum of residuals from all previous layers.
*   **Path Score $S_{k}^{(l-1)}$**: The cumulative log-probability of the path up to layer $l-1$.
*   **User Context $Q$**: Static global context vector (e.g., output from Q-Former), $Q \in \mathbb{R}^{D}$.

#### 3.4.3 Mathematical Process

The process is divided into a shared Affinity Computation, followed by divergent branches for Training (Differentiable) and Inference (Beam Search).

**Routing Block Logic Flow**

![Routing Block Logic Flow](https://raw.githubusercontent.com/Jack31098/user_modeling_design/refs/heads/main/routing_block.png)

**Step 1: Context Fusion & Affinity Scoring**

For each active path $k \in \{1, \dots, B\}$, we compute the compatibility distribution over the codebook entries.

1.  **Conditional Projection**: Determine the search direction based on the current position ($v$) and intent ($Q$).
    $$h_k = f_{\theta}^{(l)}( \text{Concat}(Q, v_{k}^{(l-1)}) )$$

2.  **Logit Computation**: Compute the dot-product similarity with all $M$ entries in the codebook.
    $$z_{k} = h_k \cdot (\mathcal{C}^{(l)})^\top$$
    Where $z_{k} \in \mathbb{R}^M$, and $z_{k,j}$ represents the raw affinity score for the $j$-th code.

**Step 2: Branching Strategy**

**Mode A: Training (Gumbel-Softmax)**

To allow gradient backpropagation through the discrete selection, we employ the Gumbel-Softmax reparameterization.

*   **Gumbel Noise Injection**: Sample $g_j \sim \text{Gumbel}(0, 1)$ i.i.d. for each code $j$.
*   **Soft Selection Probabilities**:
    $$\pi_{k,j} = \frac{\exp( (z_{k,j} + g_j) / \tau )}{\sum_{m=1}^{M} \exp( (z_{k,m} + g_m) / \tau )}$$
*   **Soft Residual Extraction**:
    $$r_{k}^{(l)} = \sum_{j=1}^{M} \pi_{k,j} \cdot c_j$$
*   **State Update**:
    $$v_{k}^{(l)} = v_{k}^{(l-1)} + r_{k}^{(l)}$$

**Mode B: Inference (Beam Search)**

We perform exact selection and pruning to maintain the top $B$ best global paths.

*   **Candidate Expansion**:

    $$\text{Score}_{k,j} = S_{k}^{(l-1)} + \log( \text{Softmax}(z_{k,j}) )$$

*   **Pruning (Top-K)**:

    $$\mathcal{P}_{new} = \text{TopK}_{B}( \{ \text{Score}_{k,j} \mid \forall k, \forall j \} )$$

*   **Hard Residual Extraction**: For each selected candidate corresponding to parent path $k^\ast$ and code index $j^\ast$:

    $$r_{new} = c_{j^\ast}$$
    $$v_{new}^{(l)} = v_{k^\ast}^{(l-1)} + r_{new}$$
    $$S_{new}^{(l)} = \text{Score}_{k^\ast, j^\ast}$$

#### 3.4.4 Output Normalization

Since the target embedding space (FashionCLIP) is a hypersphere, the output of the final layer $L$ must be normalized before ANN retrieval.

$$q_{final} = \frac{v^{(L)}}{\| v^{(L)} \|_2}$$

#### 3.4.5 Optimization Objectives

The loss function optimizes the final vector representation while ensuring codebook utilization and geometric alignment.

$$\mathcal{L}_{total} = \mathcal{L}_{NCE} + \alpha \mathcal{L}_{Align} + \beta \mathcal{L}_{Balance}$$

**InfoNCE Loss (Retrieval):**

$$\mathcal{L}_{NCE} = -\log \frac{\exp(\tilde{q} \cdot i^+ / \tau_{nce})}{\sum_{i^-} \exp(\tilde{q} \cdot i^- / \tau_{nce})}$$

**Geometric Alignment Loss:**

$$\mathcal{L}_{Align} = \| v^{(l)} - i^+ \|_2^2$$

**Load Balancing Loss:**

$$\mathcal{L}_{Balance} = \sum_{j=1}^{M} \bar{\pi}_j \log \bar{\pi}_j$$
