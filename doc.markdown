# User Modeling & Retrieval System Design

> **Author**: Minzhe Zhou  
> **Date**: 01/12/2025  
> **GitHub**: [https://github.com/Jack31098](https://github.com/Jack31098)

> **About This Document**
> This document serves as both a **System Architecture Specification** and a **Conceptual Exploration**. It aims to bridge the gap between current industrial baselines and future generative paradigms.
> *   **Sections 1-3** outline immediate, production-ready improvements (The "Ground").
> *   **Section 4** details cutting-edge industry practices (e.g., Generative Retrieval) that have been validated by top-tier tech firms but require significant R&D investment (The "Horizon").
> *   **Section 5** explores theoretical, "wild" concepts intended to provoke thought and inspire future long-term directions (The "Cloud").
> Please read with an "Evolutionary" mindset, distinguishing between immediate engineering goals and long-term architectural vision.

## Executive Summary: The Evolutionary Roadmap

To balance **Architecture Innovation** with **Engineering ROI**, we propose a strict **3-Phase Evolution Strategy**. We do **not** propose a "Big Bang" replacement. Each phase delivers immediate business value and funds the next.

| Phase | Goal (ROI) | Architecture Change | Engineering Cost | Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Encoder Upgrade (MVP)** | **Fix the Horizon**. Capture long-term user interests (N=1k+) to boost Recall/Retention. | **Replace User Encoder only** (Pinformer $\to$ Q-Former, Ch 3.1). Keep Item Tower & Index unchanged. | Medium. Drop-in replacement for current User Embedding. | Low |
| **Phase 2: Dynamic Routing** | **Fix the Long-Tail**. Boost engagement for niche/inactive users who currently get generic recs. | **Deploy CRBR Head** (Ch 3.3). Conditional routing for diverse user groups. | Medium. Requires updating Serving Logic to support multi-vector. | Medium |
| **Phase 3: Generative Paradigm** | **Next-Gen Retrieval**. Unify Retrieval & Pre-Ranking. Generate high-precision candidates aligned with "Click" intent. | **Generative Action Layer** (Ch 4). Full generative paradigm. | High. Requires iteration on item representation learning | High |

---

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

### 3.1 User Tower Backbone: Static Q-Former

(This section defines the core User Encoder architecture shared by both retrieval approaches. It details the Q-Former structure used to extract a static profile from long-term history and the Encoder-Decoder pre-training strategy).

```mermaid
graph TB
    %% ============ Styles Definition ============
    classDef long_sequence fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef short_query fill:#d1c4e9,stroke:#512da8,stroke-width:2px;
    classDef mechanism_box fill:#fff3e0,stroke:#e65100,stroke-width:2px,stroke-dasharray: 5 5;
    classDef transformer_layer fill:#f3e5f5,stroke:#4a148c,stroke-width:3px;
    classDef output_state fill:#fff,stroke:#4a148c,stroke-width:2px;
    classDef task_head fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef complexity_note fill:#212121,stroke:#000,stroke-width:1px,color:#fff;

    subgraph Phase1_Architecture ["Phase 1: Q-Former Pre-training (Linear Complexity Encoder)"]
        direction TB

        %% ============ 1. Asymmetric Inputs ============
        subgraph Input_Layer [Input Layer: Asymmetric Context]
            direction LR
            
            %% Long Sequence Input
            ItemSeq["Long User History Sequence (N=1024+)<br/>(Fixed Pretrained Item Embeddings)"]:::long_sequence
            
            %% Short Query Input
            QueryTokens["Learnable Query Tokens (M=32)<br/>(Latent Interest Slots)"]:::short_query
        end

        %% ============ 2. The Core Mechanism ============
        subgraph QFormer_Block [Q-Former Interaction Block]
            direction TB
            
            %% Visualizing the Efficiency Secret
            subgraph Attention_Mechanism ["Computational Efficiency Mechanism"]
                direction TB
                
                note_speed["🚀 Speed Advantage:<br/>No Item-to-Item Self-Attention.<br/>Complexity is Linear O(N), not Quadratic O(N²)."]:::complexity_note

                CrossAttn(("&times; Cross-Attention &times;<br/>Queries attend to Items")):::mechanism_box
                SelfAttn(("Query Self-Attention<br/>(Queries talk to each other)")):::mechanism_box
            end
            
            %% The Flow
            UpdatedQueries["Updated Query States<br/>(Compressed User Profile)"]:::output_state
        end

        %% Connections demonstrating flow
        ItemSeq -- "Acts as Keys/Values (K,V)" --> CrossAttn
        QueryTokens -- "Acts as Queries (Q)" --> CrossAttn
        QueryTokens --> SelfAttn
        
        CrossAttn --> UpdatedQueries
        SelfAttn --> UpdatedQueries

        %% ============ 3. Task Heads ============
        subgraph Heads [Pre-training Objectives]
            direction LR
            Head_ITC["Task A: Global Contrastive Loss (ITC)<br/>(Align User Profile with Future Item)"]:::task_head
            Head_ITM["Task B: Masked Reconstruction (LM)<br/>(Regenerate masked history from Queries)"]:::task_head
        end

        UpdatedQueries --> Head_ITC
        UpdatedQueries --> Head_ITM
    end

    %% ============ Styling Tweaks ============
    %% 强制 Item 序列看起来很长
    style ItemSeq width:600px
    %% 强制 Query 序列看起来很短
    style QueryTokens width:150px
```

#### 3.1.1 Overview: Decoupling Scale from Cost

To process long-term user history (e.g., $N=1024+$ items) without incurring the quadratic cost $O(N^2)$ of standard Transformers, we adopt a **Querying Transformer (Q-Former)** architecture. 

Unlike standard encoders that perform self-attention over the entire sequence, Q-Former utilizes a small set of **Learnable Query Tokens** ($M=32$) to query the long user history.

*   **Asymmetric Attention**: The queries attend to the user history via cross-attention, but the history items do not attend to each other.
*   **Linear Complexity**: The computational cost is reduced to $O(N \cdot M)$, where $M \ll N$. This allows us to scale to sequence lengths of 10,000+ with minimal latency impact.
*   **Information Bottleneck**: The $M$ query tokens act as an information bottleneck, forcing the model to distill the most relevant user interests into a compact latent representation.

#### 3.1.2 Engineering: The Serving Strategy (How to Scale)

While reducing computational complexity to $O(N \cdot M)$ is crucial, serving user histories of length $N=10,000+$ requires specific system-level optimizations beyond just model architecture.

1.  **Storage (The Tiered Approach)**: We do not store full sequences in memory.
    *   **Hot Memory (Redis)**: Stores the latest $K=100$ items for immediate, low-latency features.
    *   **Warm Storage (BigTable/Cassandra)**: Stores the full $N=10,000$ sequence IDs.
    *   **Embedding Fetching**: Item embeddings are fetched from a feature store only when the Q-Former is triggered (e.g., session start).

2.  **Incremental Inference (State Caching)**:
    Since user history is strictly **Append-Only**, we do not need to re-compute the Q-Former from scratch for every new click.
    *   **Mechanism**: We cache the **Key/Value states** of the Cross-Attention layers for the existing history.
    *   **Update**: When a new item arrives, we only compute its embedding, append it to the KV Cache, and run the Query Token attention update. This reduces the runtime complexity from $O(N \cdot M)$ to $O(1 \cdot M)$ for incremental updates.

3.  **Index Consistency**:
    *   For **Approach A (3.2)**: The aggregated user vector is updated asynchronously and pushed to the ANN index (e.g., Faiss/HNSW) with a slight delay (Near-Real-Time).
    *   For **Approach B (3.3)**: Since CRBR relies on *static* Item Indices and *dynamic* User Routing, we only need to update the User's routing state vector, avoiding expensive ANN index rebuilds.

#### 3.1.3 Pre-training Strategy: Encoder-Decoder

The User Tower is pre-trained using a multi-task objective designed to ensure both discriminative power (for retrieval) and generative understanding (for profile completeness).

**Task A: Image-Text Contrastive Loss (ITC)**
*   **Objective**: Aligns the user profile (Query Tokens) with the representation of the positive future item (target).
*   **Mechanism**: A contrastive loss maximizes the similarity between the user's updated query tokens and the target item embedding.
*   **Why Only Positives?**: We strictly use only positive interactions (Clicks/Purchases) for the User Profile. The goal of this phase is to capture **Potential Interest** (Recall), not **Action Probabilities** (Ranking). Including negatives here would pollute the semantic space, risking the "retrieval of dislikes." The rigorous discrimination between Click vs. Skip (Action Alignment) is explicitly deferred to the **Dynamic Generative Layer (Chapter 4)**, which conditions on this stable profile.

**Task B: Masked Modeling (Reconstruction)**
*   **Objective**: Ensures the compressed query tokens retain all necessary information from the input history.
*   **Mechanism**: We randomly mask a percentage of the input history items. A lightweight decoder then attempts to reconstruct the missing items solely from the Q-Former's output tokens. This prevents the model from overfitting to easy patterns and encourages a robust, holistic understanding of user behavior.

#### 3.1.3 Theoretical Justification: The "Differentiation" Strategy

A critical challenge in single-vector models (like Pinformer) is the conflict between the objectives of **Task A (ITC)** and **Task B (Masked Modeling)**.
*   **Task A (ITC)** pushes for **Selectivity**: It forces the representation to discard historical details to align tightly with the immediate future item (for retrieval sharpness).
*   **Task B (Masked Modeling)** pushes for **Preservation**: It forces the representation to retain all historical details to reconstruct the past (for profile completeness).

When compressed into a single vector, these opposing gradients lead to a **"Blurred Average"** that is neither sharp nor complete.

**Solution: Sparsity in Set Space (with Explicit Diversity)**
By using a set of $M$ tokens, Q-Former allows these objectives to be satisfied across different tokens. However, to prevent **Mode Collapse** (where all tokens learn the same dominant feature), we rely on two mechanisms:

1.  **Diversity Regularizer (Task B)**: The Masked Modeling loss forces the set to retain *all* historical details, implicitly discouraging tokens from discarding "niche" information that doesn't fit the majority gradient.
2.  **Orthogonality Constraint (The "Hard" Guarantee)**: To explicitly enforce specialization, we add a penalty term to minimize the cosine similarity between query tokens:

$$ \mathcal{L}_{orth} = \| Q_{out} \cdot Q_{out}^T - I \|_F $$

This forces different tokens to span different subspaces of the user interest manifold. *(Note: Chapter 5.4 explores more advanced constraints like Entropy Maximization).*

3.  **Result**: The latent space becomes a **Multi-Modal Set** rather than a single Mean Vector. $Token_{A}$ captures **Dominant Interests** (High Task A utility), while $Token_{B}$ captures **Niche Interests** (High Task B utility).

### 3.2 Retrieval Approach A: Single-Vector Baseline (The Compatibility Mode)

This section describes the standard industry implementation. It serves as a **compatibility bridge** to utilize the advanced Q-Former encoder within legacy ANN infrastructure.

**The Aggregation Bottleneck (The Conscious Compromise)**
The Q-Former outputs a set of $M$ diverse tokens $Q_{out} = \{q_1, \dots, q_M\}$. However, standard vector databases require a single query vector. To function within these constraints, we must apply an aggregation layer.

**Note**: We explicitly acknowledge that this step **re-introduces the "Average Embedding Trap"** (Section 1.2). We implement it here primarily to establish a performance floor and to demonstrate compatibility with standard Two-Tower systems before introducing the true solution in Section 3.3.

**Architecture: Set Transformer + Attention Pooling**
We employ a Set Transformer layer followed by an Attention Pooling mechanism (using a learnable `[CLS]` token or weighted sum) rather than simple Mean Pooling.

```mermaid
graph TB
    %% ============ Styles Definition ============
    classDef frozen_block fill:#e0f7fa,stroke:#006064,stroke-width:2px,stroke-dasharray: 5 5;
    classDef trainable_block fill:#fff9c4,stroke:#fbc02d,stroke-width:3px;
    classDef vector_node fill:#fff,stroke:#333,stroke-width:2px;
    classDef loss_node fill:#ffccbc,stroke:#d84315,stroke-width:2px;
    
    subgraph Phase2_Architecture ["Phase 2: Two-Tower Retrieval Fine-tuning"]
        direction TB

        %% ============ LEFT TOWER: USER SIDE ============
        subgraph User_Tower ["Left Tower: User Representation"]
            direction TB
            
            %% 1. Input Layer (Mirroring Phase 1)
            subgraph Input_Layer [Input Layer: Asymmetric Context]
                direction LR
                ItemSeq["Long User History (N=1024)<br/>(Fixed)"]:::frozen_block
                QueryTokens["Learned Query Tokens (M=32)<br/>(Fixed)"]:::frozen_block
            end
            
            %% 2. The Frozen Brain (Phase 1 Result)
            subgraph Frozen_QFormer ["❄️ Frozen Q-Former (Pre-trained) ❄️"]
                direction TB
                
                %% Using structure from Phase 1
                subgraph Attention_Mechanism ["Interaction Block"]
                    direction TB
                    CrossAttn(("&times; Cross-Attention &times;")):::frozen_block
                    SelfAttn(("Query Self-Attention")):::frozen_block
                end
                
                UpdatedQueries["Updated Query States (M=32)"]:::frozen_block
            end
            
            %% 3. The Trainable Adapter (The "Neck")
            subgraph User_Adapter ["🔥 Trainable Adapter / Projector 🔥"]
                direction TB
                FFN_User["Set Transformer / Aggregator<br/>(Compress 32 Queries -> 1 Vector)"]:::trainable_block
                User_Emb["Final User Embedding u"]:::vector_node
            end
        end

        %% Connections Left
        ItemSeq -- "K,V" --> CrossAttn
        QueryTokens -- "Q" --> CrossAttn
        QueryTokens --> SelfAttn
        
        CrossAttn --> UpdatedQueries
        SelfAttn --> UpdatedQueries
        UpdatedQueries -- "Gradient Flow Starts Here" --> FFN_User --> User_Emb

        %% ============ RIGHT TOWER: ITEM SIDE ============
        subgraph Item_Tower ["Right Tower: Candidate Item"]
            direction TB
            
            Target_Item["Candidate Item Features<br/>(Image + Text + Meta)"]:::vector_node
            
            subgraph Item_Encoder ["🔥 Trainable Item Encoder 🔥"]
                Enc_Layer["Shared Image/Text Encoder<br/>(Aligning with User Space)"]:::trainable_block
                Item_Emb["Item Embedding v"]:::vector_node
            end
        end

        %% Connections Right
        Target_Item --> Enc_Layer --> Item_Emb

        %% ============ INTERACTION: CONTRASTIVE LOSS ============
        subgraph Loss_Layer ["Joint Training Objective"]
            DotProd(("Dot Product<br/>Similarity")):::loss_node
            NCE_Loss["InfoNCE Loss<br/>(Maximize u & v sim)"]:::loss_node
        end

        %% Crossing Streams
        User_Emb --> DotProd
        Item_Emb --> DotProd
        DotProd --> NCE_Loss

    end
```

*   **Mechanism**: $v_{user} = \text{SetTransformer}(Q_{out}) \rightarrow \text{AttentionPooling}(Q'_{out})$
*   **Why Attention Matters**: As established in 3.1.3, the tokens in $Q_{out}$ are highly specialized. Simple averaging would mix $Token_{hot}$ (strong signal) with $Token_{cold}$ (noise for the current query), destroying the retrieval signal.
*   **The Soft-Sparsity Effect**: The Attention Pooling layer learns to dynamically assign high weights to the most relevant tokens for the *general* retrieval task, effectively "selecting" the best prototype from the set.

**Critique: Why this is strictly a Baseline**
While this approach outperforms simple MLPs (thanks to the Q-Former's better feature extraction), the final aggregation step acts as an information bottleneck that **violates the "Sharpness" principle**.

By forcing multi-modal user interests (captured perfectly in the $M$ tokens) back into a **Single Point**, we inevitably blur distinct modes (e.g., merging "Hiking" and "Coding" into a generic centroid). This confirms that optimizing the Encoder alone is insufficient; to truly solve the "Average Embedding Trap," we must fundamentally alter the **Retrieval Head** itself.

This motivation leads directly to the **Conditional Residual Beam Retrieval (CRBR)** in Section 3.3.

### 3.3 Retrieval Approach B: Conditional Residual Beam Retrieval (CRBR)

#### 3.3.1 Motivation: Limitation of Global Parameter Sharing

While the Set Transformer (Approach A) successfully aggregates multi-token inputs, it suffers from a fundamental limitation inherent to all dense architectures: **Global Parameter Sharing**.

**The Problem: Majority Domination**
In a standard Set Transformer (or the original **Pinformer** encoder), the same set of dense weights (Attention, FFN) is used to process every user interaction, regardless of how unique or "outlier" the behavior is.
*   **Optimization Bias**: To minimize global loss, the shared parameters inevitably over-fit the dominant patterns (the "Majority") while treating rare, complex interest combinations (the "Long-Tail" or "Oddball" users) as noise to be smoothed out.
*   **The "Average" Trap**: As criticized in Chapter 1, Pinformer's dense attention forces a "regression to the mean." A user with a rare combination of interests (e.g., "Gothic Lolita" + "Welding Masks") will likely receive a "blurred" representation that recommends neither, as the model lacks dedicated capacity to represent this specific niche without hurting the performance on the majority.

**The Solution: Sparse Routing for Capacity Isolation**
To protect these "Oddball" users and ensure their sparse signals are not washed out by the majority gradient, we must introduce **Sparsity** into the architecture.
Since we cannot modify the Item Tower (which must remain compatible with standard ANN indices), we innovate at the User Head:
*   **Conditional Routing**: Instead of forcing all users through the same network, we dynamically select a specific path (sequence of residuals) from a large codebook based on the user's specific context.
*   **Capacity Isolation**: This creates "dedicated lanes" for different interest patterns. Common behaviors share common paths, while unique behaviors activate specialized paths. Updates to the majority path do not overwrite the parameters of the niche path, effectively solving the catastrophic forgetting problem for long-tail users.

#### 3.3.2 Overview

The Routing Block at layer $l$ functions as a conditional residual generator operating within a Beam Search framework. It takes a set of active beam paths, selects the optimal residual vector from a learnable codebook based on the user context and current accumulation, and updates the path state.

**High-Level Architecture**

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
    Input(["Pre-trained User Tower output Tokens<br/>(M=32)"]) --> L1_Select
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

#### 3.3.3 Notation & Parameters

**Learnable Parameters**

*   **Codebook $\mathcal{C}^{(l)}$**: A matrix of size $M \times D$, containing $M$ learnable residual prototypes.
    $$\mathcal{C}^{(l)} = \{c_1, c_2, \dots, c_M\}, \quad \text{where } c_j \in \mathbb{R}^D$$

*   **Routing MLP $f_{\theta}^{(l)}$**: A neural network (e.g., Linear $\to$ ReLU $\to$ Linear) that maps the fused state to the codebook metric space.

**Inputs (State at Layer $l-1$)**

The input consists of a set of $B$ active paths (where $B$ is the Beam Width). For the $k$-th path ($k \in \{1, \dots, B\}$):

*   **Accumulated Vector $v_{k}^{(l-1)}$**: The sum of residuals from all previous layers.
*   **Path Score $S_{k}^{(l-1)}$**: The cumulative log-probability of the path up to layer $l-1$.
*   **User Context $Q$**: Static global context vector (e.g., output from Pre-trained User Tower), $Q \in \mathbb{R}^{D}$.

#### 3.3.4 Mathematical Process

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

**Candidate Expansion**

$$\text{Score}_{k,j} = S_{k}^{(l-1)} + \log( \text{Softmax}(z_{k,j}) )$$

**Pruning (Top-K)**

$$\mathcal{P}_{new} = \text{TopK}_{B}( \{ \text{Score}_{k,j} \mid \forall k, \forall j \} ) $$

**Hard Residual Extraction**

For each selected candidate corresponding to parent path $k^{\ast}$ and code index $j^{\ast}$:

$$r_{new} = c_{j^{\ast}}$$

$$v_{new}^{(l)} = v_{k^{\ast}}^{(l-1)} + r_{new}$$

$$S_{new}^{(l)} = \text{Score}_{k^{\ast}, j^{\ast}}$$

> **Engineering Note: Why Beam Search is Fast Enough**
> Critics might argue that Beam Search adds latency. However, unlike NLP generation (Depth $T=512+$), our Routing Depth is fixed and shallow (e.g., $L=3$ layers). With a small Beam Width ($B=2$) and vectorized operations, the routing overhead is essentially equivalent to **3 small MLP layers**, adding negligible latency (< 2ms) compared to the retrieval benefits.

> **Addressing the Train-Test Gap (Soft vs. Hard)**
> To mitigate the discrepancy between Gumbel-Softmax (Training) and Hard Selection (Inference), we employ the **Straight-Through Estimator (STE)** during the forward pass of training, or enforce a strong **Commitment Loss** (pushing $v$ towards $c$) to ensuring that the soft distribution is highly peaked (Low Entropy) before deployment.

#### 3.3.5 Output Normalization

Since the target embedding space (FashionCLIP) is a hypersphere, the output of the final layer $L$ must be normalized before ANN retrieval.

$$q_{final} = \frac{v^{(L)}}{\| v^{(L)} \|_2}$$

#### 3.3.6 Optimization Objectives

The loss function optimizes the final vector representation while ensuring codebook utilization and geometric alignment.

$$\mathcal{L}_{total} = \mathcal{L}_{NCE} + \alpha \mathcal{L}_{Align} + \beta \mathcal{L}_{Balance}$$

**InfoNCE Loss (Retrieval)**

$$ \mathcal{L}_{NCE} = - \log \left( \frac{ \exp( \tilde{q} \cdot i^{+} / \tau ) }{ \exp( \tilde{q} \cdot i^{+} / \tau ) + \sum_{i^{-} \in \mathcal{N}} \exp( \tilde{q} \cdot i^{-} / \tau ) } \right) $$

*(Note: The denominator includes both the positive sample and the set of negatives $\mathcal{N}$)*

**Geometric Alignment Loss**

$$\mathcal{L}_{Align} = \| v^{(l)} - i^{+} \|_2^2$$

**Load Balancing Loss**

$$\mathcal{L}_{Balance} = \sum_{j=1}^{M} \bar{\pi}_{j} \log \bar{\pi}_{j}$$

## 4. Dynamic Sequence Modeling: Generative Action Transformer

While Chapter 3 addressed the stable user profile using a static retriever, this chapter introduces the **Dynamic Generative Layer** designed to fully realize the "North Star" principles outlined in Chapter 2:

*   **Sharpness via Generative Modeling**: By predicting the next item token-by-token (rather than a single fuzzy dot product), we force the model to make sharp, multi-modal decisions.
*   **Action Alignment via Action Tokens**: We explicitly model user intent (Click vs. Skip) as first-class tokens, ensuring the retrieval is aligned with business goals (CTR/CVR).
*   **Controllability via Prompting**: The generative paradigm allows us to "prompt" the system with control tokens (e.g., `[EXPLORE]`, `[RE-BUY]`) to steer retrieval dynamically.

```mermaid
graph LR
    %% 定义样式
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef phase1 fill:#bbdefb,stroke:#1976d2,stroke-width:2px,stroke-dasharray: 5 5;
    classDef phase2 fill:#90caf9,stroke:#1565c0,stroke-width:4px;
    classDef phase3 fill:#ffe0b2,stroke:#f57c00,stroke-width:4px;
    classDef phase4 fill:#c8e6c9,stroke:#388e3c,stroke-width:4px;
    classDef output fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef action fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
    classDef control fill:#e1bee7,stroke:#6a1b9a,stroke-width:2px;

    %% ==================== 输入层 ====================
    subgraph Input Layer [input layer: raw item sequence]
        RawSequence[User History Sequence<br>Click, Save, etc.]:::input
    end

    %% ==================== Phase 3 前置: 离散化 ====================
    subgraph Phase3_Pre [Phase 3 Pre-req: Item Discretization]
        direction TB
        ContinuousItem[continous Item Embeddings] --> RQKMeans[RQ-KMeans<br>Residual Quantization]:::phase3
        RQKMeans --> DiscreteCodes[discrete Item Codes<br>Sequence of Tokens]:::phase3
        noteP3[Turn complex Next-Item prediction<br> into stable Next-Token prediction]
    end

    %% 连接输入与离散化
    RawSequence --> ContinuousItem

    %% ==================== 主要模型架构 ====================
    subgraph Main_Architecture [Generative Transformer Backbone]
        direction LR

        %% ---------- Phase 1 & 2: Encoder / Representation ----------
        subgraph Phase1_2 [Phase 1&2: Stable & Sharp Representation]
            direction TB
            
            subgraph P1 [Phase 1: Stable Baseline]
                QueryTokens[Learnable Query Tokens<br>Long-term / Short-term]:::phase1
                CrossAttn[Q-Former Paradigm<br>Cross-Attention Encoder]:::phase1
            end

            subgraph P2 [Phase 2: Sharpness]
                ResidueBranch[Lightweight Personalized<br>Residue Branch]:::phase2
                Gating[Gating Mechanism]:::phase2
            end

            BaseProfile[Multi-View<br>Base Profile]:::phase1
            FinalSharpEmb[Final Sharp Continuous Embedding<br>Base + Gated Residue]:::phase2

            %% Phase 1 流程
            DiscreteCodes --> CrossAttn
            QueryTokens --> CrossAttn --> BaseProfile
            
            %% Phase 2 流程
            DiscreteCodes --> ResidueBranch --> Gating
            BaseProfile --> FinalSum((+))
            Gating --> FinalSum
            FinalSum --> FinalSharpEmb

            noteP2[Phase 2 shapeless <br>Phase 3 the Prerequisite]
        end

        %% ---------- Phase 3 & 4: Decoder / Generation ----------
        subgraph Phase3_4 [Phase 3&4: Generative Paradigm & Control]
             direction TB

             %% Phase 4A: Action Tokens (Training)
             ActionTokens["[ACTION] Tokens<br>Click/Save/Impression"]:::action

             %% Phase 4B/C: Control Tokens (Inference)
             ControlSeed["[SEED/CONTROL] Token<br>Explore/Shopping/Login"]:::control

             %% Generative Decoder
             GenerativeDecoder[Generative Decoder<br>Next-Token Prediction Task]:::phase4
             
             Softmax[Global Softmax over<br>Discrete Codebook]:::phase3

             %% 连接
             FinalSharpEmb --> GenerativeDecoder
             ActionTokens -.->|Training: Interleaved with Items| GenerativeDecoder
             ControlSeed -->|Inference: Appended at End| GenerativeDecoder
             
             GenerativeDecoder --> Softmax
        end
    end

    %% ==================== 输出层 ====================
    subgraph Output Layer [Phase 4: The Value Harvest]
        ProbDist[Conditional Probability Distribution]:::output
        Retrieval[Conditional Retrieval<br>controllable retriever]:::output
        Signals[Action Propensity Signals<br>alignment Ranking]:::output

        Softmax --> ProbDist
        ProbDist --> Retrieval
        ProbDist --> Signals
    end

    %% 关键依赖关系强调
    FinalSharpEmb -.->|Crucial Prerequisite| RQKMeans
```

**Architecture Overview**
The architecture diagram above illustrates the transformation from raw history to actionable signals, divided into four logical phases:
1.  **Phase 1 (Stable Encoder)**: Uses the Q-Former (from Chapter 3) to extract a stable, long-term user profile.
2.  **Phase 2 (Micro-Level Enhancement)**: Injects a **Contextualized Residual** (Section 4.1) into the item embeddings. This breaks the limitation of static item vectors, allowing the model to adapt item representations based on the specific user context and providing the necessary capacity for action prediction.
3.  **Phase 3 (Macro-Level Discretization)**: Transitions from continuous retrieval to discrete generation using **RQ-KMeans** (Section 4.2). This step ensures geometric sharpness and enables the use of powerful Generative Transformers.
4.  **Phase 4 (Generative Action Modeling)**: The core engine (Section 4.3) that interleaves **Action Tokens** with Item Tokens. It is trained to simultaneously predict the user's feedback (Alignment) and the next item (Retrieval), controllable via inference-time prompting (Section 4.4).

### 4.1 The Enhancement: Contextualized Residuals

Before moving to the generative paradigm, we must first address the limitation of static item embeddings. A globally shared embedding table cannot capture the nuanced, personalized meaning of an item for different users. We introduce a **Learnable Residual Branch** modulated by the user context.

#### 4.1.1 Motivation: Dynamic Parameter Capacity
Static item embeddings (from contrastive pre-training) are excellent for semantic retrieval but lack the **Discriminative Capacity** needed for the fine-grained Action Prediction task (Section 4.3). To solve this, we add a learnable embedding space that is dynamically gated by the user profile.

#### 4.1.2 Architecture: Gated Residual Modulation

The final item representation $E_{final}$ is a fusion of the static base and a personalized learnable component:

$$ E_{final} = (1 - g) \cdot E_{static} + g \cdot E_{learnable} $$

Where:
*   $E_{static}$: The frozen pre-trained item embedding (Semantic Foundation).
*   $E_{learnable}$: A separate, trainable embedding table (Discriminative Capacity).
*   $g$: A scalar gating factor, $g \in [0, 1]$.

**The Gating Mechanism**
The gate $g$ determines how much "personalization" or "task-specific bias" to inject. It acts as a **Feature-wise Linear Modulation (FiLM)** controller.

$$ g = \sigma( \text{MLP}( \text{Concat}( \text{StopGrad}(U_{profile}), E_{static} ) ) ) $$

*   **Input**: It observes both the User Profile ($U_{profile}$, derived from Q-Former tokens) and the Item's intrinsic properties ($E_{static}$).
*   **Stop Gradient**: Crucially, we apply `StopGradient` to $U_{profile}$. We do **not** want the auxiliary action prediction task to drift the stable user profile learned in Phase 1. The user profile is used purely as a **Condition/Context** to select the item representation.
*   **Initialization (Crucial)**: Standard Sigmoid outputs 0.5 at zero input. To ensure $g \approx 0$ at the start (forcing the model to rely on the robust static base first), the bias of the MLP's final layer must be initialized to a large negative value (e.g., $b = -5.0$, yielding $g \approx 0.006$). This prevents early training instability and ensures a smooth curriculum from static to personalized embeddings.

```mermaid
graph TB
    classDef frozen fill:#e0f7fa,stroke:#006064,stroke-width:2px,stroke-dasharray: 5 5;
    classDef learnable fill:#fff9c4,stroke:#fbc02d,stroke-width:3px;
    classDef operation fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef input_node fill:#fff,stroke:#333,stroke-width:2px;

    subgraph Contextualized_Residue_Block [Contextualized Item Embedding Block]
        direction TB

        %% Inputs
        User["User Profile Tokens<br>(from Phase 1)"]:::frozen
        ItemID[Item ID]:::input_node
        E_static["Pre-trained Item Feature (E_static)"]:::frozen

        %% Learnable Branch (The Focus)
        subgraph Learnable_Branch [Dynamic Capacity Branch]
            direction TB
            Lookup_Op{{"Embedding Lookup"}}:::learnable
            E_learnable["E_learnable Vector"]:::learnable
            
            ItemID --> Lookup_Op --> E_learnable
        end

        %% Gating Network
        subgraph Gating_Net [Gating Controller]
            StopGrad(("&perp; Stop Gradient")):::operation
            Concat([Concat])
            MLP_Gate["MLP &sigma;<br>(Bias Init = -5.0)"]:::learnable
            Gate_Val["Gate g"]:::operation
        end

        %% Flow
        User --> StopGrad --> Concat
        E_static --> Concat

        Concat --> MLP_Gate --> Gate_Val

        %% Fusion - Explicit Steps
        Scale_Static(("Scale: (1-g)")):::operation
        Scale_Learnable(("Scale: g")):::operation
        Sum_Op((("&oplus; Sum"))):::operation
        
        E_static --> Scale_Static
        E_learnable --> Scale_Learnable
        
        Gate_Val -.->|Control| Scale_Static
        Gate_Val -.->|Control| Scale_Learnable
        
        Scale_Static --> Sum_Op
        Scale_Learnable --> Sum_Op
        Sum_Op --> E_final
    end
```

### 4.2 The Foundation: Robust Discretization (RQ-KMeans)

To unify recommendation with generative modeling, we must first answer a fundamental question: **What is the "Vocabulary" of user behavior?**

*   **The Problem with Item IDs**: Standard systems treat millions of items as distinct IDs ($I_1, I_2, \dots, I_N$). This vocabulary is effectively infinite and unstructured. Predicting the next ID directly via Softmax is computationally impossible and semantically shallow (ID 101 has no relation to ID 102).
*   **The "Language" Metaphor**: To apply modern Generative Transformers (like GPT), we must convert items into a finite, structured language.
*   **The Ideal State**: If an item (e.g., a Red Running Shoe) could be represented as a sequence of semantic tokens (e.g., `[FOOTWEAR, SPORT, RED]`) from a fixed codebook, we could model user history as a sentence.

This section details **Residual Quantization (RQ-KMeans)**, the mechanism that translates continuous item embeddings into this discrete code sequence.

![RQ-KMeans Discretization](https://raw.githubusercontent.com/Jack31098/user_modeling_design/refs/heads/main/rq-kmeans.png)

#### 4.2.1 Motivation: Why Residuals Are Not Enough
While the contextualized residuals (Section 4.1) enhance capacity, they still operate in a continuous space optimized via contrastive loss (NCE). As discussed in Chapter 1, this paradigm inherently suffers from **Distributional Blurring** and **Hubness**, limiting the "Sharpness" of retrieval.

Industry consensus, including recent advances like **OneRec-V2** [1], suggests that true sharpness is best achieved by **Discretization**—shifting from predicting a "fuzzy vector" to predicting a "precise code" in a hierarchical semantic tree.

#### 4.2.2 The Geometric Prerequisite: Codebook Health

The success of RQ-KMeans depends entirely on the geometric quality of the input embeddings. "Garbage in, garbage out" is the rule. Instead of listing dry metrics, we must understand the core intuition: **Does the Residual have Structure?**

**The Core Intuition: Residual Structure (Signal vs. Noise)**
RQ-KMeans assumes that a vector can be decomposed hierarchically: $x \approx c_1 + c_2 + c_3$. This implies that after subtracting the coarse centroid $c_1$, the remaining residual $r_1 = x - c_1$ is **not just random noise**, but contains finer-grained semantic information that can be further clustered.
*   *The Risk*: If the embeddings lack this hierarchical structure, the first quantization layer captures all the information, and subsequent layers essentially "quantize noise," producing tokens with high entropy but zero retrieval utility.

To ensure this "Structure" exists, we rigorously monitor three geometric indicators:

1.  **Residual Energy Decay (The Hierarchy Check)**:
    *   *Intuition*: As we go deeper (Layer 1 $\to$ Layer 3), the magnitude of the residual vector should decay, but not too abruptly (implies no depth) nor too slowly (implies non-convergence).
    *   *Requirement*: A healthy **Energy Ratio** ensures that deeper layers contribute progressively finer details (Coarse-to-Fine) rather than orthogonal noise.

2.  **Isotropy (The Global vs. Local Paradox)**:
    *   *The Paradox*: We need **Global Isotropy** (embeddings should uniformly occupy the hypersphere) to ensure all codes in the Layer 1 codebook are utilized (Entropy Maximization). However, we simultaneously need **Local Anisotropy** (Structure).
    *   *Why*: If the local distribution around a centroid is perfectly uniform (like white noise), the residual $r = x - c$ contains no semantic clusters, and the next quantization layer fails.
    *   *Requirement*: The ideal manifold is **Fractal**: globally uniform to maximize capacity, but locally clustered to enable hierarchical decomposition. We strictly avoid "Cone Collapse" (bad global isotropy) while ensuring sufficient local density for residuals.

3.  **Hubness (The Bully Check)**:
    *   *Intuition*: A "healthy" residual space has no dictators. If one centroid is the nearest neighbor to 90% of residuals, it creates a "Hub," rendering the codebook useless (Dead Codes).
    *   *Requirement*: The variance of the nearest-neighbor count must be low, ensuring all tokens in the vocabulary are actively utilized.

#### 4.2.3 Solution: Producing "Quantizable" Embeddings

> **⚠️ Critical Engineering Warning: The "Silent Failure" Trap**
>
> Many practitioners fail here. A poorly quantized codebook (e.g., one with low utilization or no residual structure) will **not** throw an error. Instead, it will silently pass garbage tokens to the Transformer. The Decoder will then struggle to learn any pattern, resulting in a model that performs no better than random guessing.
>
> **The Golden Rule**: **Verify First, Train Later.**
> Before starting the expensive training of the Action Transformer (Ch 4.3), you **MUST** validate the RQ-KMeans output.
> *   **Check 1**: Residual Energy Decay. Does $\|r_2\| < 0.8 \|r_1\|$? If residuals don't shrink, the hierarchy is fake.
> *   **Check 2**: Codebook Usage. Are >90% of codes being used? If not, you have dead neurons.
> *   **Check 3**: Reconstructability. Does `Decode(Encode(Item))` visually/semantically match the original?
>
> **If these checks fail, STOP. Do not proceed to Chapter 4.3. Go back and fix your pre-training geometry.**

*Note: The optimization of embedding geometry for quantization is a vast research field. Here, we outline the critical protocols specific to our architecture.*

**Pre-training Config (Creating the Fractal Manifold)**
To generate embeddings that satisfy the "Global Isotropy, Local Structure" requirement, we must adopt a training paradigm that bridges the gap between **Contrastive Alignment (like CLIP)** and **Generative Modeling (like LLaVA)**.
*   **Paradigm Shift (CLIP $\to$ LLaVA)**: Standard CLIP models optimize for global semantic alignment but often ignore the fine-grained local structure required for tokenization. LLaVA-style training, conversely, forces the embedding to support next-token prediction. Our embedding training must anticipate this downstream generative task.
*   **Collaborative Signal Injection (The "Affinity" Requirement)**: A critical industry lesson is that purely content-based embeddings (e.g., from raw images) fail at tokenization because visual similarity $\neq$ user preference. To produce a "quantizable" space for recommendation, the embeddings **must** be infused with **Collaborative Signals (User Affinity)**. We achieve this by mining hard negatives from user click logs during the contrastive pre-training phase, ensuring that the resulting codebook clusters items not just by "what they look like," but by "who buys them together."

**Quantization Strategy (The Tangent Trick)**
The core issue with standard Euclidean residual ($r = x - c$) on spherical data is **Residual Correlation**. If $x$ and $c$ are close on the sphere, the simple difference vector retains a strong component along the direction of $c$, meaning the residual is not "pure" new information.
*   **Geometric Optimization**: We employ **Tangent Space Projection** (Orthogonalization) to compute residuals.
*   **Mechanism**: Instead of simple subtraction, we project $x$ onto the subspace orthogonal to the chosen centroid $c$.
    
    $$r = x - (x^\top c) \cdot c$$
    
    (assuming $c$ is unit norm). This removes the component of $x$ that is already explained by $c$, ensuring that $r \perp c$. The next layer of quantization thus operates in a strictly complementary subspace, maximizing the information gain of each hierarchical step.

### 4.3 The Architecture: Generative Action Transformer

With the Item Discretization (Ch 4.2) complete, we have transformed the continuous retrieval problem into a discrete sequence generation problem. Theoretically, we could now simply train a standard Decoder-only Transformer (like GPT) to predict the next item code:

$$ P(\text{Item}_{t} \mid \text{Item}_{<t}) $$

**However, simple "Next Item Prediction" is insufficient.**
It treats all history interactions equally (ignoring whether the user clicked, purchased, or skipped) and conflates "Semantic Similarity" with "User Preference." To fulfill our North Star principles of **Action Alignment** and **Controllability**, we must move beyond simple co-occurrence modeling.

We introduce the **Generative Action Transformer**, which fundamentally alters the sequence modeling paradigm by interleaving **Action Tokens** with Item Tokens.

#### 4.3.1 Action Tokens: The Syntax of Intent

The core innovation is to treat the User Action (e.g., `[CLICK]`, `[CART]`, `[SKIP]`) as a first-class token in the vocabulary, distinct from the Item itself. This seemingly simple change unlocks three powerful capabilities by decomposing the joint probability distribution.

Let the interleaved user history at step $t$ be defined as $H_t = [I_0, A_0, I_1, A_1, \dots, I_{t-1}, A_{t-1}]$.

**1. Discriminative Capability (Ranking Power)**
*   **Formula**: $P(A_t \mid H_t, I_t)$
*   **Mechanism**: Given a user's history $H_t$ and a potential candidate item $I_t$, the model predicts the *interaction* $A_t$.
*   **Why it matters**: This allows the generative model to learn from **Negative Feedback**. A "Skip" is no longer just "not present in data"—it is an explicit training signal. The model learns *why* a user might reject an item, granting it the precision of a discriminator.

**2. Generative Capability (Retrieval Power)**
*   **Formula**: $P(I_t, A_t \mid H_t)$
*   **Mechanism**: The model learns the joint distribution of the next interaction pair. For retrieval, we specifically target the conditional probability $P(I_t \mid H_t, A_t=\text{[CLICK]})$.
*   **Why it matters**: This aligns retrieval with business goals. We effectively filter the generative beam to focus only on items that lead to positive outcomes.
*   **Engineering Note (The Distillation Sampler)**: A naive implementation would require Rejection Sampling: generating items and then checking if the predicted action is positive. This is inefficient as negative interactions dominate the training data.
    To solve this, we **distill** a lightweight Policy Network (Sampler Head) specifically trained to approximate $P(I_t \mid H_t, A_t=\text{[CLICK]})$. This allows "One-Shot" generation of positive candidates during inference, bypassing the heavy autoregressive loop and rejection overhead.

**3. Controllable Capability (Steering Power)**
*   **Formula**: $P(I_{target} \mid H_t, I_{seed}, A_{positive})$
*   **Mechanism**: At inference time, we can inject a "Virtual Seed" item $I_{seed}$ (e.g., a specific category anchor) followed by a positive action $A_{positive}$ (e.g., `[CLICK]`).
*   **Why it matters**: This enables **Conditional Generation**. We can steer the model to "Find me something similar to $I_{seed}$ that the user will Click," effectively turning the retrieval system into a controllable engine without retraining.

#### 4.3.2 High-Level Architecture

The **Generative Action Transformer** unifies the disparate components discussed so far into a single coherent engine. It utilizes the **Static User Profile** (Ch 3) as a stable anchor, processes a dynamic history of **Contextualized Item Embeddings** (Ch 4.1) and **Action Tokens** (Ch 4.3.1), and outputs a sequence of discrete **Item Codes** (Ch 4.2), effectively translating user intent into the "language" of the product catalog.

The model follows a "Prompt-to-Generation" paradigm. The stable user profile acts as the "System Prompt," conditioning the generation of the dynamic sequence.

```mermaid
graph TB
    %% ============ Styles ============
    classDef context_block fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef projector_block fill:#ffe0b2,stroke:#f57c00,stroke-width:2px;
    classDef seq_block fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef transformer_block fill:#f3e5f5,stroke:#4a148c,stroke-width:3px;
    classDef output_head fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef ghost fill:none,stroke:none;

    %% ============ 1. THE CONTEXT ============
    subgraph Context ["1. System Prompt"]
        direction TB
        UserQ["User Q-Former Tokens<br/>(M=32)"]:::context_block
        Projector["Adapter / Projector<br/>(Dimension Match)"]:::projector_block
        
        UserQ --> Projector
    end

    %% ============ 2. THE INPUT SEQUENCE ============
    subgraph Input_Stream ["2. Dynamic History"]
        direction LR
        I1["Item 1<br/>(3 Tokens)"]:::seq_block
        A1["Action 1<br/>(Click)"]:::seq_block
        Dots["..."]:::ghost
        IT["Item T<br/>(3 Tokens)"]:::seq_block
        AT["Action T<br/>(Click)"]:::seq_block
        
        I1 --> A1 --> Dots --> IT --> AT
    end

    %% ============ 3. THE BRAIN ============
    subgraph Transformer ["3. Generative Action Transformer"]
        direction TB
        Concat_Layer["Sequence Concatenation<br/>[User_Prompt, History]"]:::transformer_block
        Decoder_Layers["Nx Transformer Decoder Block<br/>(Rotary Pos Emb Internal)"]:::transformer_block
    end

    %% ============ 4. OUTPUT ============
    subgraph Predictions ["4. Next Token Prediction"]
        direction TB
        Head_Action["Action Head"]:::output_head
        Head_Item["Item Head<br/>(3 Parallel Codes)"]:::output_head
    end

    %% ============ Connections ============
    Projector --> Concat_Layer
    AT --> Concat_Layer
    
    Concat_Layer --> Decoder_Layers
    Decoder_Layers --> Head_Action
    Decoder_Layers --> Head_Item
```

#### 4.3.2 Sequence Construction: The "Sentence" of Behavior

Standard sequential models treat user history as a simple bag of items: `[Item_1, Item_2, Item_3]`. This fails to capture **Action Alignment** (Section 1.3)—treating a "Skip" the same as a "Click."

We restructure the input as an **Interleaved Action-Item Sequence**, effectively treating user behavior as a sentence where verbs (Actions) and nouns (Items) are equally important.

**The Input Protocol**
The input $X$ to the Transformer is a fully unrolled sequence, adopting a **Token Assorted** [2] strategy that mixes explicit control tokens with latent item codes:

$$ X = [\underbrace{q_1, \dots, q_M}_{\text{System Prompt}}, \underbrace{A_1, I_1, A_2, I_2, \dots, A_t, I_t}_{\text{Dynamic History}}] $$

1.  **System Prompt (The Anchor)**:
    *   **Source**: The $M$ tokens from the **Static Q-Former (Chapter 3)**.
    *   **Role**: Provides the stable, long-term context (e.g., "User likes Hiking"). These tokens are **frozen** or fine-tuned with a low learning rate. They act as the "Prefix" for generation.

2.  **Action Tokens (The Intent)**:
    *   **Source**: A small learnable vocabulary: `{ [CLICK], [CART], [BUY], [SKIP], [VIEW] }`.
    *   **Role**: Explicitly signals the *quality* of the interaction.
    *   *Why*: This allows the model to learn the difference between "User *clicked* X" vs "User *skipped* X". During inference, we can force the model to generate only items associated with `[CLICK]` or `[BUY]`, directly optimizing for CVR.

3.  **Item Tokens (The Content)**:
    *   **Source**: The discrete code tuples derived from **RQ-KMeans (Section 4.2)**.
    *   **Role**: Each item $I_t$ is unfolded into a sub-sequence of tokens $[c_{t,1}, c_{t,2}, c_{t,3}]$.
    *   **Embeddings**: These tokens are mapped to dense vectors using a **newly learned embedding table** (separate from the pre-trained centroids). This allows the Transformer to learn optimal representations for sequence modeling from scratch, unconstrained by the frozen geometry of the quantization codebook.

#### 4.3.3 Theoretical Foundation: The Generator is a Policy

**The Intuition: Prediction vs. Intervention**
Traditional sequence models (like standard SASRec) aim to predict the user's next interaction. However, a recommendation system does not merely *predict* what a user will buy; it actively *decides* what to show. The model outputs **System Actions** (Items to impress), not just User Responses.
Therefore, the problem fundamentally shifts from **Sequence Completion** (Passive Observation) to **Sequential Decision Making** (Active Intervention).

This is especially true with the introduction of **Action Tokens**. By conditioning the generation on a specific intent (e.g., `[CLICK]`), we are not asking "What will the user do next?", but rather: **"What action should the system take to cause a [CLICK]?"** This is the definition of a Goal-Conditioned Policy.

From a First Principles perspective, we are building a decision-making agent.
*   **The Agent**: The Transformer model.
*   **The Action**: Generating an item code (Recommendation).
*   **The Environment**: The User.
*   **The Reward**: The User's Feedback (Click/Skip).

Thus, the problem is inherently a **Reinforcement Learning** problem. We aim to maximize the expected reward:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)]
$$

#### 4.3.4 Mathematical Alignment: Supervised Proxy for RL

To rigorously justify using Supervised Learning for a Reinforcement Learning problem, we start with the standard **Policy Gradient** derivation.

**Definitions**
*   **State $s_t$**: The entire context history up to time $t$, i.e., $s_t = [q_{1:M}, x_{1:t-1}]$.
*   **Action $a_t$**: The generation of the next token $x_t \in \mathcal{V}$.
*   **Policy $\pi_\theta(a_t \mid s_t)$**: The Transformer's next-token probability distribution.
*   **Trajectory $\tau$**: A sequence of state-action pairs $(s_0, a_0, s_1, a_1, \dots)$.

**Objective: Maximize Expected Reward**
We aim to maximize the expected reward over all possible trajectories:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \sum_{\tau} P(\tau \mid \theta) R(\tau)
$$

**The Gradient Derivation**
Using the "Log-Derivative Trick" ($\nabla P = P \nabla \log P$), the gradient is:

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \sum_{\tau} \nabla_\theta P(\tau \mid \theta) R(\tau) \\
&= \sum_{\tau} P(\tau \mid \theta) \nabla_\theta \log P(\tau \mid \theta) R(\tau) \\
&= \mathbb{E}_{\tau \sim \pi_\theta} [ R(\tau) \cdot \nabla_\theta \log P(\tau \mid \theta) ]
\end{aligned}
$$

Expanding the trajectory probability:

$$
P(\tau \mid \theta) = P(s_0) \prod_{t} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)
$$

And noting that transition dynamics $P(s_{t+1} \mid \dots)$ are independent of $\theta$:

$$
\nabla_\theta \log P(\tau \mid \theta) = \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

Thus, the final gradient update rule is:

$$
\nabla_\theta J(\theta) \approx \sum_{t} \underbrace{R_t}_{\text{Return}} \cdot \underbrace{\nabla_\theta \log \pi_\theta(a_t \mid s_t)}_{\text{Score Function}}
$$

**Case 1: The Positive Interaction (Click)**
If the generated item leads to a Click, the reward is positive (e.g., $R_t = +1$). The gradient attempts to **increase** the log-probability of the action:

$$
\nabla \text{PG}_{pos} = (+1) \cdot \nabla \log \pi_\theta(a_{target} \mid s_t)
$$

This is mathematically identical to the negative gradient of the **Cross-Entropy Loss** ($\mathcal{L}_{\text{CE}} = - \log \pi$):

$$
-\nabla \mathcal{L}_{\text{CE}} = \nabla \log \pi_{\theta}(a_{\text{target}} \mid s_t)
$$

*Conclusion*: Minimizing Cross-Entropy on positive samples is an exact proxy for maximizing Policy Gradient with reward +1.

**Case 2: The Negative Interaction (Skip)**
If the user skips, the reward is negative (e.g., $R_t = -1$). The gradient attempts to **decrease** the probability:

$$
\nabla \text{PG}_{neg} = (-1) \cdot \nabla \log \pi_\theta(a_{skip} \mid s_t) = - \frac{1}{\pi(a_{skip})} \nabla \pi(a_{skip})
$$

**The Instability Trap**: As the model improves, $\pi(a_{skip}) \to 0$, causing the term $\frac{1}{\pi}$ to approach infinity (**Gradient Explosion**).
To fix this, we replace the unstable RL gradient with the **Binary Cross Entropy (BCE)** gradient for the negative class ($1 - \pi$), which is bounded and stable:

$$
\mathcal{L}_{Stable} \approx - \log(1 - \pi_\theta(a_{skip} \mid s_t))
$$

**Final Objective Formulation**
We conclude that the standard **Next Token Prediction (CLM)** loss functions as a **Numerically Stable Proxy** for the underlying RL objective.

$$ \mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CLM}} \quad (\text{serving as } \mathcal{L}_{\text{Stable-RL}}) $$

**Why This Derivation Matters?**
One might ask: *Why take the detour through Reinforcement Learning if we end up with a standard Supervised Loss?*
The value lies in the **Theoretical Robustness**. We are not merely "assuming" classification works for recommendation; we have proven that it is the **Optimal Policy Gradient Estimator** for positive feedback and a **Variance-Reduced Proxy** for negative feedback.
Crucially, this First Principles framework future-proofs the model. If we later need to optimize for continuous rewards (e.g., **Dwell Time, GMV**), we can simply re-introduce the scalar reward term $R_t$ into the gradient, transitioning seamlessly from Imitation Learning to full RL—something a pure classification perspective cannot theoretically justify.

#### 4.3.4 Inference Process (The Generative Flow)

During inference, we generate recommendations by simulating a conversation with the user profile.

**Step 1: Context Setup**
Initialize the sequence with the User Q-Former tokens (System Prompt) and recent history.

**Step 2: Intent Injection (The "Control" Step)**
Append a target **Action Token** to the sequence (e.g., `[CLICK]`). This conditions the model to generate only items that maximize the probability of this specific action.

**Step 3: Hierarchical Decoding (Beam Search)**
The model generates the item codes sequentially. Crucially, to ensure the generated tuple $(c_1, c_2, c_3)$ corresponds to a valid item in the RQ-KMeans tree, we must apply **Valid Path Masking**:

1.  **Level 1**: Predict top-k $c_1$ codes conditioned on the `[CLICK]` token.
2.  **Level 2**: For each $c_1$, predict top-k $c_2$ codes. **Masking**: Set logits of all $c_2$ that are not children of $c_1$ to $-\infty$.
3.  **Level 3**: For each sequence $(c_1, c_2)$, predict top-k $c_3$ codes. **Masking**: Restrict to valid children of $c_2$.

This hierarchical search allows the model to refine its prediction from coarse categories to precise items.

*   **Engineering Note (KV Cache)**: Since the history sequence is long and static during the decoding of the 3 codes, utilizing **KV Cache** is mandatory to prevent re-computing the attention map for the entire history at each step $c_1 \to c_2 \to c_3$.

**Step 4: Decode & Deduplicate**
Convert the generated code tuples $(c_1, c_2, c_3)$ back into Item IDs using the RQ-KMeans codebook lookup. Remove duplicates and items already in the user's history.

#### 4.3.5 Data Sampling & Calibration Strategy

Introducing Action Tokens (especially negatives like `[SKIP]`) creates a trade-off between **Generative Fluency** and **Discriminative Accuracy**.

**1. The "Negative Pollution" Risk**
If the training data is dominated by negative interactions (which is true in reality, where CTR < 5%), the model learns that the most probable next token is almost always `[SKIP]`.
*   **Consequence**: During inference, even when prompted with `[CLICK]`, the model's internal priors may be so skewed towards negativity that the probability mass for any specific item code $P(c_1 \mid \text{[CLICK]})$ becomes unstable or poorly calibrated.

**2. Calibration Requirement**
The probability predicted by the model $P_{\text{model}}(Action)$ is heavily dependent on the training sampling ratio $\alpha = \frac{N_{pos}}{N_{neg}}$.
To use the model for **Ranking** (predicting precise CTR), we must calibrate the output logit back to the real-world distribution using the standard log-odds correction:

$$ \text{Logit}_{\text{real}} = \text{Logit}_{\text{model}} - \log(\alpha) $$

**3. Task-Specific Construction**
We recommend constructing different training mixtures for different deployment goals:
*   **For Retrieval (Recall)**: Use a **High-Positive Ratio** (e.g., 1:1 or 1:2). We want the model to be an "optimist," fluent in generating valid item sequences. We only need enough negatives to teach it what *not* to recommend, not to estimate precise CTR.
*   **For Ranking (Precision)**: Use a **Realistic/Hard-Negative Ratio** (e.g., 1:10). Here, the goal is discrimination. The generative capability is secondary; we prioritize the model's ability to distinguish a `[CLICK]` from a `[SKIP]` given a specific item.

#### 4.3.6 Advanced Training Protocol: Transfer & Curriculum

Directly training a Generative Transformer on raw, high-negative (1:100) logs is notoriously difficult; the model tends to collapse, predicting `[SKIP]` for everything. To bridge the gap between "Generative Fluency" and "Discriminative Precision," we propose a **Two-Stage Curriculum**.

**Stage 1: Generative Pre-training (Structural Learning Phase)**
*   **Data Ratio**: High Positive (e.g., 1:1 or 1:2).
*   **Goal**: Learn the "Syntax" of item codes and basic user affinity. The model learns valid item transitions and semantic clusters.
*   **Scope**: Full model training.

**Stage 2: Discriminative Fine-tuning (Ranking Alignment Phase)**
*   **Data Ratio**: Progressively shift towards reality (Curriculum: 1:2 $\to$ 1:5 $\to$ 1:10).
*   **Goal**: Learn the subtle boundary between "Relevant" and "Clickable."
*   **Efficient Tuning Strategy**: To prevent Catastrophic Forgetting (where the model forgets how to generate valid items), we can apply **Parameter-Efficient Fine-Tuning (PEFT)**.
    *   **Freeze**: The Transformer Backbone (The "Brain").
    *   **Train**: Only the **Contextualized Residuals (Section 4.1)** and the Action Prediction Head.
    *   **Why**: As hypothesized in 4.1, the *Static Base* captures the semantic "Syntax" (learned in Stage 1), while the *Learnable Residual* captures the personalized "Preference" (refined in Stage 2). This allows us to adapt to high-negative ranking distributions with minimal computational cost and high stability.

### 4.4 Inference: Controllable Generation (Steering)

The true power of the Generative Action Transformer lies in its ability to be **steered** at inference time. Unlike standard retrieval models that output a fixed "most relevant" list, our model acts as a conditional generator. By manipulating the **Prompt Sequence** (the tokens appended after the user profile), we can dynamically alter the retrieval logic to satisfy diverse business requirements without retraining.

#### 4.4.1 Mechanism: Prompt Engineering for Recommendation

The input to the model during inference is:

$$ X_{inf} = [\text{User Profile}] + [\text{History}] + [\textbf{Control Prompts}] $$

The **Control Prompts** effectively reshape the probability distribution $P(\text{Next Item} \mid \text{Context})$, guiding the beam search toward specific subspaces of the item catalog.

#### 4.4.2 Use Case Scenarios

**Scenario A: Maximize CTR (The "Best Bet" Mode)**
*   **Goal**: Retrieve the items most likely to be clicked by the user.
*   **Prompt**: `[CLICK]`
*   **Mechanism**: This conditions the model on the positive interaction token. The model ignores items that it predicts would lead to `[SKIP]` or `[VIEW]`, focusing solely on high-confidence candidates.
*   **Application**: Homepage "Top Picks", Fallback retrieval.

**Scenario B: Exploration & Discovery**
*   **Goal**: Break the "Filter Bubble" and surface novel items that the user hasn't seen but might like.
*   **Prompt**: `[EXPLORE]` + `[CLICK]`
*   **Mechanism**: The `[EXPLORE]` token (learned during training to associate with diverse/tail items) penalizes the retrieval of items too similar to the immediate history, while the subsequent `[CLICK]` ensures the novel items are still relevant enough to engage with.
*   **Application**: "New Arrivals", "Discover" tab.

**Scenario C: Visual Similarity Search (Item-to-Item)**
*   **Goal**: Find items visually similar to a specific seed item (e.g., "Find more shoes like *this one*").
*   **Prompt**: `[Seed Item ID]` + `[CLICK]`
*   **Mechanism**: By injecting a specific Item ID as the immediate context, we force the Transformer's attention mechanism to attend heavily to that item's features. The `[CLICK]` token then filters for similarity *that matters to the user* (e.g., matching style/brand rather than just color).
*   **Application**: "More like this", "Complete the look".

**Scenario D: Category Steering**
*   **Goal**: Retrieve relevant items, but restrict them to a specific category (e.g., "Hiking Gear").
*   **Prompt**: `[Category: HIKING]` + `[CLICK]`
*   **Mechanism**: We can learn lightweight "Soft Prompts" (continuous vectors) representing categories. Pre-pending this prompt biases the item code prediction heads toward the subspace of the codebook associated with hiking gear.
*   **Application**: Category pages, Search result re-ranking.

## 5. Future Directions: End-to-End Latent Tokenization

> *Note: This chapter is a theoretical exploration intended to inspire new ways of thinking about the "Indexability vs. Expressiveness" trade-off. Due to the "Three Death Traps" (Section 5.3) and massive engineering complexity, this architecture is NOT currently feasible for production. It serves primarily as a conceptual framework to highlight the limitations of current two-stage paradigms.*

While Chapter 4 presented a robust, industry-proven solution using RQ-KMeans, it inherently suffers from a **"Two-Stage Gap"**: the Item Tokenizer is trained on geometric similarity (Euclidean distance), while the Retrieval Transformer is trained on semantic preference (Next Item Prediction). Geometric closeness does not always imply semantic substitutability.

This chapter explores a cutting-edge theoretical alternative: **What if we learn the tokenization end-to-end within the Transformer?**

Recent works such as the **Byte Latent Transformer (BLT)** [3] demonstrate that latent patches can scale better than fixed tokens, suggesting that the era of rigid tokenizers may be ending.

### 5.1 The Concept: In-Transformer Residual Quantization

Instead of using a fixed, pre-computed vocabulary (from RQ-KMeans), we propose injecting a learnable **Latent Codebook** directly into the Transformer's layers. This aligns with the "Recurrent Refinement" philosophy of **Universal Transformers** [4], where representations are iteratively polished layer-by-layer.

**The Hypothesis**:
If we force the Transformer to pass its hidden state through a discrete bottleneck (via Gumbel-Softmax) layer-by-layer, it might learn a "Language of Items" that is perfectly optimized for the retrieval task, rather than for geometric reconstruction.

This concept is structurally isomorphic to the **Conditional Residual Beam Retrieval (CRBR)** proposed in Chapter 3. In CRBR, we used a *fixed* residual codebook to route user interests. Here, we propose moving that mechanism *inside* the Transformer to learn a *dynamic* residual codebook. It is the logical evolution from "Routing over fixed paths" to "Learning the paths themselves."

### 5.2 Theoretical Architecture

We envision a **Residual Routing Transformer** that progressively refines the user intent through discrete checkpoints.

Let $H_0$ be the initial user history embedding. The model generates a sequence of latent query tokens $q_1, q_2, q_3$:

**Layer 1 (Coarse Intent)**

$$ q_1 = \text{GumbelTop1}( \text{Projector}_1(H_0), \mathcal{C}_1 ) $$
$$ H_1 = \text{TransformerBlock}(H_0, q_1) $$

**Layer 2 (Fine Intent)**

$$ q_2 = \text{GumbelTop1}( \text{Projector}_2(H_1), \mathcal{C}_2 ) $$
$$ H_2 = \text{TransformerBlock}(H_1, q_2) $$

**Layer 3 (Precise Item)**

$$ q_3 = \text{GumbelTop1}( \text{Projector}_3(H_2), \mathcal{C}_3 ) $$

**Training Objective**

$$ \mathcal{L} = \| \text{StopGrad}(q_1+q_2+q_3) - \text{TargetItem} \| + \mathcal{L}_{\text{Hierarchy}} $$

### 5.3 Key Optimization Challenges

While theoretically elegant, this approach faces three formidable optimization challenges that explain why it has not yet replaced RQ-KMeans:

**1. The Posterior Collapse (Codebook Death)**
*   **The Phenomenon**: Neural networks are "lazy." They often find that relying on just 5-10 specific codes in the dictionary is enough to minimize the average loss, ignoring the other 990 codes.
*   **The Result**: The "Vocabulary" collapses. The model loses resolution, predicting the same generic "average items" for everyone. RQ-KMeans avoids this by forcibly partitioning the space geometrically.

**2. Hierarchy Vanishing (The "Residual" Lie)**
*   **The Phenomenon**: Without strong geometric constraints, the model may not obey the "Coarse $\to$ Fine" hierarchy. $q_1$ might learn nothing, leaving $q_3$ to do all the work, or vice versa.
*   **The Result**: The multi-stage structure becomes redundant. The vectors do not form a meaningful residual chain ($y \approx q_1 + q_2 + q_3$), making the discrete codes semantically meaningless.

**3. The Indexability Problem**
*   **The Problem**: In RQ-KMeans, every item has a fixed ID `[12, 55, 9]`. We can build an inverted index. In this Latent approach, the "Code" is a dynamic activation of the User Tower.
*   **The Consequence**: To retrieve an item, we must know its dynamic code. This requires training a **Dual-Tower VQ-VAE** (an Item Tower that mimics the User Tower's tokenization), adding complexity to ensure the two towers stay aligned.

### 5.4 Path Forward: The "Hierarchy Guarantee" Protocol

To prevent the "Death Traps" and make this end-to-end dream a reality, we must impose strict constraints that force the model to respect the residual hierarchy. We propose a **Progressive Residual Training Protocol**:

**1. Explicit Residual Targets (Don't let layers "redo" work)**
Instead of letting every layer predict the same target $y$, we must mathematically enforce the residual definition.
*   **Layer 1 Target**: The Item Embedding $y$.
*   **Layer 2 Target**: The Residual $r_1 = y - \text{StopGrad}(\hat{y}_1)$.
*   **Layer 3 Target**: The Fine Residual $r_2 = r_1 - \text{StopGrad}(\hat{y}_2)$.
This forces $q_2$ and $q_3$ to learn *only* what previous layers missed.

**2. "Strictly Better" Constraint (Progressive Margin Loss)**
We add a ranking loss to ensure that adding a layer *always* improves prediction accuracy.

$$ \mathcal{L}_{margin} = \sum_{k=1}^{2} \max(0, \mathcal{L}(H_{k+1}, y) - \mathcal{L}(H_k, y) + \delta) $$

This penalizes the model if deeper layers (which consume more compute/tokens) do not provide a tangible gain ($\delta$) over shallower layers.

**3. Capacity & Orthogonality Constraints**
*   **Codebook Utilization**: Enforce strict entropy maximization ($\mathcal{L}_{entropy}$) on the Gumbel-Softmax distribution to prevent collapse to a few tokens.
*   **Orthogonality**: Force latent query vectors to be orthogonal ($\langle q_k, q_{k+1} \rangle \approx 0$) to ensure they capture complementary information subspaces.

**4. Dual-Tower Codebook Alignment**
To solve the indexing crisis, we must train a lightweight **Item Encoder** (Tower B) alongside the User Tower (Tower A).
*   **Objective**: User Code $z_u$ should match Item Code $z_i$.
*   **Mechanism**: Shared Codebooks. The Item Encoder acts as the "Ground Truth" tokenizer, while the User Tower learns to predict these tokens from history.


## 6. References

[1] **OneRec-V2 Technical Report**. (2025). *arXiv preprint arXiv:2508.20900*. Retrieved from https://arxiv.org/abs/2508.20900

[2] **Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning**. (2025). *arXiv preprint arXiv:2502.03275*. Retrieved from https://arxiv.org/pdf/2502.03275

[3] Patraucean, V., et al. **Byte Latent Transformer: Patches Scale Better Than Tokens**. (2024). *arXiv preprint arXiv:2412.09871*. Retrieved from https://arxiv.org/pdf/2412.09871

[4] Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. **Universal Transformers**. (2019). *International Conference on Learning Representations (ICLR)*. Retrieved from https://arxiv.org/pdf/1807.03819