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

    subgraph Phase1_Architecture ["Phase 1: Q-Former Pre-training (The 'Speedy' Encoder)"]
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
            subgraph Attention_Mechanism ["Efficiency Core: Cross-Attention Only"]
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

#### 3.1.2 Pre-training Strategy: Encoder-Decoder

The User Tower is pre-trained using a multi-task objective designed to ensure both discriminative power (for retrieval) and generative understanding (for profile completeness).

**Task A: Image-Text Contrastive Loss (ITC)**
*   **Objective**: Aligns the user profile (Query Tokens) with the representation of the positive future item (target).
*   **Mechanism**: A contrastive loss maximizes the similarity between the user's updated query tokens and the target item embedding, while pushing away negatives. This directly optimizes for the **Action Alignment** principle.

**Task B: Masked Modeling (Reconstruction)**
*   **Objective**: Ensures the compressed query tokens retain all necessary information from the input history.
*   **Mechanism**: We randomly mask a percentage of the input history items. A lightweight decoder then attempts to reconstruct the missing items solely from the Q-Former's output tokens. This prevents the model from overfitting to easy patterns and encourages a robust, holistic understanding of user behavior.

#### 3.1.3 Theoretical Justification: The "Differentiation" Strategy

A critical challenge in single-vector models (like Pinformer) is the conflict between the objectives of **Task A (ITC)** and **Task B (Masked Modeling)**.
*   **Task A (ITC)** pushes for **Selectivity**: It forces the representation to discard historical details to align tightly with the immediate future item (for retrieval sharpness).
*   **Task B (Masked Modeling)** pushes for **Preservation**: It forces the representation to retain all historical details to reconstruct the past (for profile completeness).

When compressed into a single vector, these opposing gradients lead to a **"Blurred Average"** that is neither sharp nor complete.

**Solution: Sparsity in Set Space**
By using a set of $M$ tokens, Q-Former allows these objectives to be satisfied **orthogonally across different tokens**, utilizing the "Soft Sparsity" of the attention mechanism:
1.  **Specialization via Task B**: The Masked Modeling loss acts as a **Diversity Regularizer**, penalizing "Mode Collapse" (where all tokens learn the same recent interest). It forces specific tokens to specialize in encoding long-tail history (which Task A would otherwise discard).
2.  **Alignment via Task A**: The ITC loss then tunes specific tokens to align with potential future interests.
3.  **Result**: The latent space becomes a **Multi-Modal Set** rather than a single Mean Vector. $Token_{A}$ may capture **Dominant Interests** (e.g., "Running Shoes", High Task A utility), while $Token_{B}$ captures **Niche/Long-Tail Interests** (e.g., "Climbing Carabiners", High Task B utility), ensuring the static profile remains sharp across the full spectrum of user preferences.

### 3.2 Retrieval Approach A: Set Transformer (The Standard)

This section describes the standard industry baseline for utilizing the Q-Former's output.

**The Aggregation Bottleneck**
The Q-Former outputs a set of $M$ diverse tokens $Q_{out} = \{q_1, \dots, q_M\}$. To perform standard Two-Tower retrieval (which requires a single vector dot-product), these tokens must be aggregated.

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
            subgraph User_Adapter ["🔥 Trainable User Adapter 🔥"]
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

**Limitations**
While this approach is better than a simple MLP, it still forces the multi-modal user interest into a **Single Point** in the embedding space for the final retrieval step. This fundamentally limits the ability to retrieve items from disparate categories simultaneously (e.g., retrieving both "Shoes" and "Watches" in one pass if they lie in different directions). This limitation motivates the multi-path approach in Section 3.3.

### 3.3 Retrieval Approach B: Conditional Residual Beam Retrieval (CRBR)

#### 3.3.1 Motivation: The Curse of Parameter Sharing

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

#### 3.3.5 Output Normalization

Since the target embedding space (FashionCLIP) is a hypersphere, the output of the final layer $L$ must be normalized before ANN retrieval.

$$q_{final} = \frac{v^{(L)}}{\| v^{(L)} \|_2}$$

#### 3.3.6 Optimization Objectives

The loss function optimizes the final vector representation while ensuring codebook utilization and geometric alignment.

$$\mathcal{L}_{total} = \mathcal{L}_{NCE} + \alpha \mathcal{L}_{Align} + \beta \mathcal{L}_{Balance}$$

**InfoNCE Loss (Retrieval)**

$$\mathcal{L}_{NCE} = -\log \frac{\exp(\tilde{q} \cdot i^{+} / \tau_{nce})}{\sum_{i^{-}} \exp(\tilde{q} \cdot i^{-} / \tau_{nce})}$$

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

    subgraph Contextualized_Residue_Block [Contextualized Item Embedding Block]
        direction TB

        %% Inputs
        User["User Profile Tokens<br>(from Phase 1)"]:::frozen
        ItemID[Item ID]:::frozen

        %% Branches
        subgraph Static_Branch [Static Semantic Branch]
            E_static[E_static<br>(Frozen Pre-trained)]:::frozen
        end

        subgraph Learnable_Branch [Dynamic Capacity Branch]
            E_learnable[E_learnable<br>(Trainable Table)]:::learnable
        end

        %% Gating Network
        subgraph Gating_Net [Gating Controller]
            StopGrad(("&perp; Stop Gradient")):::operation
            Concat([Concat])
            MLP_Gate["MLP &sigma;<br>(Init close to 0)"]:::learnable
            Gate_Val["Gate g"]:::operation
        end

        %% Flow
        User --> StopGrad --> Concat
        ItemID --> E_static --> Concat
        ItemID --> E_learnable

        Concat --> MLP_Gate --> Gate_Val

        %% Fusion
        Fusion(("(1-g)*Static + g*Learnable")):::operation
        E_final[Final Contextualized<br>Item Embedding]:::operation

        E_static --> Fusion
        E_learnable --> Fusion
        Gate_Val -.->|Modulates| Fusion
        Fusion --> E_final
    end
```

### 4.2 The Foundation: Robust Discretization (RQ-KMeans)

While Contextualized Residuals improve the representation, standard contrastive losses (NCE) still suffer from "distributional blurring" in the continuous space. To achieve true sharpness, we shift the paradigm from "predicting a vector" to "predicting a code" via **Residual Quantization (RQ-KMeans)**.

To be populated.
(Details on Item Embedding Health: Anisotropy, Hubness; and Codebook Health: Dead Codes, Entropy, Inertia).

### 4.3 The Architecture: Generative Action Transformer

With a sharp discrete space (Phase 3) and personalized capacity (Phase 2), we build the **Generative Action Transformer**. This model treats user actions (Click, Cart, etc.) as first-class citizens in the sequence.

To be populated.
(Details on interleaving Action Tokens, the dual training objectives, and how this achieves the "Action Alignment" North Star).

### 4.4 Inference: Controllable Generation

To be populated.
(Details on using Control Tokens at inference time to steer the generative process for specific business goals).
