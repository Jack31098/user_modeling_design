Design Specification: Differentiable Routing Block

Model: Conditional Residual Beam Retrieval (CRBR)

Component: Routing Block (Layer $l$)

1. Overview

The Routing Block at layer $l$ functions as a conditional residual generator operating within a Beam Search framework. It takes a set of active beam paths, selects the optimal residual vector from a learnable codebook based on the user context and current accumulation, and updates the path state.

1.1 High-Level Architecture

The overall architecture employs a deep stack of routing blocks sharing global codebooks, terminating in a standard ANN index.

(Note: The following Mermaid code describes the architecture. Please render it using a Mermaid-compatible editor to generate the diagram image.)

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


2. Notation & Parameters

2.1 Learnable Parameters

Codebook $\mathcal{C}^{(l)}$: A matrix of size $M \times D$, containing $M$ learnable residual prototypes.

$$\mathcal{C}^{(l)} = \{c_1, c_2, \dots, c_M\}, \quad \text{where } c_j \in \mathbb{R}^D$$

Routing MLP $f_{\theta}^{(l)}$: A neural network (e.g., Linear $\to$ ReLU $\to$ Linear) that maps the fused state to the codebook metric space.

2.2 Inputs (State at Layer $l-1$)

The input consists of a set of $B$ active paths (where $B$ is the Beam Width). For the $k$-th path ($k \in \{1, \dots, B\}$):

Accumulated Vector $v_{k}^{(l-1)}$: The sum of residuals from all previous layers.

Path Score $S_{k}^{(l-1)}$: The cumulative log-probability of the path up to layer $l-1$.

User Context $Q$: Static global context vector (e.g., output from Q-Former), $Q \in \mathbb{R}^{D}$.

3. Mathematical Process

The process is divided into a shared Affinity Computation, followed by divergent branches for Training (Differentiable) and Inference (Beam Search).

3.1 Routing Block Logic Flow
![Routing Block Logic Flow]([https://raw.githubusercontent.com/Jack31098/user_modeling_design/refs/heads/main/routing_block.png])

3.2 Step 1: Context Fusion & Affinity Scoring

For each active path $k \in \{1, \dots, B\}$, we compute the compatibility distribution over the codebook entries.

1. Conditional Projection
Determine the search direction based on the current position ($v$) and intent ($Q$).

$$h_k = f_{\theta}^{(l)}( \text{Concat}(Q, v_{k}^{(l-1)}) )$$

2. Logit Computation
Compute the dot-product similarity with all $M$ entries in the codebook.

$$z_{k} = h_k \cdot (\mathcal{C}^{(l)})^\top$$

Where $z_{k} \in \mathbb{R}^M$, and $z_{k,j}$ represents the raw affinity score for the $j$-th code.

3.3 Step 2: Branching Strategy

Mode A: Training (Gumbel-Softmax)

To allow gradient backpropagation through the discrete selection, we employ the Gumbel-Softmax reparameterization.

Gumbel Noise Injection: Sample $g_j \sim \text{Gumbel}(0, 1)$ i.i.d. for each code $j$.

Soft Selection Probabilities: The probability $\pi_{k,j}$ for path $k$ selecting code $j$ is:

$$\pi_{k,j} = \frac{\exp( (z_{k,j} + g_j) / \tau )}{\sum_{m=1}^{M} \exp( (z_{k,m} + g_m) / \tau )}$$

(where $\tau$ is the temperature parameter)

Soft Residual Extraction: We calculate the expected residual vector.

$$r_{k}^{(l)} = \sum_{j=1}^{M} \pi_{k,j} \cdot c_j$$

State Update:

$$v_{k}^{(l)} = v_{k}^{(l-1)} + r_{k}^{(l)}$$

Mode B: Inference (Beam Search)

We perform exact selection and pruning to maintain the top-$B$ best global paths.

Candidate Expansion: Expand all $B$ current paths into $B \times M$ potential new paths. The global score for candidate $(k, j)$ is:

$$\text{Score}_{k,j} = S_{k}^{(l-1)} + \log( \text{Softmax}(z_{k,j}) )$$

Pruning (Top-K): Select the top-$B$ candidates with the highest scores from the pool.

$$\mathcal{P}_{new} = \text{TopK}_{B}( \{ \text{Score}_{k,j} \mid \forall k, \forall j \} )$$

Hard Residual Extraction: For each selected candidate corresponding to parent path $k^*$ and code index $j^*$:

$$r_{new} = c_{j^*}$$

$$v_{new}^{(l)} = v_{k^*}^{(l-1)} + r_{new}$$

$$S_{new}^{(l)} = \text{Score}_{k^*, j^*}$$

4. Output Normalization

Since the target embedding space (FashionCLIP) is a hypersphere, the output of the final layer $L$ must be normalized before ANN retrieval.

$$q_{final} = \frac{v^{(L)}}{\| v^{(L)} \|_2}$$

5. Optimization Objectives

The loss function optimizes the final vector representation while ensuring codebook utilization and geometric alignment.

$$\mathcal{L}_{total} = \mathcal{L}_{NCE} + \alpha \mathcal{L}_{Align} + \beta \mathcal{L}_{Balance}$$

5.1 InfoNCE Loss (Retrieval)

Maximizes the similarity between the soft-generated query $\tilde{q}$ and the positive item $i^+$.

$$\mathcal{L}_{NCE} = -\log \frac{\exp(\tilde{q} \cdot i^+ / \tau_{nce})}{\sum_{i^-} \exp(\tilde{q} \cdot i^- / \tau_{nce})}$$

5.2 Geometric Alignment Loss

Forces the accumulated vector $v^{(l)}$ to geometrically approach the target item $i^+$, ensuring the residuals have physical meaning.

$$\mathcal{L}_{Align} = \| v^{(l)} - i^+ \|_2^2$$

5.3 Load Balancing Loss

Maximizes the entropy of the batch-averaged probability distribution $\bar{\pi}$ to prevent mode collapse.

$$\mathcal{L}_{Balance} = \sum_{j=1}^{M} \bar{\pi}_j \log \bar{\pi}_j$$
