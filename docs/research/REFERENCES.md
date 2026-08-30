# Research References and Design Connections

This file is a research map, not a claim that any cited work implements APC as a complete system.

## 1. Conditional memory and capacity/compute separation

### Qwen3.8-Flash-Next
Uses large N-gram embedding memory in addition to the main model while keeping per-token active parameters much smaller than total capacity. Relevant to separating stored capacity from recurrent neural computation.

https://qwen.ai/blog?id=qwen3.8-flash-next

### Engram — Conditional Memory via Scalable Lookup
Frames scalable lookup memory as an axis complementary to conditional computation and reports mechanistic evidence that static memory can free Transformer depth for other computation.

https://arxiv.org/abs/2601.07372

## 2. Fine-grained sparse computation

### PEER — Mixture of a Million Experts
Product-key retrieval over very large numbers of tiny experts. Relevant to scaling a primitive bank without scoring every primitive densely.

https://arxiv.org/abs/2407.04153

## 3. Decomposing representations and transformations

### MOLT — Sparse Mixture of Linear Transforms
Anthropic preliminary research decomposing MLP behavior into sparsely activated low-rank transforms. Conceptually close to computation primitives rather than only feature dictionaries.

https://transformer-circuits.pub/2025/bulk-update/index.html

### Mixture of Decoders
Decomposes pretrained MLP behavior into many specialized sparse sublayers while retaining model behavior at tested scales.

https://papers.neurips.cc/paper_files/paper/2025/hash/d51ab0fc62fe2d777c7569952f518f56-Abstract-Conference.html

## 4. Reusable/composable circuits

### ModCirc
Studies task-agnostic modular circuit vocabularies and criteria including reusability and composability.

https://openreview.net/pdf?id=do5vVfKEXZ

### Circuit Compositions
Evidence in smaller Transformer settings that circuits for related tasks overlap and that compositions of circuits can represent more complex behavior.

https://arxiv.org/abs/2410.01434

### Attribution Graphs / circuit tracing
Tools and methodology for tracing causal computational structure in language models; useful in later phases for validating extracted primitives.

https://www.anthropic.com/research/open-source-circuit-tracing

## 5. Dynamic growth and pruning

### Neuroplastic Expansion
Dynamic neuron generation, consolidation and pruning for continual learning. Relevant to temporary resource allocation.

https://proceedings.iclr.cc/paper_files/paper/2025/hash/e094d1e30e88949ed466067aef6be546-Abstract-Conference.html

### SCALE
Continual pretraining with architectural expansion while preserving pretrained parameters. Relevant to expansion without destructive overwriting.

https://aclanthology.org/2026.findings-acl.2037/

### DEMM
Dynamic expandable/mergeable model using fast and slow components, expert construction, merging and pruning. Relevant to the Plastic -> Consolidate -> Release loop.

https://www.sciencedirect.com/science/article/pii/S0952197626012534

## 6. Stability–plasticity and uncertainty

### MESU
Metaplasticity via uncertainty for continual learning and OOD detection. Relevant to novelty signals and parameter-specific learning rates.

https://www.nature.com/articles/s41467-025-64601-w

### Metaplasticity / heterogeneous synaptic plasticity work
Useful conceptual support for having different update rates or stability levels across persistent and temporary parameters.

https://www.sciencedirect.com/science/article/pii/S0893608025006422

## 7. Test-time and multi-timescale learning

### Titans
Neural memory that can update at test time. Relevant to fast adaptation without treating inference as immutable.

https://research.google/pubs/titans-learning-to-memorize-at-test-time/

### Nested Learning / Hope
Models learning systems as nested optimization processes with different update frequencies. Relevant to stable, plastic and meta-learning timescales.

https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

## 8. Biological inspiration

### Complementary Learning Systems
Fast episodic learning and slower statistical consolidation provide a biological analogy for separating high-plasticity temporary learning from stable long-term representation.

Recent review:
https://pubmed.ncbi.nlm.nih.gov/42421581/

## Notes for future literature reviews

Before Phase D and before publishing claims, re-check current literature for:
- dynamic model expansion;
- sparse expert retrieval;
- circuit extraction/factorization;
- continual learning under bounded memory;
- online/test-time learning;
- learned architecture/resource controllers;
- consolidation and replay;
- mechanistic interpretability of modular computation.
