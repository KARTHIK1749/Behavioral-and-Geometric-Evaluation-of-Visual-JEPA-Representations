# Behavioral and Geometric Evaluation of Visual JEPA Representations
![Python](https://img.shields.io/badge/Python-3.9%2B-orange)
![Reproducible](https://img.shields.io/badge/Reproducible-success)
![Research](https://img.shields.io/badge/Category-Research-blue)
![License](https://img.shields.io/badge/License-MIT-violet)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-white)
![Status](https://img.shields.io/badge/Status-Active-yellow)




This repository contains the official codebase and experimental framework accompanying the paper:

### Behavioral and Geometric Evaluation of Visual JEPA Representations

**Karthik Raj Panuganti**  

OSI Preprint : ![Preprint](https://osf.io/cx629/files/ut73f)

---

## Overview
This project presents a **multi-axis behavioral probing framework** for analyzing the geometry and stability of visual representation models, with a primary focus on **Visual JEPA (VL-JEPA)**.

Rather than evaluating models through downstream task accuracy, we examine **latent embedding behavior** across seven hypothesis-driven probes designed to characterize semantic stability, shortcut reliance, robustness geometry, temporal coherence, ambiguity handling, dataset subsampling behavior, and scaling dynamics.

### Behavioral Probes(H1-H7)
- H1 — Semantic Faithfulness under Corruption  
- H2 — Shortcut Bias (Foreground vs Background)  
- H3 — Robustness Geometry  
- H4 — Temporal Stability  
- H5 — Ambiguity Handling  
- H6 — Dataset Subsampling Stability  
- H7 — Scaling Geometry  

Baselines include **ResNet-18, DINO-ViT, CLIP-ViT, and VideoMAE**.

---

## Repository Structure

```graphql
VL-JEPA/
│
├── experiments/        # H1–H7 experimental probes
├── metrics/            # SDI, TSS, robustness, entropy similarity metrics
├── utils/              # embedding loaders, helpers, VL-JEPA wrappers
├── tools/              # dataset construction utilities (Pexels API)
├── notebook/           # exploratory analysis
├── plots/              # generated visualizations
├── results/            # experiment outputs (JSON/CSV)
├── models/             # pretrained checkpoints (not tracked)
├── data/               # curated dataset (384 images + 2 videos)
├── paper/              # LaTeX source, figures, tables, draft
├── external/           # official VL-JEPA source
├── run_guide.md        # reproducible setup and execution guide
└── README.md

```

---

## Dataset

We construct a **controlled exploratory dataset** of **384 images** across eight categories:

- abstract --> 29 images 
- ambiguous --> 40 images
- animals --> 65 images
- humans --> 70 images
- indoor --> 45 images
- outdoor --> 45 images
- objects --> 45 images
- vehicles --> 45 images

**Note : Images are sourced from Pexels under permissible license terms and used solely for research.**

Temporal probes use **two short video clips (~11 seconds each).**

Dataset generation script:

```bash
python tools/build_dataset.py
```
**Note** : API keys and private credentials are not included

---

## Running Experiments

Each hypothesis is implemented as a standalone script:

```bash
python experiments/h1_semantic_faithfulness.py
python experiments/h2_shortcut_bias.py
python experiments/h3_noise_robustness.py
python experiments/h4_temporal_stability.py
python experiments/h5_ambiguity.py
python experiments/h6_data_efficiency.py
python experiments/h7_scalability.py
```

Outputs are saved to:

```bash 
plots/ # generated figures
results/ # json/csv metrics
```
---

## Metrics Implemented

- Shortcut Dependency Index (SDI)

- Temporal Stability Score (TSS)

- Robustness Curvature

- Intrinsic Dimensionality Growth

- Neighborhood Stability

- Embedding Drift & Collapse Scores

Metrics Implementations reside in :

```bash
metrics/
```
**Formal definitions appear in Section 3 of the paper.**

---
## Compute Environment

Experiments were run on:

- NVIDIA RTX 3050 (6GB VRAM)

- 16GB system RAM

- Compact pretrained model checkpoints

Results should be interpreted as **exploratory and compute-constrained.**

---
## Paper

The manuscript is located in:

```bash
paper/paper.tex
```
---

## Citation & Credit

If you use this code, dataset, or findings in your research, please cite the accompanying paper:


```bibtex
@article{panuganti2024jepa,
  title={Behavioral and Geometric Evaluation of Visual JEPA Representations},
  author={Panuganti, Karthik Raj},
  year={2024},
  archivePrefix={arXiv}
}

```

---

## Research Intent
This project does not claim benchmark superiority.

Its goal is to understand how predictive learning shapes **latent representation geometry**, offering tools for **behavioral and geometric interpretability** of vision models

--- 

## Reproducibility
All experiments can be reproduced by following [run_guide.md](run_guide.md), which lists the environment setup, dataset regeneration steps, and the command sequence for each experiment script.

---

## Author

**Karthik Panuganti** 

Independent Researcher (ML / Vision / Representation Learning)

