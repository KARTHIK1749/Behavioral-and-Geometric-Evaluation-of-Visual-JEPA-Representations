# Behavioral and Geometric Evaluation of Visual JEPA Representations


### ABSTRACT
We present a multi-axis behavioral and geometric evaluation of Visual Joint Embedding Predictive Architectures (JEPA), analyzing how learned visual representations respond to corruption, shortcut bias, temporal variation, ambiguity, dataset subsampling, and scaling. Across seven hypothesis-driven probes (H1–H7), we compare JEPA with ResNet-18, DINO-ViT, CLIP-ViT, and VideoMAE using representation-level metrics rather than downstream task accuracy.

Within our constrained experimental setting, JEPA exhibits slower semantic degradation under corruption, reduced reliance on background shortcuts, smoother robustness degradation profiles, stronger short-term temporal embedding stability, more structured responses to ambiguous inputs, improved preservation of local neighborhood structure under subsampling, and faster intrinsic dimensionality expansion under scale. Effect sizes and cross-hypothesis trends suggest these behavioral differences are consistent within our evaluation regime.

Rather than proposing a new architecture, this work contributes an exploratory behavioral evaluation framework and an empirical characterization of JEPA representation geometry, offering insight into how predictive learning objectives shape latent structure beyond standard benchmark-driven evaluation.

---

### 1. INTRODUCTION
Vision models are predominantly evaluated using accuracy-based benchmarks, which provide limited insight into the internal organization, stability, and geometry of learned representations. Predictive and self-supervised learning frameworks, including JEPA, claim to produce more semantically structured and stable latent representations; however, systematic behavioral evaluations of these claims remain limited.

In this work, we evaluate JEPA from a representation-centric perspective, focusing on how embeddings behave under controlled perturbations rather than how models perform on classification tasks. We probe seven representational properties: semantic faithfulness, shortcut bias, robustness geometry, temporal stability, ambiguity handling, dataset subsampling stability, and scaling behavior.

**Contributions** :

- We introduce a seven-axis behavioral probe suite for exploratory evaluation of visual representation structure.

- We provide an early multi-probe behavioral analysis of JEPA relative to CNN and contrastive baselines.

- We present empirical evidence that predictive representations in JEPA exhibit smoother degradation profiles, stronger object-centric encoding, more structured ambiguity responses, and accelerated intrinsic dimensionality expansion under scale.

Our goal is not to claim task-level superiority, but to characterize how JEPA organizes semantic information in latent space under controlled experimental conditions.

---

### 2. RELATED WORK 

Our work builds on advances in self-supervised and predictive representation learning, robustness analysis, shortcut bias, and latent space geometry.

**Self-Supervised and Predictive Representation Learning.**  
Contrastive learning methods such as SimCLR (Chen et al., 2020), MoCo (He et al., 2020), DINO (Caron et al., 2021), and CLIP (Radford et al., 2021) have demonstrated strong unsupervised visual representation learning by enforcing instance-level discrimination. However, contrastive objectives often require large batch sizes and may amplify shortcut features.

Joint Embedding Predictive Architectures (JEPA) (LeCun et al., 2022; Bardes et al., 2023) propose learning representations by predicting future latent states rather than relying on negative samples, encouraging invariance and semantic abstraction. Despite strong empirical results, systematic behavioral evaluation of JEPA-style representations remains limited.

**Robustness, Shortcut Learning, and Texture Bias.**  
Prior studies have shown that vision models frequently exploit spurious correlations such as background textures, color cues, and dataset-specific shortcuts (Geirhos et al., 2019; Ilyas et al., 2019; Sagawa et al., 2020). Efforts to evaluate robustness under corruption (Hendrycks & Dietterich, 2019) and distribution shift highlight persistent fragility in CNN and Transformer-based models.

**Representation Geometry and Interpretability.**  
Recent work explores intrinsic dimensionality (Ansuini et al., 2019), manifold smoothness (Pope et al., 2021), latent space collapse (Jing et al., 2022), and embedding topology in neural representations. Studies on representation probing (Alain & Bengio, 2016; Hewitt & Liang, 2019) emphasize understanding internal structure beyond task accuracy.

**Positioning of This Work.**  
Unlike prior studies that emphasize downstream benchmark performance or isolated robustness metrics, we present a **multi-axis behavioral probing framework** to characterize how predictive learning objectives shape representation geometry, shortcut reliance, temporal stability, and ambiguity handling. Our work focuses on **empirical latent space dynamics**, complementing architecture- and benchmark-driven evaluation paradigms.

---

### 2.5 Mathematical Intuition and Representation Geometry

We model representation learning as a function  
$$
f_\theta : \mathcal{X} \rightarrow \mathbb{R}^d
$$
that maps inputs $$ x \in \mathcal{X} $$  to latent embeddings $$ z = f_\theta(x) $$  
The **geometry of the latent space** determines semantic stability, robustness, and generalization.

Predictive learning objectives, such as those used in Joint Embedding Predictive Architectures (JEPA), optimize representations by predicting future latent states rather than contrasting against negative samples. This encourages **smooth latent trajectories**, **temporal coherence**, and **reduced sensitivity to spurious low-level cues**.

From a geometric perspective, high-quality representations exhibit:

- **Low curvature under perturbations**, indicating stable semantic manifolds  
- **Consistent local neighborhoods**, reflecting object-centric organization  
- **Controlled intrinsic dimensionality expansion**, enabling expressiveness without collapse  

Corruption robustness can be interpreted through embedding sensitivity:

$$
S(x, \delta) = \| f_\theta(x) - f_\theta(x + \delta) \|_2
$$

Shortcut bias is modeled as anisotropic dependence on foreground vs background features:

$$
\Delta_{\text{shortcut}} =
\mathbb{E}\left[\| f_\theta(x) - f_\theta(x_{fg}) \|_2\right]
-
\mathbb{E}\left[\| f_\theta(x) - f_\theta(x_{bg}) \|_2\right]
$$

Finally, intrinsic dimensionality growth reflects **latent manifold capacity**, where faster expansion under scale indicates richer representation structure without degeneracy.

This theoretical framing motivates our seven probes as **empirical tests of manifold smoothness, stability, and structural coherence**.


---

### **3. Metric Formalization and Quantitative Definitions**

We formally define the core metrics used across our hypothesis-driven probes.

---

### 3.1 Shortcut Dependency Index (SDI)

Let $$ f_\theta(x) \in \mathbb{R}^d $$ denote the embedding of image $$ x $$.  
We define foreground-perturbed and background-perturbed samples as $$ x_{fg} $$ and $$x_{bg}$$, respectively.

The Shortcut Dependency Index (SDI) is defined as:

$$
\text{SDI} = 
\mathbb{E}_{x} \left[ \| f_\theta(x) - f_\theta(x_{fg}) \|_2 \right]
\;-\;
\mathbb{E}_{x} \left[ \| f_\theta(x) - f_\theta(x_{bg}) \|_2 \right]
$$

Higher SDI values indicate **stronger object-centric encoding** and **reduced reliance on background shortcuts**.

---

### 3.2 Temporal Stability Score (TSS)

Given a video sequence $$ \{x_1, x_2, \dots, x_T\} $$, we compute temporal consistency as the mean cosine similarity between consecutive frame embeddings:

$$
\text{TSS} = \frac{1}{T-1} \sum_{t=1}^{T-1}
\frac{
f_\theta(x_t) \cdot f_\theta(x_{t+1})
}{
\| f_\theta(x_t) \|_2 \, \| f_\theta(x_{t+1}) \|_2
}
$$

Higher TSS reflects **stronger identity persistence and temporal coherence**.

---

### 3.3 Robustness Curvature (RC)

Let $$ x_\delta $$ denote an input corrupted with severity level $$ \delta $$.  
We define representation sensitivity under corruption as:

$$
D(\delta) = \| f_\theta(x) - f_\theta(x_\delta) \|_2
$$

Robustness curvature is estimated as the second derivative:

$$
\kappa = \frac{d^2 D(\delta)}{d \delta^2}
$$

Lower curvature values indicate **smoother degradation geometry** and **more stable latent manifolds**.

---

### 3.4 Intrinsic Dimensionality Growth (IDG)

We estimate intrinsic dimensionality using neighborhood-based estimators applied to embedding sets:

$$
\text{IDG}(n) = \hat{d}_{\text{intrinsic}}(f_\theta(X_n))
$$

where $$ X_n $$ denotes subsets of increasing sample scale.  
Faster IDG growth suggests **higher latent expressiveness without representational collapse**.


These formal definitions ground our behavioral probes in **latent manifold geometry**, reducing reliance on heuristic metrics.


---

### **4. METHODOLOGY : H1 - H7 Probes**

We design seven hypothesis-driven behavioral probes (H1–H7) to evaluate representation stability and structure. All experiments operate on frozen pretrained encoders, isolating representational properties rather than training dynamics.

**Models :**

We evaluate JEPA against ResNet-18, DINO-ViT, CLIP-ViT, and VideoMAE.

**Datasets :**

We construct a controlled exploratory evaluation dataset consisting of **384** images retrieved via the **Pexels API**. Images are distributed across eight semantic categories: **abstract (29), ambiguous (40), animals (65), humans (70), indoor scenes (45), outdoor scenes (45), objects (45), and vehicles (45).** 

This dataset is intended for behavioral probing rather than **large-scale generalization** and allows **controlled analysis of representation dynamics** under varied semantic conditions.

For temporal analysis, we use **two short video clips (~11 seconds each)**, evaluated across multiple vision encoders including VideoMAE.
 
Due to limited sample size, video-based findings are treated as **illustrative rather than statistically conclusive.**

---

### Compute and Experimental Constraints

All experiments were conducted on a single NVIDIA **RTX 3050 GPU (6GB VRAM) with 16GB system RAM**. Due to memory constraints, compact model variants were evaluated rather than the largest available **JEPA, CLIP, DINO, or ResNet checkpoints.**

Findings should therefore be interpreted as **empirical observations under constrained compute and model scale, rather than definitive large-scale comparisons.**

---

**H1 — Semantic Faithfulness under Corruption**

We measure representation similarity decay under **blur and Gaussian noise using similarity decay slopes, corruption curve stability, and Cohen’s d**. 

Lower degradation slopes indicate stronger semantic preservation.

---
**H2 — Shortcut Bias (Foreground vs Background)**

Foreground and background regions are **perturbed independently to compute similarity differentials, a Shortcut Dependency Index (SDI), and effect sizes.** 

Higher SDI indicates stronger object-centric encoding.

---
**H3 — Robustness Geometry across Corruption Types**

We evaluate multiple corruption regimes using **degradation slopes, failure curvature, and area under robustness curves (AURC).** 

Lower curvature indicates smoother degradation profiles.

---
**H4 — Temporal Stability across Video Frames**

We compute **mean temporal embedding similarity, Temporal Stability Score (TSS), and drift variance across frame sequences.**

Findings are interpreted as **illustrative due to limited video samples.**

---
**H5 — Ambiguity Handling**

Under ambiguous visual stimuli, we measure **embedding spread, cluster silhouette, and bifurcation strength.** 

Higher values indicate more structured multi-modal representation behavior.

---
**H6 — Stability under Dataset Subsampling**

We evaluate **k-NN neighborhood overlap, embedding drift, collapse scores, and embedding variance** under progressive subsampling. 

This probe measures local structural robustness, not universal data efficiency.

---
**H7 — Scaling Geometry**

We estimate **intrinsic dimensionality, cluster compactness, margin growth, and neighborhood stability.**

Faster intrinsic dimensionality expansion suggests richer latent capacity under increasing scale.

---

### 5. RESULTS 

### H1 - Semantic Stability

JEPA exhibits slower similarity decay within our evaluation setting under corruption than ResNet, DINO, and CLIP, indicating stronger preservation of semantic identity

---

### H2 - Shortcut Bias

JEPA demonstrates reduced reliance on background texture, with higher object-centric dependency scores relative to baselines

---

### H3 - Robustness Geometry

JEPA displays smoother degradation profiles under corruption, whereas baseline models show sharper representational collapse

---

### H4 — Temporal Stability

Across two evaluated video clips, JEPA achieves higher temporal stability scores and lower drift variance than ResNet, DINO, CLIP, and VideoMAE, suggesting stronger short-term identity consistency under constrained temporal evaluation.

---

### H5 - Ambiguity Handling

Under ambiguous conditions, JEPA maintains broader embedding spread, stronger cluster separation, and increased bifurcation dynamics, indicating preservation of competing semantic interpretations

---
### H6 — Data Subsampling

JEPA preserves local neighborhood structure more consistently than baselines, reflected in higher k-NN overlap. 

Other stability metrics exhibit mixed behavior, indicating that JEPA’s advantage is concentrated in local structural robustness rather than universal data efficiency.

---

### H7 - Scaling Geometry 

JEPA demonstrates faster intrinsic dimensionality expansion under scale, consistent with richer latent capacity growth relative to contrastive and CNN baselines

---

### 6. DISCUSSION

Across seven exploratory behavioral probes, JEPA exhibits stronger semantic stability, reduced shortcut reliance, smoother robustness degradation, and more structured responses to ambiguity compared to CNN and contrastive baselines. These patterns align with the hypothesis that predictive learning objectives encourage smoother and more semantically organized latent representations.

However, JEPA does not uniformly outperform baselines across all metrics. In particular, performance under dataset subsampling is mixed, suggesting that representational smoothness does not necessarily imply universal sample efficiency.

Rather than claiming universal superiority, we interpret JEPA’s advantage as structural: it appears to encode semantic information in a more geometry-preserving and object-centric manner under corruption, ambiguity, and scaling regimes. These findings contribute empirical evidence toward understanding how predictive representation learning shapes latent manifold structure beyond accuracy-centric evaluation.

---

### 7. LIMITATIONS 

This study evaluates pretrained models on a small, custom exploratory dataset and limited video samples under constrained compute. As a result, findings should not be interpreted as universal model rankings or large-scale generalization claims.

Larger datasets, broader model variants, and downstream task evaluations may yield different quantitative outcomes.Our composite scoring framework reflects exploratory aggregation rather than a universal ranking metric.

---

### 8. CONCLUSION

We presented a multi-axis behavioral evaluation framework for analyzing Visual JEPA representations and conducted an exploratory empirical study spanning corruption robustness, shortcut bias, temporal stability, ambiguity handling, dataset subsampling, and scaling geometry.

Within our constrained experimental regime, JEPA demonstrates stronger semantic preservation, reduced reliance on spurious background cues, smoother degradation geometry, improved short-term temporal consistency, structured ambiguity responses, and accelerated intrinsic dimensionality expansion. However, gains under dataset subsampling remain mixed, underscoring the distinction between representational stability and data efficiency.

Our results provide empirical evidence that predictive learning objectives shape latent manifold structure in ways that emphasize **semantic coherence, geometric smoothness, and structural robustness**. We view this work as a step toward a broader scientific agenda on behavioral and geometric interpretability of learned representations.


---
### APPENDIX A — Additional Results and Robustness Checks

We provide supplementary plots illustrating per-category performance, corruption sensitivity breakdowns, and failure cases where JEPA does not outperform contrastive baselines.

Additional experiments evaluate sensitivity to corruption severity, embedding normalization strategies, and alternative similarity metrics. These results indicate that while JEPA shows consistent advantages in geometric stability, improvements are not universal across all perturbation regimes.

Future work will expand this appendix with larger-scale ablations, cross-dataset validation, and extended temporal sequences.

