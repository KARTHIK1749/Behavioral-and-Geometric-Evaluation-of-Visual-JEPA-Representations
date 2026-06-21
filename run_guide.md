# Run Guide

This repository is structured so the experiments can be reproduced from a clean checkout.

## 1. Recommended environment

Use Python 3.10 in a fresh virtual environment. On Windows PowerShell:

```powershell
py -3.10 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

## 2. Required local files

Before running the experiments, make sure these paths exist:

- `models/sam_vit_b.pth`
- `external/vl-jepa/`
- `data/raw/` with the curated image categories
- `data/videos/` if you want to reproduce the temporal experiments exactly

The model loader in `utils/embeddings.py` imports the vendored VL-JEPA source directly from `external/vl-jepa/`, so you do not need to install that subdirectory as a separate package.

## 3. Optional dataset regeneration

If you want to rebuild the curated image dataset from Pexels, create a `.env` file in the project root with your API key:

```env
PEXELS_API_KEY=your_key_here
```

Then run:

```powershell
python tools\build_dataset.py
```

This script will populate `data/raw/` by category. If you are only reproducing the published experiments, you can skip this step and use the existing dataset.

## 4. Run the experiments

Each hypothesis is a standalone script under `experiments/`.

Run them individually:

```powershell
python experiments\h1_semantic_faithfulness.py
python experiments\h2_shortcut_bias.py
python experiments\h3_noise_robustness.py
python experiments\h4_temporal_stability.py
python experiments\h5_ambiguity.py
python experiments\h6_data_efficiency.py
python experiments\h7_scalability.py
```

Or run them in sequence from PowerShell:

```powershell
$experiments = @(
  'experiments\h1_semantic_faithfulness.py',
  'experiments\h2_shortcut_bias.py',
  'experiments\h3_noise_robustness.py',
  'experiments\h4_temporal_stability.py',
  'experiments\h5_ambiguity.py',
  'experiments\h6_data_efficiency.py',
  'experiments\h7_scalability.py'
)

foreach ($script in $experiments) {
  python $script
}
```

## 5. Outputs

Successful runs write results to:

- `results/` for JSON or CSV metrics
- `plots/` for generated figures

The paper source lives in `paper/paper.tex`, and the optional notebook analysis is in `notebook/analysis.ipynb`.

## 6. Reproducibility notes

- Keep `env/`, `.env`, `data/raw/`, `data/videos/`, `data/processed/`, `results/`, `plots/`, and `models/*.pth` out of Git.
- Regenerate the dataset before rerunning experiments if you change the Pexels sourcing logic.
- Run the experiment scripts in a stable order if you want the output folders to line up with the paper tables and figures.

## 7. Quick sanity check

If you only want to verify the setup, run one lightweight script first, such as:

```powershell
python experiments\h1_semantic_faithfulness.py
```

If it completes and writes `results/h1_semantic_faithfulness.json`, the core pipeline is working.
