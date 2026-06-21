VL-JEPA/
│
├── env/                       --> venv
│
├── data/                      --> datasets & test samples
│   ├── raw/ ========>[abstract/ , ambiguous/ , animals/ , humans/ , indoor/ , outdoor/ , objects/ , vehicles/](384 images in raw folder)
│   ├── processed/ (empty folder)
│   ├── synthetic/ (empty folder)
|   └──vidoes/ =======>[video_clip1.mp4 , video_clip2.mp4]            
│
├── models/ sam_vit_b.pth      --> pretrained / checkpoints
│
├── experiments/               --> H1–H6 experiment scripts
│   ├── h1_semantic_faithfulness.py
│   ├── h2_shortcut_bias.py
│   ├── h3_noise_robustness.py
│   ├── h4_temporal_stability.py
│   ├── h5_ambiguity.py
│   └── h6_data_efficiency.py
|   └── h7_scalability.py
|
├──external/ (taken from official github repo of facebook about VL-JEPA )
|
├── metrics/                   --> scoring functions
│   ├── cosine.py
│   ├── robustness.py
│   ├── stability.py
│   └── entropy.py
|   ├── __pycache_(we need to git ignore it)
│
├── plots/                     --> auto-generated graphs
│   ├── h1/
│   ├── h2/
│   ├── h3/
│   └── ...
│
├── results/                   --> experiment outputs (JSON/CSV) ==> [h1_semantic_faithfulness.json, ......]
│
├── notebooks/                 --> exploration notebooks
│   └── analysis.ipynb
|
├── tools/
|   └── build_dataset.py       --> to build dataset via pexels api key 
│
├── utils/                     --> helper functions
│   ├── embeddings.py
│   ├── image_tools.py
│   └── vl-jepa_loader.py
│
├── paper/                     --> research writing
│   ├── figures/
│   ├── tables/
│   ├── draft.md
|   └──paper.tex               --> Latex code for research paper
│
├── README.md                  --> project explanation
├── requirements.txt           --> dependencies
├── .gitignore
└──.env (pexels api key )     ---> we are not supposed to push it 