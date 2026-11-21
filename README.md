# PanPPI

**PanPPI (Partner-ranking Alignment Network)** scores protein-protein interactions from sequence alone and is built for screening: each protein is embedded once, then millions of pairs are scored per minute on one GPU. Manuscript: *PanPPI: ranking-trained protein embeddings for proteome-scale interaction screening* (Fremin and Ronda).

## Quick start
From this folder:
```
pip install .
panppi-score --fasta examples/proteins.fasta --pairs examples/pairs.csv --out scores.csv
```
That scores the example pairs and writes `scores.csv`. The first run downloads the model once (2.6 GB, from the Zenodo record [10.5281/zenodo.24407809](https://doi.org/10.5281/zenodo.24407809)) and caches it in `~/.cache/panppi`. A GPU is strongly recommended; the tool also runs on CPU, slowly.

**Your own pairs:** a CSV with columns `a,b` and a FASTA containing every protein (the ID is the first word of each header).
```
panppi-score --fasta proteins.fasta --pairs pairs.csv --out scores.csv
```
**A whole proteome:** score every pair and keep the top 1%.
```
panppi-score --fasta proteome.fasta --all-vs-all --top-fraction 0.01 --out top_pairs.csv
```
**Output:** `protein_a, protein_b, global_logit, maxsim_logit, blend_logit, probability`. Rank by `blend_logit` (the mean of the two calibrated logits); `probability` is its sigmoid.

## Options
| Option | Meaning |
|---|---|
| `--model human` (default) / `bernett` | human = trained on the LoGoPPI human set; bernett = trained on the hub-controlled Bernett split (more conservative scores on new proteins). Seeds: `human_s7`, `bernett_s141`, ... |
| `--weights FILE --calibration FILE` | use your own checkpoint instead of a named model |
| `--top N` / `--top-fraction F` | all-vs-all: how many pairs to keep |
| `--bf16` | embed in bfloat16 (what the benchmark scores in the paper used); default is 32-bit |
| `--device cuda\|cpu` | default: GPU if available |
| `--esm PATH_OR_ID` | ESM-2 650M folder for offline machines (default: Hugging Face `facebook/esm2_t33_650M_UR50D`) |

**Good to know**
* Only the first 800 residues of each protein are used (as in training); the tool tells you how many proteins were truncated. Pairs are symmetric: (A, B) and (B, A) score the same.
* Speed and memory (measured on one RTX A5000 and one A40, all-vs-all on 3,000 human proteins = 4.5 million pairs): about 31,000-34,000 pairs per second for the scoring step, 5-6 minutes in total including model loading, embedding and the built-in self-check, and a peak of 5.8 GB GPU memory. Memory grows with the total number of residues (the embedding cache is about 1 KB per residue); the 19,408-protein human proteome ran on a 24 GB card.
* The 0.5/0.5 blend, calibration and scoring are exactly those of the manuscript. Scores computed in bfloat16 and in 32-bit agree to about 0.01 logits on average (Spearman 0.99995), with the same AUPR.
* Scores are most useful for ranking; as in the paper, precision at a given score depends on how common true pairs are in your candidate set.

## Reproduce the paper
Everything below is for reproducing the manuscript. Trained weights, pair-level inputs and annotation files are archived on Zenodo; `panppi_data.zip` unpacks into the layout of this repository.

| Folder | Contents |
|---|---|
| `panppi/` | the installable tool (`panppi-score`) |
| `src/` | training and research code. `main_e3_bernett.py` trains PanPPI (`--pos-weight 10`, the default, is the human recipe; `--pos-weight 1` was used for the Bernett split). `main_ctrl_t120.py` is the lambda = 0 control. `src/analysis/` has checkpoint scoring, calibration and post-hoc uncertainty |
| `jobs/` | Slurm scripts for human-trained runs, Bernett-trained runs and scoring |
| `human_af_screen/`, `supp/`, `paper/` | evaluation sets, comparator scoring, bootstrap, Figure 1 and 2 builders, supplementary analyses |
| `tests/`, `src/tests/` | unit tests (CPU only) |
| `validation/` | scripts used to validate the tool against the stored scores of the original runs |
| `figures/` | final Figures 1 and 2 |

### Code names
The code uses internal names: **E3a** is the released PanPPI configuration (`--no-latent --no-sn --no-gp`, ranking loss weight 1); **ctrl** is the lambda = 0 control; `tunaplus` is the code base inherited from TUnA, and run folders are named `e3a_nounc_650M_800_s<seed>` (human-trained) and `e3a_bernett_650M_800_s<seed>_h1` (Bernett-trained).

### Quick check (CPU, a few minutes)
```
pip install torch numpy pandas scipy pyyaml
pip install -e .
python tests/test_panppi.py | tail -1          # the tool, with a stand-in encoder
cd src && for t in tests/*.py; do python -I $t | tail -1; done     # research code; no GPU or ESM weights needed
```

### Reproduce the figures
1. Unzip the Zenodo data zip into the root of this repository.
2. `bash set_root.sh "$PWD"` replaces the `<ROOT>` and `<SCRATCH>` path placeholders (and edit `<ACCOUNT>`, `<PARTITION>`, `<user>` in `jobs/`).
3. Run the scripts listed in `REPRODUCE.md`. Figure 2 is rebuilt by `human_af_screen/fig2/build_fig2_v17.py`.

## Notes
* Comparator models (LoGoPPI, PLM-interact, PPLM-PPI, ProteomeLM-PPI, TUnA) are not redistributed; obtain them from their authors. Our derived per-pair scores are in the Zenodo data.
* Training reads the LoGoPPI human and Bernett data formats; download them from their original sources (`data/logoppi/data`, `data/bernett/data`).
* Parts of the Gaussian-process layers derive from TUnA and uncertaintyAwareDeepLearn (MIT; licences in `THIRD_PARTY_LICENSES/`).
* Licence: MIT.
