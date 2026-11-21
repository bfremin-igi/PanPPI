# Reproducing the manuscript

Scripts are the versions that produced the manuscript numbers, with paths replaced by placeholders (see `set_root.sh`). Cluster jobs (training, comparator scoring) need a Slurm GPU system; the analyses downstream of the pair-level scores run on a laptop.

## Training and scoring (cluster)
| Step | Script(s) |
|---|---|
| PanPPI, human data, seeds 42/7/141 | `jobs/train_human/j_e3a_*.sbatch` -> `src/main_e3_bernett.py` |
| Control (lambda = 0), seeds 42/7/141 | `jobs/train_human/j_ctrl_*.sbatch` -> `src/main_ctrl_t120.py` |
| PanPPI, Bernett split, seeds 42/7/141 | `jobs/bernett/j_e3a_bernett_*.sbatch` |
| Calibrated blend scores | `jobs/train_human/j_blend_score*.sbatch` -> `src/analysis/score_ckpt_e3.py` |
| Post-hoc GP / familiarity features | `src/analysis/posthoc_uncertainty*.py`, `jobs/bernett/j_posthoc_*`, `paper/unc_*.py` |

## Figures and tables (laptop, from the Zenodo data)
| Item | Scripts |
|---|---|
| Fig. 1, Supp. Table 1 | `paper/bootstrap_fig1*.py`, `paper/boot_lib.py`, `human_af_screen/fig1/`, `human_af_screen/xsp_bern/` |
| Fig. 2a-c, Supp. Table 2 | `human_af_screen/fig2/retention_extended.py`, `retention_ci.py`, `precision_ci.py`, `precision_curve.py` |
| Fig. 2d-g, Supp. Tables 4, 7 | `human_af_screen/fig2/contact_*.py`, `func_enrich_clean.py`, `find_examples.py` |
| Fig. 2 builder | `human_af_screen/fig2/build_fig2_v17.py` |
| Evaluation sets | `human_af_screen/bern_univ/*.py` |
| Supp. Figs. 1-7, Notes 1-4 | `supp/*.py`, `paper/unc_*.py` |
| Supp. Fig. 8 (Fig. 2 extras) | `human_af_screen/fig2/supp2/supp2_compute.py`, `supp2_figure.py` |
| Supp. Fig. 9 (five further species) | `human_af_screen/newsp/` (`newsp_*.py`, `supp9_v2.py`; result tables in `results/`) |
| Supp. Tables 3, 14-16 | `human_af_screen/fig2/precision_ci.py`, `retention_hub_trim.py`, `no_trim_reweight.py`, `retention_by_string.py` |

The evaluation-set builders in `human_af_screen/bern_univ/` were run on eight species. The inputs for the two species that the paper does not use (*M. tuberculosis*, *P. aeruginosa*) are not archived, and the stored result tables keep their rows. Every figure and table in the paper is rebuilt from the archived data for human, yeast, mouse, fly, worm and *E. coli*.

## Weights
Six checkpoints (human-trained seeds 42/7/141; Bernett-trained seeds 42/7/141) are in the Zenodo record as `weights_<human|bernett>_s<seed>_best.pt` with calibration and run configuration. The lambda = 0 control checkpoints are not archived.
