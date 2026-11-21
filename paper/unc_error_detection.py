"""Does the post-hoc GP variance flag the model's mistakes? Error = misclassification of the calibrated blend at probability 0.5. Reported per set and seed: AUROC of the variance for predicting errors, AUROC of low confidence (-|logit|) for predicting errors,
Spearman-free Pearson correlation of variance with |logit|. Standard files (human-trained model), Bernett test, Bernett-style cross-species sets (Bernett-trained model). Output unc_error_detection.csv."""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU
HERE = os.path.dirname(os.path.abspath(__file__)); UI = HERE + "/unc_inputs/runs_ft/"; rows = []
RUN = {"42": "e3a_nounc_650M_800_s42_r2", "7": "e3a_nounc_650M_800_s7", "141": "e3a_nounc_650M_800_s141_clean2"}
def one(path):
    d = pd.read_csv(path, sep="\t", usecols=["label", "cal_blend", "var"]); err = ((d.cal_blend > 0) != (d.label == 1)).astype(int).to_numpy(); v = d["var"].to_numpy(); c = np.abs(d.cal_blend.to_numpy())
    return AU(err, v), AU(err, -c), float(np.corrcoef(v, c)[0, 1]), float(err.mean()), len(d)
for s in RUN:
    for sp in ["yeast", "mouse", "fly", "worm", "ecoli"]:
        a, b, c, e, n = one(UI + f"{RUN[s]}/posthoc_gp/scores_{sp}_posthoc.tsv"); rows.append(dict(setting="standard", seed=s, set=sp, auroc_var=a, auroc_lowconf=b, corr_var_absLogit=c, error_rate=e, n=n))
    B = UI + f"e3a_bernett_650M_800_s{s}_h1/"
    a, b, c, e, n = one(B + "posthoc_bernett/scores_bernett_posthoc.tsv"); rows.append(dict(setting="bernett_test", seed=s, set="bernett", auroc_var=a, auroc_lowconf=b, corr_var_absLogit=c, error_rate=e, n=n))
    for sp in ["yeast", "mouse", "fly", "worm", "ecoli"]:
        a, b, c, e, n = one(B + f"posthoc_bernett_xsp/scores_{sp}_posthoc.tsv"); rows.append(dict(setting="bernett_style", seed=s, set=sp, auroc_var=a, auroc_lowconf=b, corr_var_absLogit=c, error_rate=e, n=n))
R = pd.DataFrame(rows); R.round(4).to_csv(HERE + "/unc_error_detection.csv", index=False)
print(R.groupby(["setting", "set"])[["auroc_var", "auroc_lowconf", "corr_var_absLogit", "error_rate"]].mean().round(3).to_string())
