"""Build PanPPI_supplementary_data.xlsx (Supplementary Data 1): the long supplementary tables as numeric spreadsheets, one sheet per table.
Sources: bootstrap_fig1_with_auroc.csv (Supp Table 1), ../human_af_screen/fig2/retention_extended.csv (Supp Table 2), ../human_af_screen/fig2/candidate_examples_table.csv (Supp Table 13)."""
import os, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
HERE = os.path.dirname(os.path.abspath(__file__)); F2 = HERE + "/../human_af_screen/fig2/"
wb = Workbook(); ws = wb.active; ws.title = "README"
for r in (["PanPPI: ranking-trained protein embeddings enable proteome-scale interaction screening"], ["Supplementary Data 1. Long supplementary tables (one sheet per table; the captions are in the Supplementary Information)."], [],
          ["Sheet", "Content"], ["Supplementary Table 1", "Paired bootstrap comparisons for Figure 1 (AUPR with 95% interval; AUROC at full data)."],
          ["Supplementary Table 2", "Retention of AFM-confident pairs in the top 0.01%, 0.1%, 1% and 10% of all proteome pairs, with pairs sent to AFM per AFM-confident pair."],
          ["Supplementary Table 13", "All human direct-contact pairs that satisfy the example-selection rule (Methods, 'Selection of examples'), including the two pairs shown in Figure 2f,g."], ["Supplementary Table 14", "Retention of AFM-confident pairs after removing pairs that involve the most-connected proteins (hub control for Figure 2b)."], ["Supplementary Table 15", "Figure 2c sets without trimming: all STRING positives kept, negatives reweighted by degree."], ["Supplementary Table 16", "Retention of AFM-confident pairs by STRING support."], ["Supplementary Table 17", "Models compared: checkpoints and training data."]): ws.append(r)
ws["A1"].font = Font(bold=True); ws["A4"].font = Font(bold=True); ws["B4"].font = Font(bold=True)
def sheet(name, df, widths=None):
    s = wb.create_sheet(name); s.append(list(df.columns))
    for row in df.itertuples(index=False): s.append([None if (isinstance(v, float) and v != v) else v for v in row])
    for c in s[1]: c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="top")
    s.freeze_panes = "A2"
    for i, col in enumerate(df.columns, 1): s.column_dimensions[get_column_letter(i)].width = max(10, min(34, max(len(str(col)) * 0.9, df[col].astype(str).str.len().max() * 1.1)))
    return s
T1 = pd.read_csv(HERE + "/bootstrap_fig1_with_auroc.csv")
T1 = T1.rename(columns={"panel": "Panel", "set": "Set", "tool": "Comparator", "aupr_tool": "Comparator AUPR", "aupr_panppi": "PanPPI AUPR", "diff": "AUPR difference (PanPPI minus comparator)", "lo": "95% interval, lower", "hi": "95% interval, upper", "frac_positive": "Share of resamples with positive difference", "boot_mean_diff": "Bootstrap mean difference", "auroc_tool": "Comparator AUROC", "auroc_panppi": "PanPPI AUROC", "n_pairs": "Pairs", "n_pos": "Positives"})
panel_names = {"b": "D-SCRIPT species, human-trained (Fig. 1b)", "c": "Bernett test set (Fig. 1c)", "d": "Bernett-style cross-species sets (Fig. 1d)"}; T1["Panel"] = T1["Panel"].map(panel_names)
cols = ["Panel", "Set", "Pairs", "Positives", "Comparator", "Comparator AUPR", "PanPPI AUPR", "AUPR difference (PanPPI minus comparator)", "95% interval, lower", "95% interval, upper", "Share of resamples with positive difference", "Comparator AUROC", "PanPPI AUROC"]
sheet("Supplementary Table 1", T1[cols].round(4))
X = pd.read_csv(F2 + "retention_extended.csv"); X = X[(X.set != "Human (all)") & X.fraction.isin([1e-4, 1e-3, 1e-2, 1e-1])]
X = X.rename(columns={"set": "Set", "n_afm_confident": "AFM-confident pairs", "universe_pairs": "All pairs of the proteome", "fraction": "Top fraction", "pairs_sent": "Pairs sent to AFM", "PanPPI blend": "PanPPI retention", "PanPPI blend lo": "PanPPI lower", "PanPPI blend hi": "PanPPI upper", "LoGoPPI": "LoGoPPI retention", "LoGoPPI lo": "LoGoPPI lower", "LoGoPPI hi": "LoGoPPI upper", "diff Pan-Lg": "PanPPI minus LoGoPPI", "diff lo": "Difference lower", "diff hi": "Difference upper", "pairs_per_hit_pan": "Pairs per AFM-confident hit, PanPPI", "pairs_per_hit_random": "Pairs per hit, random selection"})
cols2 = ["Set", "AFM-confident pairs", "All pairs of the proteome", "Top fraction", "Pairs sent to AFM", "PanPPI retention", "PanPPI lower", "PanPPI upper", "LoGoPPI retention", "LoGoPPI lower", "LoGoPPI upper", "PanPPI minus LoGoPPI", "Difference lower", "Difference upper", "Pairs per AFM-confident hit, PanPPI", "Pairs per hit, random selection"]
order = {"Human (new pairs)": 0, "Yeast": 1, "Mouse": 2, "Fly": 3, "Worm": 4, "E. coli": 5}; X = X.assign(_o=X.Set.map(order)).sort_values(["_o", "Top fraction"])
sheet("Supplementary Table 2", X[cols2].round(4))
T13 = pd.read_csv(F2 + "candidate_examples_table.csv").rename(columns={"gene_a": "Gene A", "gene_b": "Gene B", "a": "UniProt A", "b": "UniProt B", "pdb_entry": "PDB entry (largest interface)", "chains_in_entry": "Chains in entry", "interface_residues_a": "Interface residues, A", "interface_residues_b": "Interface residues, B", "n_qualifying_entries": "Qualifying PDB entries", "string_physical": "STRING physical score", "afm_ipTM": "AFM ipTM", "afm_ipSAE": "AFM ipSAE", "panppi_percentile_among_negatives": "PanPPI percentile among negatives", "panppi_calls_of_3": "PanPPI calls (of 3 replicates)", "logoppi_percentile": "LoGoPPI percentile among negatives", "logoppi_calls_of_3": "LoGoPPI calls (of 3 replicates)", "proteomelm_percentile": "ProteomeLM-PPI percentile among negatives", "overlap_stratum": "Training-overlap stratum (1-5)", "new_to_panppi": "New to PanPPI (stratum 3-5)"})
sheet("Supplementary Table 13", T13)
H14 = pd.read_csv(F2 + "retention_hub_trim.csv").rename(columns={"set": "Set", "hub_fraction_removed": "Share of most-connected proteins removed", "hub_degree_cutoff": "Degree cut-off (STRING-linked candidate partners)", "n_afm_confident_left": "AFM-confident pairs remaining", "fraction": "Top fraction", "PanPPI": "PanPPI retention", "LoGoPPI": "LoGoPPI retention", "diff": "PanPPI minus LoGoPPI", "diff_lo": "Difference lower", "diff_hi": "Difference upper"})
sheet("Supplementary Table 14", H14.round(4))
N15 = pd.read_csv(F2 + "no_trim_reweight.csv").rename(columns={"species": "Species", "n_pos": "STRING positives (all kept)", "n_neg": "Low-confidence candidate negatives", "tool": "Tool", "aupr_reweighted": "AUPR, negatives degree-reweighted", "aupr_unweighted_raw": "AUPR, no degree control", "degree_only_auroc_after_weighting": "Degree-only AUROC after weighting"})
sheet("Supplementary Table 15", N15.round(4))
S16 = pd.read_csv(F2 + "retention_by_string.csv").rename(columns={"set": "Set", "string_tier": "STRING physical score", "n_afm_confident": "AFM-confident pairs", "fraction": "Top fraction", "PanPPI": "PanPPI retention", "PanPPI_lo": "PanPPI lower", "PanPPI_hi": "PanPPI upper", "LoGoPPI": "LoGoPPI retention", "LoGoPPI_lo": "LoGoPPI lower", "LoGoPPI_hi": "LoGoPPI upper", "diff": "PanPPI minus LoGoPPI"})
sheet("Supplementary Table 16", S16.round(4))
C17 = pd.DataFrame([
 ["PanPPI", "human-trained (seeds 42, 7, 141)", "trained by us", "LoGoPPI human set (421,312 pairs, 38,267 positive, 1:10)", "this work"],
 ["PanPPI", "Bernett-trained (seeds 42, 7, 141)", "trained by us", "Bernett training split (163,192 pairs, 1:1)", "this work"],
 ["LoGoPPI", "Cross-species", "released LoGoPPI-Cross-species checkpoint, authors' inference script", "LoGoPPI human set (47,932 positive pairs, 80/20 training/validation, negatives 10:1)", "ref. 7"],
 ["LoGoPPI", "Bernett", "released Bernett-trained checkpoint", "Bernett training split", "ref. 7"],
 ["PPLM-PPI", "cross-species", "released weights", "D-SCRIPT human (H. sapiens) set; 10-fold cross-validation, best-AUPRC checkpoint", "ref. 6"],
 ["PLM-interact", "human", "released human checkpoint (humanV11)", "D-SCRIPT human pairs (421,792)", "ref. 5"],
 ["PLM-interact", "Bernett", "released Bernett-trained checkpoint", "Bernett training split (80% used by its authors)", "ref. 5"],
 ["ProteomeLM-PPI", "cross-species", "released ProteomeLM-S backbone with the author-bundled supervised head (not retrained)", "backbone self-supervised on proteomes; head trained on the D-SCRIPT human set, as bundled by the authors", "ref. 14"],
 ["TUnA", "human", "released checkpoints (seeds 47, 94, 141)", "D-SCRIPT human set", "ref. 8"],
 ["TUnA", "Bernett", "released Bernett-trained checkpoint", "Bernett training split", "ref. 8"]], columns=["Tool", "Setting", "Checkpoint used", "Training data (as stated by its authors)", "Source"])
sheet("Supplementary Table 17", C17)
out = HERE + "/PanPPI_supplementary_data.xlsx"; wb.save(out); print("wrote", out, [(s.title, s.max_row - 1) for s in wb.worksheets[1:]])
