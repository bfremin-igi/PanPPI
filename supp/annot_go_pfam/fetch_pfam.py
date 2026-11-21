"""Fetch Pfam cross-references from the UniProt REST API for the Bernett test proteins (as LoGoPPI did for its hard-negative analysis). Reads the protein accessions from the Bernett test scores; writes pfam_bernett.tsv (Entry, Pfam)."""
import sys, time, urllib.request, urllib.parse, pandas as pd
d = pd.read_csv("../../paper/boot_inputs/runs_ft/e3a_bernett_650M_800_s42_h1/scores_bernett.tsv", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b"]); prots = sorted(set(d.a) | set(d.b)); print(len(prots), "proteins")
out = []; head = None
for i in range(0, len(prots), 80):
    q = "(" + " OR ".join(f"accession:{p}" for p in prots[i:i + 80]) + ")"
    url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode({"query": q, "fields": "accession,xref_pfam", "format": "tsv"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: txt = r.read().decode()
            break
        except Exception as e: print("retry", i, e); time.sleep(3)
    else: sys.exit("failed at chunk %d" % i)
    lines = txt.strip().split("\n"); head = lines[0]; out += lines[1:]; time.sleep(0.3)
open("pfam_bernett.tsv", "w").write(head + "\n" + "\n".join(out) + "\n"); print("rows", len(out), "header", head)
