#!/bin/bash
# Replace the path placeholders in scripts with your own locations. Usage: bash set_root.sh /abs/path/to/this/repo [/abs/path/to/cluster/scratch]
set -euo pipefail
ROOT="${1:?usage: bash set_root.sh /abs/path/to/repo [scratch]}"; SCR="${2:-$ROOT/scratch}"
grep -rIl -e '<ROOT>' -e '<SCRATCH' . --include='*.py' --include='*.sbatch' --include='*.sh' --include='*.md' --include='*.txt' --include='*.json' | while read f; do
  sed -i.bak -e "s#<ROOT>#$ROOT#g" -e "s#<SCRATCH_HOME>#$SCR#g" -e "s#<SCRATCH>#$SCR#g" "$f" && rm -f "$f.bak"; done
echo "ROOT=$ROOT SCRATCH=$SCR. Still edit <ACCOUNT>, <PARTITION> and <user> in jobs/*.sbatch for your cluster."
