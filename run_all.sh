#!/bin/bash
cd "$(dirname "$0")" || exit 1
for step in index.py index_stats.py phase2.py phase3.py phase4a.py hyde_generate.py phase4b.py phase5.py make_tables.py; do
    echo "=== $step $(date) ===" | tee -a logs/run_all.log
    python $step 2>&1 | grep -vE "WARN|INFO|warn" | tee -a logs/run_all.log
done
echo "=== done $(date) ===" | tee -a logs/run_all.log
