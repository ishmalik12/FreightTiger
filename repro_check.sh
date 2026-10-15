#!/usr/bin/env bash
# Three untouched runs on the same input; every output file must be byte-identical.
# PYTHONHASHSEED is varied on purpose to prove nothing depends on hash/dict ordering.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf outputs/repro && mkdir -p outputs/repro
for i in 1 2 3; do
  PYTHONHASHSEED=$((i * 1234)) python3 run.py --out "outputs/repro/run$i" > /dev/null
done
status=0
for f in flagged_routes.csv weekly_metrics_all.csv cost_log.json; do
  h1=$(sha256sum "outputs/repro/run1/$f" | cut -d' ' -f1)
  for i in 2 3; do
    h=$(sha256sum "outputs/repro/run$i/$f" | cut -d' ' -f1)
    if [ "$h" != "$h1" ]; then echo "MISMATCH $f run1 vs run$i"; diff "outputs/repro/run1/$f" "outputs/repro/run$i/$f" | head; status=1; fi
  done
  echo "$f  sha256=$h1  (runs 1,2,3 identical: $([ $status -eq 0 ] && echo yes || echo NO))"
done
exit $status
