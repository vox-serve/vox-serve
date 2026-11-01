#!/bin/bash
# Usage: smoke_all.sh <model>...: runs smoke_model.sh for each model, continuing past failures.
for m in "$@"; do
  echo "######## $m"
  bash rust/scripts/smoke_model.sh "$m" || echo "######## FAILED $m"
done
