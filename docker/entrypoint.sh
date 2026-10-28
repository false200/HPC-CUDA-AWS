#!/bin/bash
# Pull a job from S3 when JOB_S3 is set, run the CUDA engine, push the JSON back.
set -euo pipefail
JOB="${JOB_PATH:-/work/job.txt}"
OUT="${OUT_PATH:-/work/result.json}"
mkdir -p "$(dirname "$JOB")" "$(dirname "$OUT")"
if [ -n "${JOB_S3:-}" ]; then
  aws s3 cp "$JOB_S3" "$JOB"
fi
/opt/mc_engine run "$JOB" "$OUT"
if [ -n "${OUT_S3:-}" ]; then
  aws s3 cp "$OUT" "$OUT_S3"
fi
