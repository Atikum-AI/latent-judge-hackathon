#!/bin/zsh
# precut.sh <step> shoot_<date> [opts] — run one pre-cut agent step with the lab python (numpy + requests).
# steps: ingest | takes | sync | dialogue | label | plan | report | all | calls
#   --no-transcribe / --no-spark on the takes/dialogue/label steps: offline dry run, no Meta API calls.
#   --replay on dialogue/label: reuse saved answers from precut_cache/, no API call.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(cd "$(dirname "$0")" && pwd)/.venv/bin/python"   # its own venv if present (scenedetect, numpy, requests, pillow)
[[ -x "$PY" ]] || PY="${PRECUT_PYTHON:-$(command -v python3)}"   # else PRECUT_PYTHON or python3 on PATH
[[ -x "$PY" ]] || { echo "missing lab venv python: $PY" >&2; exit 1; }

step="${1:-}"; shift || true
case "$step" in
  ingest|takes|sync|dialogue|label|plan|report) exec "$PY" -m "precut.$step" "$@" ;;
  cloud) exec "$PY" -m precut.cloud_watch "$@" ;;   # watch shoot_<date>/cloud_sync/ (Blackmagic Cloud via Resolve)
  test) exec "$PY" -m pytest "$@" ;;                # the unit suite (no network, no spend)
  calls) exec "$PY" -c "import sys; from precut.metaapi import call_report; print(call_report(sys.argv[1]))" "$@" ;;
  all)
    dir="${1:?usage: ./precut.sh all shoot_<date> [--offline]}"; shift
    if [[ " $* " == *" --offline "* ]]; then OFF_T="--no-transcribe"; OFF_L="--no-spark"; else OFF_T=""; OFF_L=""; fi
    "$PY" -m precut.ingest "$dir"
    "$PY" -m precut.takes "$dir" $OFF_T
    "$PY" -m precut.sync "$dir"
    "$PY" -m precut.dialogue "$dir" $OFF_T
    "$PY" -m precut.label "$dir" $OFF_L
    "$PY" -m precut.plan "$dir"
    "$PY" -m precut.report "$dir"
    ;;
  *) echo "usage: ./precut.sh {ingest|takes|sync|dialogue|label|plan|report|all|cloud|test|calls} shoot_<date> [opts]" >&2; exit 2 ;;
esac
