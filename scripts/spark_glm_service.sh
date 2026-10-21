#!/usr/bin/env bash
# Native package entrypoint. Preparation/export and service handoff happen separately.
set -euo pipefail
if (( $# < 4 )); then
    echo "Usage: $0 FLARE_BINARY PREPARED_PACKAGE RANK RENDEZVOUS [native flare run options...]" >&2
    exit 2
fi
binary=$1
package=$2
rank=$3
rendezvous=$4
shift 4
[[ -x "$binary" && -f "$package" ]] || { echo "Executable and prepared package must exist" >&2; exit 2; }
[[ "$rank" == 0 || "$rank" == 1 ]] || { echo "RANK must be 0 or 1 for this two-Spark deployment" >&2; exit 2; }
# Native Flare verifies package geometry, signatures, GPU compatibility and peer membership.
# It never imports Python or loads the preparation-only EXL3 shared library.
exec "$binary" run "$package" --gpus 0 --host 0.0.0.0 --port 9999 \
    --rank "$rank" --rendezvous "$rendezvous" \
    --served-model-name GLM-5.3-Flash-EXL3 \
    --served-model-name spark/GLM-5.3-Flash-EXL3 \
    --default-stop-scope after_reasoning --default-max-new-tokens 65536 \
    --max-images-per-prompt 20 --mtp-draft-tokens 0 "$@"
