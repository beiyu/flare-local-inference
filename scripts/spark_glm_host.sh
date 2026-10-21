#!/usr/bin/env bash
# Task-local, native-only launch wrapper. check never initializes CUDA or starts a rank.
set -euo pipefail
if (( $# < 4 )); then
    echo "Usage: $0 RUNTIME_DIR check PACKAGE PUBLISHER_PROFILE" >&2
    echo "       $0 RUNTIME_DIR run PACKAGE PUBLISHER_PROFILE SIGNER_SHA256 NAMESPACE [native options...]" >&2
    exit 2
fi
runtime=$1
mode=$2
package=$3
profile=$4
shift 4
binary="$runtime/native/release/flare"
helper="$(cd -- "$(dirname -- "$0")" && pwd)/spark_glm_service.sh"
case "$(hostname)" in
    spark-01) rank=0; rendezvous=0.0.0.0:29901 ;;
    spark-03) rank=1; rendezvous=10.77.13.1:29901 ;;
    *) echo "This deployment is restricted to the verified spark-01 and spark-03 hosts" >&2; exit 2 ;;
esac
[[ -x "$binary" && -x "$helper" && -r "$package" && -r "$profile" ]] || {
    echo "Native executable, helper, package and public publisher profile must exist" >&2; exit 2;
}
case "$mode" in
    check)
        (( $# == 0 )) || { echo "check takes no execution options" >&2; exit 2; }
        exec "$binary" inspect "$package" --rank "$rank" --publisher-keys "$profile" --json
        ;;
    run)
        (( $# >= 2 )) || { echo "run requires the reviewed signer fingerprint and namespace" >&2; exit 2; }
        fingerprint=$1
        namespace=$2
        shift 2
        [[ "$fingerprint" =~ ^[0-9a-f]{64}$ && -n "$namespace" ]] || {
            echo "Expected the exact SHA256 signing-key fingerprint and publisher namespace" >&2; exit 2;
        }
        auth=()
        # The old upstream uses an empty key file. Preserve that state; never
        # advertise an empty file as authentication or expose key contents.
        if [[ "$rank" == 0 && -r "$runtime/api-key" ]] && LC_ALL=C grep -q '[^[:space:]]' "$runtime/api-key"; then
            auth=(--api-key-file "$runtime/api-key")
        fi
        exec "$helper" "$binary" "$package" "$rank" "$rendezvous" \
            --offline --publisher-keys "$profile" --trust-store "$runtime/service/trust.toml" \
            --dangerously-approve-key "$fingerprint" --dangerously-accept-publisher "$namespace" \
            "${auth[@]}" "$@"
        ;;
    *) echo "MODE must be check or run" >&2; exit 2 ;;
esac
