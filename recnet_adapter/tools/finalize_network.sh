#!/bin/bash
# Finalize a Recnet C-network exploration: aggregate + TS QC + gate + anchors.
#
# Example:
#   finalize_network.sh \
#     --root $R/recnet-runs/pilot \
#     --root $R/recnet-runs/campaign-c2 \
#     --expected-root $R/recnet-runs/campaign-c2 \
#     --report $R/recnet-runs/reports/c2-final-v1 \
#     --anchors $R/validation-pipeline/reaction/paper_network_table.json

set -euo pipefail

R=/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva
TOOLDIR="$R/Recnet/recnet_adapter/tools"
ANCHORS="$R/validation-pipeline/reaction/paper_network_table.json"
COVERAGE=0.95
ROOTS=()
EXPECTED_ROOTS=()
REPORT=""

while (($#)); do
    case "$1" in
        --root)
            [[ -n "${2:-}" ]] || { echo "ERROR: --root needs a path" >&2; exit 2; }
            ROOTS+=("$2"); shift 2 ;;
        --expected-root)
            [[ -n "${2:-}" ]] || { echo "ERROR: --expected-root needs a path" >&2; exit 2; }
            EXPECTED_ROOTS+=("$2"); shift 2 ;;
        --report)
            [[ -n "${2:-}" ]] || { echo "ERROR: --report needs a path" >&2; exit 2; }
            REPORT="$2"; shift 2 ;;
        --anchors)
            [[ -n "${2:-}" ]] || { echo "ERROR: --anchors needs a path" >&2; exit 2; }
            ANCHORS="$2"; shift 2 ;;
        --coverage-threshold)
            [[ -n "${2:-}" ]] || { echo "ERROR: --coverage-threshold needs a number" >&2; exit 2; }
            COVERAGE="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ ${#ROOTS[@]} -gt 0 ]] || { echo "ERROR: at least one --root is required" >&2; exit 2; }
[[ ${#EXPECTED_ROOTS[@]} -gt 0 ]] || EXPECTED_ROOTS=("${ROOTS[@]}")
[[ -n "$REPORT" ]] || { echo "ERROR: --report is required" >&2; exit 2; }
[[ -f "$ANCHORS" ]] || { echo "ERROR: anchors not found: $ANCHORS" >&2; exit 2; }

DPEVA_PYTHON="${DPEVA_PYTHON:-/home/liuzhaoqing/.conda/envs/dpeva-dpa4/bin/python}"
[[ -x "$DPEVA_PYTHON" ]] || { echo "ERROR: runtime python not executable: $DPEVA_PYTHON" >&2; exit 2; }
export PYTHONNOUSERSITE=1

mkdir -p "$REPORT"
export PYTHONPATH="$R/Recnet${PYTHONPATH:+:$PYTHONPATH}"

root_args=()
for root in "${ROOTS[@]}"; do root_args+=(--root "$root"); done
expected_args=()
for root in "${EXPECTED_ROOTS[@]}"; do expected_args+=(--campaign-root "$root"); done
data_args=()
for root in "${ROOTS[@]}"; do data_args+=(--data-root "$root"); done

python_canon() { "$DPEVA_PYTHON" "$@"; }
python_canon "$TOOLDIR/aggregate_network.py" "${root_args[@]}" --out "$REPORT"
python_canon "$TOOLDIR/collect_ts_qc.py" "${root_args[@]}" --out "$REPORT/ts_qc.json"
python_canon "$TOOLDIR/c2_gate.py" \
    "${expected_args[@]}" \
    "${data_args[@]}" \
    --aggregate-json "$REPORT/network_summary.json" \
    --qc-json "$REPORT/ts_qc.json" \
    --out "$REPORT/c2_gate.json" \
    --coverage-threshold "$COVERAGE"
python_canon "$TOOLDIR/audit_missing_attempts.py" \
    "${expected_args[@]}" \
    --gate-json "$REPORT/c2_gate.json" \
    --out "$REPORT/missing_attempt_audit.json"
python_canon "$TOOLDIR/make_network_report.py" \
    --summary "$REPORT/network_summary.json" \
    --anchors "$ANCHORS" \
    --out "$REPORT/paper_anchored_network.md" \
    --title "Fe5C2(510) C2 网络收口报告"

{
    sha256sum \
        "$REPORT/network_summary.json" \
        "$REPORT/network_summary.md" \
        "$REPORT/ts_qc.json" \
        "$REPORT/c2_gate.json" \
        "$REPORT/c2_gate.md" \
        "$REPORT/missing_attempt_audit.json" \
        "$REPORT/paper_anchored_network.md"
} | sort -k2 > "$REPORT/SHA256SUMS"
sha256sum -c "$REPORT/SHA256SUMS"

echo "[finalize] report=$REPORT"
cat "$REPORT/c2_gate.md"
