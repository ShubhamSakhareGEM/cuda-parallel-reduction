#!/usr/bin/env bash

# full Nsight Compute profiling script

set -euo pipefail

BIN=./reduction
REPORT=profile_report


echo "════════════════════════════════════════════"
echo "  Building..."
echo "════════════════════════════════════════════"
make clean && make

echo ""
echo "════════════════════════════════════════════"
echo "  Running baseline benchmark"
echo "════════════════════════════════════════════"
$BIN

echo ""
echo "════════════════════════════════════════════"
echo "  Nsight Compute profiling"
echo "  (requires sudo or relaxed perf permissions)"
echo "════════════════════════════════════════════"



ncu \
  --metrics \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
sm__occupancy_pct.avg,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__cycles_active.avg \
  --launch-skip 3 \
  --launch-count 1 \
  --target-processes all \
  -o "$REPORT" \
  $BIN




echo ""
echo "Done. Open ${REPORT}.ncu-rep in the Nsight Compute GUI"
echo "  or run: ncu-ui ${REPORT}.ncu-rep"