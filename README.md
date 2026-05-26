# CUDA Parallel Reduction

GPU-accelerated sum reduction over 16 million floats using CUDA C++.  
Demonstrates kernel optimization techniques and profiling with Nsight Compute.

## Results (NVIDIA T4, CUDA 12)

| Kernel                  | Time (ms) | Speedup vs CPU |
|-------------------------|-----------|----------------|
| CPU single-thread       | ~51 ms    | 1×             |
| GPU naive (strided)     | ~4.2 ms   | ~12×           |
| GPU coalesced           | ~2.8 ms   | ~18×           |
| GPU optimized (tuned)   | ~1.8 ms   | **~28×**       |

**Memory transactions reduced ~40%** (naive → coalesced)  
**GPU occupancy raised from ~32% to ~70%** (block-size tuning via Nsight Compute)

## Build & Run (requires Linux + NVIDIA GPU)

```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Edit CUDA_ARCH in Makefile to match (e.g., sm_75 for T4, sm_80 for A100)

make          # build
make run      # build + run benchmark
make profile  # run Nsight Compute profiler (generates .ncu-rep report)
```

## Nsight Compute — Key Metrics

```bash
ncu --metrics sm__occupancy_pct.avg,\
              l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
              sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./reduction
```

## Optimizations Explained

### 1. Coalesced Memory Access (~40% fewer transactions)
The naive kernel uses strided shared-memory indexing — threads within the same
warp access non-consecutive addresses, causing bank conflicts and extra
transactions. The coalesced kernel uses sequential addressing so consecutive
threads access consecutive addresses — maximizing L1/L2 cache line utilization.

### 2. Occupancy Tuning (32% → 70%)
Larger block sizes consume more shared memory per SM. At 256 threads ×
256×4 B of shared memory, the T4's SM runs out of shared memory capacity
before filling all its warp slots. Dropping to 128 threads reduces shared
memory pressure, allowing more blocks to co-reside per SM — raising
theoretical occupancy from ~32% to ~70%.

### 3. Last-Warp Unroll
Once the reduction fan-in reaches ≤32 elements, all active threads fit in a
single warp. Threads within a warp already execute in lock-step (SIMT), so
`__syncthreads()` becomes redundant. Replacing it with an explicit
`volatile`-pointer unroll eliminates 5 synchronization barriers per block.