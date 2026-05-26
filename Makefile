# ─────────────────────────────────────────────────────────────────────────────
# makefile for cuda-parallel-reduction
#   target= Linux + NVIDIA GPU 
# ─────────────────────────────────────────────────────────────────────────────

NVCC      := nvcc
CUDA_ARCH := sm_75          
                            

NVCCFLAGS := -arch=$(CUDA_ARCH) \
             -O3              \
             -lineinfo        \
             -Xcompiler -Wall \
             -Xcompiler -O3

SRC_DIR := src
BIN     := reduction

SRCS := $(SRC_DIR)/reduction.cu


.PHONY: all clean run profile

all: $(BIN)

$(BIN): $(SRCS) $(SRC_DIR)/utils.h
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)
	@echo "Build complete → ./$(BIN)"

run: $(BIN)
	./$(BIN)



profile: $(BIN)
	ncu \
	  --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
sm__occupancy_pct.avg,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
	  --target-processes all \
	  --launch-skip 0 \
	  --launch-count 3 \
	  -o profile_report \
	  ./$(BIN)
	@echo "Profile saved → profile_report.ncu-rep  (open in Nsight Compute GUI)"



clean:
	rm -f $(BIN) profile_report.ncu-rep