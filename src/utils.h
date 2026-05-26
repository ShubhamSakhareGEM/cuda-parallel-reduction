#pragma once
#include <stdio.h>

#include <stdlib.h>
#include <cuda_runtime.h>


#include <time.h>


// ─── CUDA error checking ──────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        
    do {                                                                        
        cudaError_t _err = (call);                                              
        if (_err != cudaSuccess) {                                              
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                        
                    __FILE__, __LINE__, cudaGetErrorString(_err));              
            exit(EXIT_FAILURE);                                                 
        }                                                                       
    } while (0)

// ─── Timing helpers ───────────────────────────────────────────────────────────
typedef struct { cudaEvent_t start, stop; } GpuTimer;


static inline void gpuTimerStart(GpuTimer* t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
    CUDA_CHECK(cudaEventRecord(t->start));
}



static inline float gpuTimerStop(GpuTimer* t) {
    float ms;
    CUDA_CHECK(cudaEventRecord(t->stop));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
    return ms;
}

static inline double cpuTimerMs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
// ─── correctness check ────────────────────────────────────────────────────────
static inline void verify(float ref, float got, const char* label, float tol) {
    float err = fabsf(ref - got) / (fabsf(ref) + 1e-6f);
    if (err < tol)
        printf("  ✓  %-28s  result=%.4f  (rel err=%.2e)\n", label, got, err);
    else
        printf("  ✗  %-28s  MISMATCH ref=%.4f got=%.4f  (rel err=%.2e)\n",
               label, ref, got, err);
}

// ─── pretty divider ──────────────────────────────────────────────────────────
static inline void banner(const char* msg) {
    printf("\n══════════════════════════════════════════════\n");
    printf("  %s\n", msg);
    printf("══════════════════════════════════════════════\n");
}