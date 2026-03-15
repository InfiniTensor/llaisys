# Project 1: Linear OP Performance Optimization (CPU)

## Benchmark
```
(venv) scbz@dsw-607126-85f54bdf75-5lzlx:~/llaisys$ OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 python test/ops/linear.py --profile
Testing Ops.linear on cpu
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <f32>
        Torch time: 0.00374 ms 
        LLAISYS time: 0.00158 ms
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <f16>
        Torch time: 0.01351 ms 
        LLAISYS time: 0.00374 ms
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <bf16>
        Torch time: 0.01428 ms 
        LLAISYS time: 0.00366 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
        Torch time: 11.43182 ms 
        LLAISYS time: 17.31473 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f16>
        Torch time: 72.20329 ms 
        LLAISYS time: 20.56061 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <bf16>
        Torch time: 38.64677 ms 
        LLAISYS time: 19.27763 ms
Test passed!
```

## Main changes (this branch)
- **Aligned allocation + memcpy reduction:** `src/device/cpu/cpu_runtime_api.cpp` now allocates aligned buffers and skips memcpy for `float`/`double` when inputs are already aligned.
- **Vectorized cast helpers:** `src/ops/utils.hpp` adds faster casting paths (SIMD-friendly) with alignment checks.
- **GEMM backend:** `linear_cpu.cpp` now prefers OpenBLAS for `fp32/fp16` when inputs are aligned, avoiding unnecessary copy stage.

## Roadmap (next improvements)
- SIMD casting path for `bf16` / full `f16` pipeline.
- Reduce memory passes (aim for 1-2 passes instead of 3) by avoiding extra temporary buffers.
- Add runtime dispatch for AVX2/AVX-512 (and fallback for AMD) to maximize portability and performance.
- Standardize benchmarking on a dedicated machine (remove background work) to get stable numbers.

## Known issues / caveats
1. Current implementation still uses multiple memory passes; performance isn't yet optimal.
2. `bf16` is still slower than `fp16`/`fp32` due to casting overhead.
3. Benchmark host is shared; results may vary with background load.
4. Recommended tuning: `OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32` (not fully validated).

## CPU ISA (benchmark host)
```
(venv) scbz@dsw-607126-85f54bdf75-5lzlx:~/llaisys$ cat /proc/cpuinfo | grep flags | head -1
flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves wbnoinvd avx512vbmi umip pku avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear arch_capabilities
```

- Supports AVX2 + F16C.
- AVX-512 is available on this host but is avoided for broader AMD compatibility.
