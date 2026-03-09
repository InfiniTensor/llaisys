# 

## road map
SIMD for casting
OpenBlas for gemm

## Known Issues
not fastest, 3 mem passes. However, amending OpenBlas/BLIS is too costly.

## Group casting

### FP16C
only support casting between standardrised IEEE-754 layouts. Not supported here.

### AVX-512
only support some intel chips. More efficient. Might apply later.

### AVX-2
