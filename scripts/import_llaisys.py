import sys
sys.path.insert(0, 'python')
import llaisys
print('import ok')
print('has qwen create', hasattr(llaisys.libllaisys.LIB_LLAISYS, 'llaisysQwen2ModelCreate'))
