import sys
sys.path.insert(0, 'python')
import llaisys.libllaisys as libmod
lib = libmod.LIB_LLAISYS
print('loaded lib handle:', getattr(lib, '_name', None), lib._handle)
print('has create:', hasattr(lib, 'llaisysQwen2ModelCreate'))
try:
    f = lib.llaisysQwen2ModelCreate
    print('callable repr:', f)
except Exception as e:
    print('error getting create:', e)

print('available names via dir (sample):')
names = [n for n in dir(lib) if 'Qwen2' in n or 'tensorCreate' in n]
print(names)
