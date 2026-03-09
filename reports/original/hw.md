什么时候连续?

1. shape 和 strides 什么形状? 
    // see line 21
    // strides' dim = shape' dim 
    // e.g. shape is of [3,2]
    // strides should be [2,1] (when cont)

    // when a tensor is contiguous?
    // 1. obv, there must be a stride = 1;
    // 2. stides[i] = strides[i+1] * shapes[i+1];

2.  什么时候连续?

连续是指内存的连续布局

举例而言
内存布局
位置: 0    1    2    3    4    5
值:   [0]  [1]  [2]  [3]  [4]  [5]
逻辑视图:
[[0, 1, 2],   // 行0
 [3, 4, 5]]   // 行1

形状 (shape): (2, 3)
步长 (strides): (3, 1)

数学化的表示 stride[i] = stride[i+1] * shape[i+1] 就连续了.
例如这里 3 = 3 * 1.

现在, 我们对张量进行转置:

[[0, 3],   // 新行0（原来列0）
 [1, 4],   // 新行1（原来列1）
 [2, 5]]   // 新行2（原来列2）

形状 (shape): (3, 2) – 原来2行3列，现在3行2列。
步长 (strides): (1, 3) – 原来 (3, 1) 交换后变为 (1, 3)，因为转置不改变内存布局，只改变访问方式。
总元素数: 仍为6，内存布局不变。

!!!note "转置可以直接转置 shape 和 strides"

!!!note "这里是连续张量吗?"
    理论上是, 是列优先的. 
    但是例如 pytorch 的实现可能认为不是.

现在, 我们不满足之前的规律了. 好在, 我们修正一下就好了

stride[i] = stride[i-1] * shape[i-1]
现在又变成 3 = 3 * 1 了.

~~显然~~, 我们最内层的维度必须有stride = 1. (这不显然, 要想很久) 然后按照一定顺序, 满足 stride[第i内层] = stride[第i+1内层] * shape[第i+1内层], 就能判断是不是内存连续了. 为此, 我们可以定义一个 perm 数组, 用来记录维度的排序方式. 

!!!note 连续张量判定:
    1. stride[perm[-1]] == 1
    2. stride[perm[i]] == stride[perm[i+1]] * shape[perm[i+1]]

终于, 回到我们题目, 居然没有 perm? 大概就是用 pytorch 一样标准只考虑行布局? 对的.


permute, shape 直接 新 strides[i] = 旧 strides[order[i]], strides 呢?
也是! 

- [ ] 为啥呢? ...









(.venv) root@node-212:~/other/llaisys# xmake --root
[ 39%]: cache compiling.release src/tensor/tensor.cpp
error: src/tensor/tensor.cpp:170:33: error: comparison of integer expressions of different signedness: 'const value_type' {aka 'const long int'} and 'size_t' {aka 'long unsigned int'} [-Werror=sign-compare]
  170 |         if (strides()[ndim - i] != stride)
cc1plus: all warnings being treated as errors
  > in src/tensor/tensor.cpp

  为什么我这样不行
  bool Tensor::isContiguous() const {
      size_t stride = 1;
      size_t ndim = shape().size();
      for (size_t i = 1; i <= ndim; i++) {
          if (strides()[ndim - i] != stride) 
              return false;
          stride *= shape()[ndim - i];
      }
      return true;
  }

他那样就行?
size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
### 问题原因
编译错误是因为在 `isContiguous()` 中，`strides()[ndim - i]` 返回 `ptrdiff_t`（有符号，`long int`），而 `stride` 是 `size_t`（无符号，`long unsigned int`）。比较有符号和无符号整数时，编译器发出警告（`-Werror=sign-compare` 将其视为错误）。

- `ptrdiff_t` 用于 strides，因为步长可能为负（但在连续张量中通常正）。
- `size_t` 用于正整数如尺寸。

`create()` 中没有警告，因为赋值 `strides[ndim_ - i] = stride` 是隐式转换（`size_t` 到 `ptrdiff_t`），编译器允许，但比较时严格。

### 为什么 `create` “行”？
`create` 中是赋值，不是比较，所以没触发警告。但为了避免潜在问题，最好统一类型。

### 修复方法
1. **强制转换比较**（推荐，保持 `stride` 为 `size_t`）：
   ```cpp
   bool Tensor::isContiguous() const {
       size_t stride = 1;
       size_t ndim = shape().size();
       for (size_t i = 1; i <= ndim; i++) {
           if (static_cast<size_t>(strides()[ndim - i]) != stride) 
               return false;
           stride *= shape()[ndim - i];
       }
       return true;
   }
   ```

2. **将 `stride` 改为 `ptrdiff_t`**：
   ```cpp
   bool Tensor::isContiguous() const {
       ptrdiff_t stride = 1;
       size_t ndim = shape().size();
       for (size_t i = 1; i <= ndim; i++) {
           if (strides()[ndim - i] != stride) 
               return false;
           stride *= static_cast<ptrdiff_t>(shape()[ndim - i]);
       }
       return true;
   }
   ```

第一种更安全，因为 strides 在连续张量中是正数。用第一种修复后，代码应编译通过。



    struct LlaisysRuntimeAPI {
        get_device_count_api get_device_count;
        set_device_api set_device;
        device_synchronize_api device_synchronize;
        create_stream_api create_stream;
        destroy_stream_api destroy_stream;
        stream_synchronize_api stream_synchronize;
        malloc_device_api malloc_device;
        free_device_api free_device;
        malloc_host_api malloc_host;
        free_host_api free_host;
        memcpy_sync_api memcpy_sync;
        memcpy_async_api memcpy_async;
    };

谁知道呢, api 真的是 api.