/**
 * Tensor 类实现：多维数组的创建、元数据访问、view/permute/slice/load 等。
 *
 * 张量由 (storage, offset, meta) 描述：storage 为一块内存，offset 为字节偏移，
 * meta 含 dtype、shape、strides。strides 单位为「元素个数」，元素 (i0,i1,...) 的
 * 线性下标为 sum(ik * strides[k])，再乘以 elementSize() 得字节偏移。
 */
 #include "tensor.hpp"

 #include "../utils.hpp"
 
 #include <cstring>
 #include <numeric>
 #include <sstream>
 
 namespace llaisys {
 
 /// 私有构造：用已有 meta、storage、offset 构造张量（供 create/view/permute/slice 等内部使用）
 /// 【大白话注释】：这就像是给一批已经存在的货物，贴上一张新的说明书。
 Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
     : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}
 
 /**
  * 创建新张量：按 shape 分配 storage，并计算行主序（row-major）的默认 strides。
  * 若当前 Context 的 runtime 不是 CPU 但请求 device_type 为 CPU，则分配主机内存（allocateHostStorage），
  * 否则切到目标设备后分配设备内存（allocateDeviceStorage）。
  */
 tensor_t Tensor::create(const std::vector<size_t> &shape,
                         llaisysDataType_t dtype,
                         llaisysDeviceType_t device_type,
                         int device) {
     size_t ndim_ = shape.size();
     std::vector<ptrdiff_t> strides(ndim_);
     size_t stride = 1;
     
     // 【魔法核心 1：步长是怎么算出来的？】
     // 行主序：最后一维 stride=1，往前逐维乘上后一维的 shape
     // 假设 shape 是 [2, 3, 4]（2个大块，每块3行，每行4列）。
     // 最后一维（列）步长是 1：往右走一格，在物理内存上跳过 1 个元素。
     // 倒数第二维（行）步长是 4：往下走一行，在物理内存上要跳过 4 个元素。
     // 倒数第三维（块）步长是 3*4=12：往下一块走，在物理内存上要跳过 12 个元素。
     for (size_t i = 1; i <= ndim_; i++) {
         strides[ndim_ - i] = stride;
         stride *= shape[ndim_ - i];
     }
     
     TensorMeta meta{dtype, shape, strides};
     // 算出来的最终 stride 就是这批货的总元素个数
     size_t total_elems = stride; 
     size_t dtype_size = utils::dsize(dtype); // 算出一个元素占几个字节
 
     // 确保当前线程已激活设备，避免 context().runtime() 在未 setDevice 时断言失败
     if (device_type == LLAISYS_DEVICE_CPU) {
         core::context().setDevice(LLAISYS_DEVICE_CPU, 0);
         if (core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
             auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
             return std::shared_ptr<Tensor>(new Tensor(meta, storage));
         }
     }
     core::context().setDevice(device_type, device);
     auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
     return std::shared_ptr<Tensor>(new Tensor(meta, storage));
 }
 
 // 【大白话】：去仓库拿货。注意必须加上 _offset！
 // 比如被 slice 切片过的张量，它的起点可能不在仓库最开头，加上 offset 才能精准找到货。
 std::byte *Tensor::data() {
     return _storage->memory() + _offset;
 }
 
 const std::byte *Tensor::data() const {
     return _storage->memory() + _offset;
 }
 
 size_t Tensor::ndim() const {
     return _meta.shape.size();
 }
 
 const std::vector<size_t> &Tensor::shape() const {
     return _meta.shape;
 }
 
 const std::vector<ptrdiff_t> &Tensor::strides() const {
     return _meta.strides;
 }
 
 llaisysDataType_t Tensor::dtype() const {
     return _meta.dtype;
 }
 
 llaisysDeviceType_t Tensor::deviceType() const {
     return _storage->deviceType();
 }
 
 int Tensor::deviceId() const {
     return _storage->deviceId();
 }
 
 /// 元素总数：各维 shape 的乘积
 size_t Tensor::numel() const {
     return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
 }
 
 size_t Tensor::elementSize() const {
     return utils::dsize(_meta.dtype);
 }
 
 /// 调试用字符串：shape、strides、dtype
 std::string Tensor::info() const {
     std::stringstream ss;
 
     ss << "Tensor: "
        << "shape[ ";
     for (auto s : this->shape()) {
         ss << s << " ";
     }
     ss << "] strides[ ";
     for (auto s : this->strides()) {
         ss << s << " ";
     }
     ss << "] dtype=" << this->dtype();
 
     return ss.str();
 }
 
 /**
  * 递归按维打印数据：最后一维逐元素输出，其余维递归到下一维；
  * 下标 (i0, i1, ..., i_{dim}) 对应指针 data + i_dim * strides[dim]，再递归时传入 data + i_dim * strides[dim]。
  */
 template <typename T>
 void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
     if (dim == shape.size() - 1) {
         for (size_t i = 0; i < shape[dim]; i++) {
             if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                 std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
             } else {
                 std::cout << data[i * strides[dim]] << " ";
             }
         }
         std::cout << std::endl;
     } else if (dim < shape.size() - 1) {
         for (size_t i = 0; i < shape[dim]; i++) {
             print_data(data + i * strides[dim], shape, strides, dim + 1);
         }
     }
 }
 
 /// 按 dtype 将 data 转成对应类型指针，再调用 print_data 打印
 void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
     switch (dtype) {
     case LLAISYS_DTYPE_BYTE:
         return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_BOOL:
         return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_I8:
         return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_I16:
         return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_I32:
         return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_I64:
         return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_U8:
         return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_U16:
         return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_U32:
         return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_U64:
         return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_F16:
         return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_F32:
         return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_F64:
         return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
     case LLAISYS_DTYPE_BF16:
         return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
     default:
         EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
     }
 }
 
 /**
  * 调试输出：先同步设备，打印 info()，再按 shape/strides 递归打印元素。
  * 若张量在设备上，先 D2H 拷到临时 CPU 张量再打印。
  */
 // 【大白话】：打印功能。如果是 GPU 上的张量，电脑屏幕没法直接看，必须先偷偷拷回 CPU 才能打印。
 void Tensor::debug() const {
     core::context().setDevice(this->deviceType(), this->deviceId());
     core::context().runtime().api()->device_synchronize();
     std::cout << this->info() << std::endl;
     if (this->deviceType() == LLAISYS_DEVICE_CPU) {
         debug_print(this->data(), this->shape(), this->strides(), this->dtype());
     } else {
         auto tmp_tensor = create({this->_storage->size()}, this->dtype());
         core::context().runtime().api()->memcpy_sync(
             tmp_tensor->data(),
             this->data(),
             this->numel() * this->elementSize(),
             LLAISYS_MEMCPY_D2H);
         debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
     }
 }
 
 /**
  * 判断是否行主序连续：strides[n-1]=1，且 strides[i] = strides[i+1] * shape[i+1]。
  * view 等操作要求张量连续，否则无法在不拷贝的前提下用新 shape 解释同一块内存。
  */
 bool Tensor::isContiguous() const {
     const auto &shp = shape();
     const auto &strd = strides();
     size_t n = ndim();
     if (n == 0) return true;
     
     // 【质检员逻辑】：
     // 假设理想状态下，最后一维步长必须是 1（expected = 1）。
     // 然后倒着往前推：倒数第二维的步长，必须等于（最后一维的步长 * 最后一维的长度）。
     // 只要有任何一维算出来的步长，跟说明书上写的不一样，就说明货物中间被掏空了或者被打乱了（不连续）。
     ptrdiff_t expected = 1;
     for (size_t i = n; i > 0; i--) {
         size_t idx = i - 1;
         if (strd[idx] != expected) return false;
         expected *= static_cast<ptrdiff_t>(shp[idx]);
     }
     return true;
 }
 
 /**
  * 调换维度顺序：新张量 shared storage 与 offset，new_shape[i]=old_shape[order[i]]，
  * new_strides[i]=old_strides[order[i]]。不拷贝数据，仅改变「怎么看」同一块内存。
  */
 tensor_t Tensor::permute(const std::vector<size_t> &order) const {
     size_t n = ndim();
     CHECK_ARGUMENT(order.size() == n, "permute: order size must equal tensor ndim");
     std::vector<bool> seen(n, false);
     for (size_t i = 0; i < n; i++) {
         CHECK_ARGUMENT(order[i] < n, "permute: order index out of range");
         CHECK_ARGUMENT(!seen[order[i]], "permute: order must be a permutation of [0..ndim-1]");
         seen[order[i]] = true;
     }
     std::vector<size_t> new_shape(n);
     std::vector<ptrdiff_t> new_strides(n);
     
     // 【转置的魔法核心】：
     // 假设原来的 shape 是 [2行, 3列]，strides 是 [3, 1]（跨行跳3，跨列跳1）。
     // 你想把它转置成 [3行, 2列]。
     // order 传进来的是 [1, 0]（意思是把第1维放到前面，第0维放到后面）。
     // 代码执行后：new_shape 变成了 [3列, 2行]，new_strides 变成了 [1, 3]（跨行跳1，跨列跳3）！
     // 仓库里的货完全没动，但下次读取时，系统就会按新的步长去跳跃读取，自动变成了竖着读。
     for (size_t i = 0; i < n; i++) {
         new_shape[i] = _meta.shape[order[i]];
         new_strides[i] = _meta.strides[order[i]];
     }
     
     // 把新的说明书（meta）和旧的仓库（_storage）打包在一起，返回给用户
     TensorMeta meta{_meta.dtype, new_shape, new_strides};
     return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
 }
 
 /**
  * 用新 shape 重新解释当前张量，不拷贝数据；要求当前张量连续且新 shape 元素总数等于 numel()。
  * 新张量使用相同的 storage 和 offset，按新 shape 计算行主序 strides。
  */
 tensor_t Tensor::view(const std::vector<size_t> &shape) const {
     size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
     // 【安全检查】：你把 12 个杯子看成 3x4 或者 2x6 都可以，但不能把它当成 5x5 看，所以总数必须一样。
     CHECK_ARGUMENT(new_numel == numel(), "view: shape element count must match tensor numel");
     CHECK_ARGUMENT(isContiguous(), "view: tensor must be contiguous (no data copy)");
     
     size_t ndim_ = shape.size();
     std::vector<ptrdiff_t> strides(ndim_);
     ptrdiff_t stride = 1;
     
     // 【大白话】：既然你要换个形状看，那我顺便帮你把新的“连续步长”重新算一遍。
     for (size_t i = ndim_; i > 0; i--) {
         strides[i - 1] = stride;
         stride *= static_cast<ptrdiff_t>(shape[i - 1]);
     }
     
     // 同样，新说明书 + 老仓库
     TensorMeta meta{_meta.dtype, shape, strides};
     return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset));
 }
 
 /**
  * 沿第 dim 维取 [start, end)，左闭右开。新张量 shared storage，仅改 shape[dim] 和 offset：
  * new_shape[dim] = end - start，new_offset = _offset + (start * strides[dim]) * elementSize()。
  * strides 不变（单位是元素个数），故新张量仍按原步长访问子区域。
  */
 
 tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
     size_t n = ndim();
     CHECK_ARGUMENT(dim < n, "slice: dim must be less than ndim");
     CHECK_ARGUMENT(start <= end, "slice: start must be <= end");
     CHECK_ARGUMENT(end <= _meta.shape[dim], "slice: end must be <= shape[dim]");
     
     std::vector<size_t> new_shape = _meta.shape;
     // 【切片魔法 1】：比如原来这一维长度是 10，你只想要第 2 到第 5 个。那新长度就是 5 - 2 = 3。
     new_shape[dim] = end - start;
     
     // 【切片魔法 2：调整起始点 offset】
     // start * _meta.strides[dim]：算出在这一维上跳过 start 个单位，总共等于跳过了几个物理格子。
     size_t elem_offset = start * static_cast<size_t>(_meta.strides[dim]);
     // 乘上每个格子占用的字节数（elementSize），算出最终需要在物理内存上往后平移多少字节。
     size_t new_offset = _offset + elem_offset * elementSize();
     
     // 步长原封不动！因为虽然你把框划小了，但跨行跨列的物理距离依然没变。
     TensorMeta meta{_meta.dtype, new_shape, _meta.strides};
     return std::shared_ptr<Tensor>(new Tensor(meta, _storage, new_offset));
 }
 
 /**
  * 从主机内存 src_ 拷贝 numel()*elementSize() 字节到本张量（可位于 CPU 或设备）。
  * 若本张量在 CPU 则 H2H，否则 H2D。
  */
 void Tensor::load(const void *src_) {
     // 【大白话】：这是全村唯一的“老实人”。只有它在真金白银地搬运内存！
     // 算出总字节数，调用底层的 memcpy_sync 把源数据生硬地砸进我们申请好的 _storage 里。
     size_t size_bytes = numel() * elementSize();
     core::context().setDevice(deviceType(), deviceId());
     llaisysMemcpyKind_t kind = (deviceType() == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
     core::context().runtime().api()->memcpy_sync(data(), src_, size_bytes, kind);
 }
 
 tensor_t Tensor::contiguous() const {
     TO_BE_IMPLEMENTED();
     return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
 }
 
 tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
     TO_BE_IMPLEMENTED();
     return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
 }
 
 tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
     TO_BE_IMPLEMENTED();
     return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
 }
 
 } // namespace llaisys