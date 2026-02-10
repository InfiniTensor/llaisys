# Welcome to LLAISYS

<p align="center">
<a href="README.md" target="README.md">English</a> ｜
<a href="README_ZN.md" target="README_ZN.md">中文</a>
</p>

## Introduction

LLAISYS (Let's Learn AI SYStem) is an educational project that aims to provide a platform for new and future AI engineers to learn how to build AI systems from scratch. LLAISYS consists of several assignments, which help students learn and build the basic modules, and projects that challenge them to add more fancy features to their systems. LLAISYS uses C++ as primary programming language for system backend, and is compiled into shared libraries exposing C language APIs. Frontend codes are written in Python which calls these APIs to provide more convenient testing and interaction with other architectures such as PyTorch.

### Project Structure Overview

- `\include`: directory that contains of the header files which defines all the C APIs exposed by the shared library. (Functions declarations start with `__export`)

- `\src`: C++ source files.
  - `\src\llaisys` contains all the direct implementation of waht are defined in the header files and follows the same directory structure as the `\include`. This is also as far as C++ codes can go.
  - other directories contain the actual implementaion of different modules.

- `xmake.lua`: build rules for llaisys backend. `\xmake` directory contains the sub-xmake files for different devices. You may add `nvidia.lua` in the directory in the future for instance to support CUDA.

- `\python`: Python source files.
  - `\python\llaisys\libllaisys` contains all the ctypes wrapper functions of llaisys APIs. It basically matches the structure of C header files.
  - `\python\llaisys` contains Python warppers of the ctypes functions to make the package more Python-like.

- `\test`: Python test files that import llaisys python package.

## Assignment #0: Getting Started

### Task-0.1 Install Prerequisites

- Compile Tool: [Xmake](https://xmake.io/)
- C++ Compiler: MSVC (Windows) or Clang or GCC
- Python >= 3.9 (PyTorch, Transformers, etc.)
- Clang-Format-16 (Optional): for formatting C++ codes.

### Task-0.2 Fork and Build LLAISYS

- FORK LLAISYS Repository and Clone it to your local machine. Both Windows and Linux are supported.

- Compile and Install

  ```bash
  # compile c++ codes
  xmake
  # install llaisys shared library
  xmake install
  # install llaisys python package
  pip install ./python/
  ```

- Github Auto Tests

  LLAISYS uses Github Actions to run automated tests on every push and pull request. You can see testing results on your repo page. All tests should pass once you have finished all assignment tasks.

### Task-0.3 Run LLAISYS for the First Time

- Run cpu runtime tests

  ```bash
  python test/test_runtime.py --device cpu
  ```

  You should see the test passed.

### Task-0.4 Download test model

- The model we use for assignments is [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).

- Run an inference test with the model using PyTorch

  ```bash
  python test/test_infer.py --model [dir_path/to/model]
  ```

  You can see that PyTorch is able to load the model and perform inference with the sample input. You can debug into `transformers` library codes to see how what is going on behind. Right now, your code cannot do anything yet, but you are going to build a system that can achieve the same functionality in the assignments.

## Assignment #1: Tensor

Tensor is a data structure that represents multi-dimensional data. It is the basic building block of LLAISYS, and most AI frameworks such as PyTorch. In this assignment, you will learn how to implement a basic tensor class.

A Tensor object has the following fields:

- `storage`: a shared pointer to a memory block that stores the tensor's data. It can be shared by multiple tensors. Check storage class for more details.
- `offset`:  the starting index (in bytes) of the tensor in the storage.
- `meta`: metadata that describes the tensor's shape, data type, and strides.

Implement the following functions defined in the `src/tensor/tensor.hpp`:

### Task-1.1

```c++
void load(const void *src);
```

Load host (cpu) data to the tensor (can be on device). Check contructor to see how to get runtime apis of the current device context, and do a memcpy from host to device.

### Task-1.2

```c++
bool isContiguous() const; 
```

Check shape and strides of the tensor, and tell wether it is contiguous in memory.

### Task-1.3

```c++
tensor_t view(const std::vector<size_t> &shape) const;
```

Create a new tensor which reshapes the original tensor to the given shape by splitting or merging the original dimensions. No data transfer is involved. For example change a tensor of shape (2, 3, 5) to (2, 15) by merging the last two dimensions.

This function is not as easy as simply changing the shape of the tensor, although the test will pass. It should raise an error if new view is not compatible with the original tensor. Think about a tensor of shape (2, 3, 5) and strides (30, 10, 1). Can you still reshape it to (2, 15) without data transfer?

### Task-1.4

```c++
tensor_t permute(const std::vector<size_t> &order) const;
```

Create a new tensor which changes the order of the dimensions of original tensor. Transpose can be achieved by this function without moving data around.

### Task-1.5

```c++
tensor_t slice(size_t dim, size_t start, size_t end) const;
```

Create a new tensor which slices the original tensor along the given dimension,
start (inclusive) and end (exclusive) indices.

### Task-1.6

Run tensor tests.

```bash
python test/test_tensor.py
```

You should see all tests passed. Commit and push your changes. You should see the auto tests for assignment #1 passed.

## Assignment #2: Operators

In this assignment, you will implement the cpu verision the following operators:

- argmax
- embedding
- linear
- rms_norm
- rope
- self_attention
- swiglu

Read the codes in `src/ops/add/` to see how "add" operator is implemented. Make sure you understand how the operator codes are organized, compiled, linked, and exposed to Python frontend. **Your operators should at least support Float32, Float16 and BFloat16 data types**. A helper function for naive type casting is provided in `src/utils/`. All python tests are in `test/ops`, you implementation should at least pass these tests. Try running the test script for "add" operator for starting.

### Task-2.1 argmax

```c++
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
```

Get the max value and its index of tensor `vals`, and store them in `max_val` and `max_idx` respectively. You can assume that `vals` is a 1D tensor for now, and `max_idx` and `max_val` are both 1D tensors with a single element (, which means the dimension of `vals` is kept).

You should be able to pass the test cases in `test/ops/argmax.py` after you finish the implementation.

### Task-2.2 embedding

```c++
void embedding(tensor_t out, tensor_t index, tensor_t weight);
```

Copy the rows in `index` (1-D) from `weight` (2-D) to `output` (2-D). `index` must be of type Int64 (the default data type for int of PyTorch).

You should be able to pass the test cases in `test/ops/embedding.py` after you finish the implementation.

### Task-2.3 linear

```c++
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
```

Compute the following:

$$
Y = xW^T + b
$$

- `out`: output $Y$ . You can assume output is a 2D contiguous tensor  and no broadcasting is involved for now.
- `input`: input $X$ . You can assume input is a 2D contiguous tensor  and no broadcasting is involved for now.
- `weight`: weight $W$ . 2D contiguous tensor. Note that weight tensor is not transposed. You need to deal with this during your calculation.
- `bias` (optional): bias $b$ . 1D tensor. You need to support the situation where bias is not provided.

You should be able to pass the test cases in `test/ops/linear.py` after you finish the implementation.

### Task-2.4 rms normalization

```c++
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
```

Compute the following for each row:

$$
Y_i = \frac{W_i \times  X_i}{\sqrt{\frac{1}{d}(\sum_{j=1}^d X_j^2) + \epsilon}}
$$

- `out`: output $Y$ . You can assume output is a 2D contiguous tensor and no broadcasting is involved for now.
- `input`: input $X$ . You can assume input is a 2D contiguous tensor and no broadcasting is involved for now. The normalization is performed along the last dimension (a.k.a. each row of length $d$ ) of the input tensor.
- `weight`: weight $W$ . 1D tensor, same length as a row of input tensor.
- `eps`: small value $\epsilon$ to avoid division by zero.

You should be able to pass the test cases in `test/ops/rms_norm.py` after you finish the implementation.

### Task-2.5 rope

```c++
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
```

Compute the following for each vector of input tensor `in`, corresponding to a position id in `pos_ids`:

Let $\mathbf{x}_i = [\mathbf{a}_i, \mathbf{b}_i] \in \mathbb{R}^d$ be the input vector and $\mathbf{y}_i = [\mathbf{a}'_i, \mathbf{b}'_i] \in \mathbb{R}^d$ be the output vector at index $i$, where $\mathbf{a}_i, \mathbf{b}_i,\mathbf{a}'_i, \mathbf{b}'_i \in \mathbb{R}^{d/2}$ .

Let $\theta$ be a fixed base (e.g. $\theta = 10000$) and $j = 0, 1, \ldots, d/2 - 1$.

Let $p_i \in \mathbb{N}$ is the position id for token at input index i.

Then the angle for RoPE is $\phi_{i,j} = \frac{p_i}{\theta^{2j/d}}$

The output vector $\mathbf{y}_i = [\mathbf{a}'_i, \mathbf{b}'_i]$ is computed as follows:

$$a_{i,j}' = a_{i,j} \cos(\phi_{i,j}) - b_{i,j} \sin(\phi_{i,j})$$

$$b_{i,j}' = b_{i,j} \cos(\phi_{i,j}) + a_{i,j} \sin(\phi_{i,j})$$

- `out`: the resulting **q** or **k** tensor. Shape should be [seqlen, nhead, d] or [seqlen, nkvhead, d]. You can assume that the tensor is contiguous for now.
- `in`: the orignal **q** or **k** tensor. Shape should be [seqlen, nhead, d] or [seqlen, nkvhead, d]. You can assume that the tensor is contiguous for now.
- `pos_ids`: the position id (index in the whole context) for each token in the input sequence. Shape should be [seqlen,], dtype should be int64.
- `theta`: the base value for the frequency vector.

You should be able to pass the test cases in `test/ops/rope.py` after you finish the implementation.

### Task-2.6 self-attention

```c++
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
```

Compute the self-attention for query tensor `q`, key tensor `k`, and value tensor `v`. You should concat kvcache tensors, if needed, before doing this calculation.

$$
A = Q K^\top * scale \\
$$

$$
Y = \mathrm{causalsoftmax}(A) \cdot V \\
$$

- `attn_val`: the resulting attention value tensor. Shape should be [seqlen, nhead, dv]. You can assume that the tensor is contiguous for now.
- `q`: the query tensor. Shape should be [seqlen, nhead, d]. You can assume that the tensor is contiguous for now.
- `k`: the key tensor. Shape should be [total_len, nkvhead, d]. You can assume that the tensor is contiguous for now.
- `v`: the value tensor. Shape should be [total_len, nkvhead, dv]. You can assume that the tensor is contiguous for now.
- `scale`: a scaling factor. It is set to $\frac{1}{\sqrt{d}}$ in most cases.

You should be able to pass the test cases in `test/ops/self_attention.py` after you finish the implementation.

### Task-2.7 swiglu

```c++
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
```

This is an element-wise function that computes the following:

$$
out_{i} = up_{i} \circ \frac { gate_{i}}{1 + e^{-gate_{i}}}
$$

`out`, `up` and `gate` are 2D contiguous tensors with the same shape [seqlen, intermediate_size].

You should be able to pass the test cases in `test/ops/swiglu.py` after you finish the implementation.

### Task-2.8

Run operator tests.

```bash
python test/test_ops.py
```

You should see all tests passed. Commit and push your changes. You should see the auto tests for assignment #2 passed.

### Task-2.9 (Optional) rearrange

This is a bonus task. You may or may not need it for model inference.

```c++
void rearrange(tensor_t out, tensor_t in);
```

This operator is used to copy data from a tensor to another tensor with the same shape but different strides. With this, you can easily implement `contiguous` functionality for tensors.

## Assignment #3: Large Language Model Inference

Finally, it is the time for you to achieve text generation with LLAISYS.

- In `test/test_infer.py`, your implementation should be able to generate the same texts as PyTorch, using argmax sampling. The model we use for this assignment is [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).

- The python wrapper of your implementation is in `python/llaisys/models/qwen2.py`. You are NOT allowed to implement your model infer logic here using any python based frameworks, such as PyTorch. Instead, you need to implement the model with C/C++ in LLAISYS backend. The script loads each tensor in the safetensors file, and you will need to load data from them into your model backend.

- In `include/llaisys/models/qwen2.h`, a prototype is defined for you. Feel free to modify the codes as you want, but you should at least provide basic APIs for model creation, destruction, data loading, and infer. Implement your C APIs in `src/llaisys/` and organize your C++ codes as other modules in `src/`. Remember to define the compiling procedures in `xmake.lua`.

- In `python/llaisys/libllaisys/`, define the ctypes wrapper functions for your C APIs. Implement `python/llaisys/models/qwen2.py` with your wrapper functions.

- You need to implement KV Cache, or your model will be too slow.

- Debug until your model works. Take advantage of tensor's `debug` function which prints the tensor data. It allows you to compare the data of any tensor during the model inference with PyTorch.

After you finish the implementation, you can run the following command to test your model:

```bash
python test/test_infer.py --model [dir_path/to/model] --test
```

Commit and push your changes. You should see the auto tests for assignment #3 passed.


## You can proceed to the projects only after you finish the assignments.

## Project #1: Optimize LLAISYS for CPU
You probably have already noticed that your model inference is very slow compared to PyTorch. This is mostly because your operators are not optimized. Run your operater test scripts with "--profile" flag to see how your operators perform. You would probably see that `linear` operation is much slower than PyTorch. This operator is mainly a matrix multiplication, and is the most time consuming operation in transformer-based models.

There are several ways to optimize your operators for CPU:

### SIMD instructions

SIMD (Single Instruction Multiple Data) instructions are instructions that can perform the same operation on multiple data elements in a single instruction. Modern CPUs have support for SIMD instructions. Look for online materials to learn about compiler intrinsics (such as AVX2, AVX-512, NEON, SVE) to vectorize your operations.

### Use OpenMP for parallelism

You can use multi-threading to parallelize your operators. OpenMP is a popular library for multi-threading in C/C++. Add OpenMP support for LLAISYS to parallelize your `linear` and other operators.

### 3rd-party Libraries

There are several libraries that can help you optimize your operators for CPU. Look for libraries like Eigen, OpenBLAS, MKL, etc. to optimize your linear algebra operations. Note that some libraries are supported only for certain hardware platforms. Check their documentations and use them in your codes with care. You can also try to dig out how PyTorch implement these operators and see if you can use them.

Optimize your implementation with any methods you like and report your performance improvement.

## Project #2: Intigrate CUDA into LLAISYS

This project does not depend on **Project #1**. You should choose two CUDA/CUDA-ish hardware platforms from Nvidia, Iluvatar, Metax, and Moore Threads.

This camp session provides computation resources from the four platforms above, access to which is granted based on applications from the official website. You can accelerate your model with CUDA on these GPU platforms. Before doing that, let's dive deeper into LLAISYS framework. 

LLAISYS is actually a framework with homogeous hardware support. When using LLAISYS, each thread will create a thread-local `Context` object which manages all the device `Runtime` objects used by this thread. A `Runtime` object is a resource manager for a device, and `Context` will create (with lazy initialization) a single `Runtime` object for each device. You can set and switch between them using `setDevice` function in `Context`. Only one device will be active at a time for each thread. Check `src/core/context.hpp` for more details. 

### Implement CUDA Runtime APIs
Each `Runtime` object is intialized with a set of generic functions called `Runtime APIs`. You will need to implement CUDA version of these APIS. Check `src/device/cpu/cpu_runtime_api.cpp` to see how these functions are implemented for CPU and look for CUDA APIs to use in [`CUDA Runtime documentation`](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html).

You can see in `src/device/runtime_api.hpp` that `nvidia::getRuntimeAPI()` is guarded by `ENABLE_NVIDIA_API` macro.

```c++
#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const LlaisysRuntimeAPI *getRuntimeAPI();
}
#endif
```

This macro is defined in `xmake.lua` as a switch to enable/disable CUDA support. CUDA codes will not be compiled if the switch is off. In `xmake/` directory, create a `nvidia.lua` that configs your compiling process. (Similar to `cpu.lua` for CPU.) Search online to learn how to do it with Xmake.

After you implement the CUDA Runtime APIs, config your xmake with `--nv-gpu=y` to enable CUDA support and recompile your program. Run runtime tests to see if your implementation works.

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
python test/test_runtime.py --device nvidia
```

### Implement CUDA Operators
Create a `nvdia/` sub-directory in each operator source directory and implement a cuda version. Check `src/ops/add/op.cpp` to see how to include your cuda implementations. Remeber to define the compiling procedures in the xmake files. Run the operator tests with `--device nvidia` flag to test your CUDA implementation.

You can use CUDA libraries like cuBLAS, cuDNN, etc. to accelerate your operators. Check their documentations to see how to use them. You can store extra device resources in `src/device/nvidia/nvidia_resource.cu`.

Modify your model codes to support CUDA inference. 

```bash
python test/test_infer.py --model [dir_path/to/model] --test --device nvidia
```

## Project #3: Build an AI chatbot

In this project you will build an AI chatbot that can do live conversations with single user with LLAISYS. 

### Random Sampling

So far we have been testing our model with argmax sampling. This is good enough for testing, but a chatbot should be able to generate more natural responses. Implement a random sample operator. Try to add supports for **Temperature**, **Top-K** and **Top-P**.

### Build a Chatbot Server

In your Python frontend, implement a server that can receive http requests from user and send responses back. You can use frameworks like FastAPI to build the server. You should follow the OpenAI chat-completion APIs. Try to support streaming responses if you can. You can assume, for now, that the server is only serving one user, and block the endpoint until the previous request is served.


### Interactive Chat UI

Build a UI that send requests to and receive responses from the chatbot server. You can build a simple command-line interface or a fancy web interface. You should be able to keep a conversation going with the chatbot by sending messages and receiving responses consecutively.

### (Optional) Chat Session Management

In real-world AI applications, users are allowed to start new conversations and switch between them. Users can also edit a past question and let the AI regenerate an answer. Enhance your UI to support these features. Implement a KV-Cache pool with prefix matching to reuse past results as much as possible.


## Project #4: Multi-user Inference Service

You need to finish **Project #2** and achieve streaming response first before proceeding to this project.

### Serving Multiple Users

In real-world scenarios, an inference service will serve multiple users. Requests can come in at any time, and the service should be able to handle them concurrently. Your endpoint should add a new request to a request pool or queue and have a another looping process or thread to serve the requests. 

### Continous Batching
To maximize the throughput of your inference service, you need to batch your requests instead of serving them one by one. Since each request can have different length, you will need a continous and iteration-level batching mechanism. For each interation you extract several requests from pool to form a batch, do one round of batch inference, and then return the unfinished requests back to the pool. Use batched matrix multiplication when possible to speed up your inference. Note that every request in the batch need to bind with a different KV-Cache. You should build a KV-Cache pool with prefix matching to reuse past results as much as possible.

## Project #5: Distributed Inference
Introduce Tensor Parallelism to LLAISYS. Shard your model across multiple devices and implement distributed model inference. Support NCCL in LLAISYS if your are uing Nvidia GPUs, or MPI if you are using CPUs.

## Project #6: Support New Models

Support another model type than the one we use for homework in LLAISYS.
