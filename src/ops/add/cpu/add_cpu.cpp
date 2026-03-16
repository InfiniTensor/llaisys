/**
 * Add 算子的 CPU 具体实现：按 dtype 分支，逐元素 c[i] = a[i] + b[i]。
 *
 * F16/BF16 先转 float 再相加再转回，避免半精度运算的精度与溢出问题；
 * F32 直接相加。其它 dtype 通过 EXCEPTION_UNSUPPORTED_DATATYPE 报错。
 */
 #include "add_cpu.hpp" // 【大白话】：引入我们刚才说的“任务派发单”

 #include "../../../utils.hpp" // 【大白话】：引入车间里的通用工具箱，比如这里的 cast（类型转换工具）
 
 #include <cmath> // 【大白话】：引入 C++ 标准的数学库（虽然这里加法没直接用到，但算子文件一般都会备着）
 
//  当这段模板函数被调用时，假设传入的类型 T 是 llaisys::bf16_t（一种 16 位浮点数）：
//  编译器看到 if constexpr，检查条件。
//  因为 T 是 bf16_t，std::is_same_v<T, llaisys::bf16_t> 返回 true。
//  编译器只保留 if 里面的代码进行编译，把 a[i] 和 b[i] 强制转换成 float 类型相加，然后把结果再次强制转换回 bf16_t 类型，赋值给 c[i]。
//
//  假设传入的类型 T 是 float：
//  编译器检查条件。
//  两个 is_same_v 都返回 false。
//  编译器直接把 if 块里的代码删掉，只编译 else 里面的代码：c[i] = a[i] + b[i];。
//  这种写法保证了底层数学运算的代码既精简，又能针对不同的数据类型生成最高效的底层机器指令。
 template <typename T>
 void add_(T *c, const T *a, const T *b, size_t numel) {
     // 【大白话】：流水线开启！numel 就是包裹里总共有多少个数字。循环一次，处理一个数字。
     for (size_t i = 0; i < numel; i++) {
         
        // 与普通的 if（在程序运行时判断）不同，if constexpr 是在程序编译时执行的判断。
        //  编译器在编译阶段会计算括号里的条件。如果条件为 true，编译器就把大括号 {} 里的代码编译进最终的机器码；
        // 如果为 false，编译器会直接丢弃这块代码，就像它从来没写过一样。
         if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) { // 如果 T 是 bf16_t 或 fp16_t
             
            // llaisys::utils::cast 表示我们要调用的是 llaisys 命名空间下、utils 子命名空间里的 cast 函数，而不是其他地方可能存在的同名 cast 函数。
            // 这里的 cast 是项目代码库中自定义的一个类型转换函数。尖括号 <float> 明确指示该函数：无论输入参数 a[i] 的原始类型是什么，必须将其转换为 float（32位浮点数）类型并返回。
             c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
         
         } else {
             // 【大白话】：如果本来就是大尺寸（float），那就别折腾了，直接简单粗暴地相加即可。
             c[i] = a[i] + b[i];
         }
     }
 }
 
 namespace llaisys::ops::cpu {
 
 // 【流水线入口】：这是车间主任实际调用的地方。
 // 注意看参数：进来的 a, b, c 全是 std::byte *。
 // 在电脑底层，其实根本没有所谓的“数字”，全是一堆毫无意义的二进制乱码（字节 byte）。
 void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
     
     // 【大白话】：调度员根据标签（type）来看看这批乱码到底该按什么规格处理。
     switch (type) {
     case LLAISYS_DTYPE_F32:
         // 【魔法核心 3：强制透视镜 reinterpret_cast】
         // reinterpret_cast<float *>(c) 的意思就是：给工人戴上一副“浮点数透视镜”。
         // 工人戴上后，看那些原本的字节乱码，就会自动把每 4 个字节当成一个 32位小数(float) 来理解！
         // 然后调用我们上面写的 add_ 通用模具开始干活。
         return add_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
         
     case LLAISYS_DTYPE_BF16:
         // 【大白话】：如果是 BF16，就戴上 BF16 的透视镜（每 2 个字节看成一个数字）。
         return add_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                     reinterpret_cast<const llaisys::bf16_t *>(b), numel);
                     
     case LLAISYS_DTYPE_F16:
         // 【大白话】：如果是 F16，就戴上 F16 的透视镜。
         return add_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                     reinterpret_cast<const llaisys::fp16_t *>(b), numel);
                     
     default:
         // 【大白话】：如果送来了不认识的材料（比如整数格式），直接报警停工。
         EXCEPTION_UNSUPPORTED_DATATYPE(type);
     }
 }
 } // namespace llaisys::ops::cpu