import gc  # 导入垃圾回收模块，用于后续手动释放 PyTorch 占用的内存
from test_utils import * # 导入项目中通用的测试辅助函数（比如处理 device_name 的函数）

import argparse  # 用于解析我们在终端输入的命令行参数（如 --model, --test 等）
from transformers import AutoModelForCausalLM, AutoTokenizer  # 拥抱脸（HuggingFace）库的核心类，用于加载官方模型和分词器
import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import snapshot_download  # 用于从云端下载模型权重文件
import os
import time
import sys
import io

# 强制将终端的标准输出设置为 utf-8 编码，防止大模型生成特殊字符或中文时终端乱码报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 确保我们使用的是项目本地的 llaisys 后端（无论是从仓库根目录运行还是配置了 PYTHONPATH）
import llaisys_py
print(f"[test_infer] llaisys_py loaded from: {os.path.abspath(os.path.dirname(llaisys_py.__file__))}")


def load_hf_model(model_path=None, device_name="cpu"):
    """
    加载 Hugging Face (PyTorch) 版本的基准模型。
    用于生成“标准答案”以供后续对比。
    """
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # 如果用户提供了本地路径且路径有效，则从本地加载；否则从 Hugging Face 远程下载
    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
        
    # 加载分词器（Tokenizer），它负责把文本变成数字 ID (Tokens)，或者把数字 ID 还原成文本
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载大语言模型本体，以 bfloat16（一种节省内存的 16 位浮点数）格式加载到指定的设备（CPU 或 GPU）上
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    """
    使用 PyTorch/HuggingFace 模型进行推理生成文本。
    """
    # 使用模型专属的对话模板（Chat Template）格式化用户输入的 prompt
    # 比如自动在问题前后加上 <｜User｜> 和 <｜Assistant｜> 这种特殊标记
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # 将格式化后的文本转换为模型能看懂的张量（Tensor），并移动到模型所在的计算设备上
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    
    # 禁用 PyTorch 的梯度计算（因为我们只是生成/推理，不需要训练模型，这样可以大幅节省显存/内存并提速）
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens, # 最大生成的 token 数量
            top_k=top_k,                   # 随机采样参数：限制只从概率最高的 k 个词中选择
            top_p=top_p,                   # 随机采样参数：限制只从累积概率超过 p 的词库中选择
            temperature=temperature,       # 温度参数：控制生成的随机性，值越小越稳定，越大越发散
        )
        
    # 将模型输出的 Token ID 数字列表解码回人类可读的字符串，并跳过特殊的控制字符
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def load_llaisys_model(model_path, device_name):
    """
    加载我们自己手写的 LLAISYS C++ 引擎版本的模型。
    """
    # 调用 LLAISYS Python 前端封装的 Qwen2 类进行初始化
    model = llaisys_py.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    """
    使用自定义的 LLAISYS 引擎进行推理。
    这里的输入输出接口设计得和 PyTorch 版本非常相似。
    """
    # 同样使用分词器应用对话模板
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # 编码文本，这里不需要 return_tensors="pt"，因为 LLAISYS 接收的是普通的 Python 列表或自定义格式
    inputs = tokenizer.encode(input_content)
    if not inputs:
        raise ValueError(
            "tokenizer.encode returned empty list for prompt %r" % (prompt[:50],)
        )
        
    # 调用 LLAISYS C++ 后端暴露的 generate 方法开始推理（这里也是算力消耗最大的地方）
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    
    # --- 下面是一系列的安全检查，用于在 C++ 后端报错或返回异常空值时给出友好的提示 ---
    if not outputs:
        raise RuntimeError(
            "LLAISYS generate() returned no tokens. "
            "Run 'xmake install' in the project root (after xmake), then 'pip install -e ./python' "
            "so Python uses the project's llaisys_py and the updated llaisys.dll (or .so)."
        )
    # 语言模型生成是“续写”的过程，所以输出的长度必须大于等于输入的长度
    if len(outputs) < len(inputs):
        raise RuntimeError(
            "LLAISYS generate() returned %d tokens but input had %d (expected >= input length). "
            "Check that the project llaisys is used (see path printed at start)."
            % (len(outputs), len(inputs))
        )

    # 返回 token 列表和解码后的最终文本
    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    def _log(msg):
        print(msg, flush=True)
    _log("test_infer: starting...")
    # 配置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str) # 选择硬件平台
    parser.add_argument("--model", default=None, type=str) # 模型路径
    parser.add_argument("--prompt", default="Who are you?", type=str) # 测试用的提示词
    parser.add_argument("--max_steps", default=128, type=int) # 最大生成步数
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true") # 是否开启严格测试模式的开关

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    
    # 【核心逻辑】如果是跑作业的自动测试 (args.test 为真)
    # 为了保证 PyTorch 和 LLAISYS 输出百分之百一致，必须关闭随机性。
    # 设置 top_k=1 代表只取概率最高的唯一结果，这就是经典的 "Argmax 贪婪采样"。
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    # 阶段一：加载并运行官方 PyTorch 模型
    try:
        _log("Loading PyTorch model and tokenizer (may take a while)...")
        tokenizer, model, model_path = load_hf_model(args.model, args.device)
        _log("PyTorch model loaded.")
    except Exception as e:
        _log(f"Failed to load PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[Stage 1] Running PyTorch inference (max_new_tokens={args.max_steps})... CPU 上可能较慢，请耐心等待。\n")
    start_time = time.time() # 记录开始时间
    tokens, output = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time() # 记录结束时间

    # 因为大模型极度吃内存，在跑你的 LLAISYS 之前，必须把 PyTorch 模型从内存中删掉，并手动清空垃圾
    del model
    gc.collect()

    # 打印基准答案
    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    # 阶段二：加载并运行自定义的 LLAISYS 模型
    print("[Stage 2] Loading LLAISYS model...")
    try:
        model = load_llaisys_model(model_path, args.device)
    except Exception as e:
        print("\n\033[91mLLAISYS model load failed:\033[0m")
        import traceback
        traceback.print_exc()
        raise

    print("[Stage 2] Running LLAISYS inference...")
    start_time = time.time()
    try:
        llaisys_tokens, llaisys_output = llaisys_infer(
            args.prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_steps,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
    except Exception as e:
        print("\n\033[91mLLAISYS inference failed:\033[0m")
        import traceback
        traceback.print_exc()
        raise

    end_time = time.time()

    # 打印你的测试结果
    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    # 阶段三：对账单环节
    # 如果处于测试模式下，代码会严格断言 (assert) 两者的 Token 数组必须一模一样
    if args.test:
        if llaisys_tokens != tokens:
            raise AssertionError(
                "LLAISYS token sequence did not match PyTorch. "
                "If LLAISYS returned [], ensure 'xmake build llaisys' was run and the DLL in "
                "python/llaisys_py/libllaisys/ is up to date."
            )
        print("\033[92mTest passed!\033[0m\n") # 如果没报错，打印绿色的通过信息