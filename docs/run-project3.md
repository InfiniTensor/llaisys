# 如何运行 Project #3 聊天机器人

## 前置条件

1. **编译环境**：已安装 [Xmake](https://xmake.io/) 和 C++ 编译器（MSVC / Clang / GCC）
2. **模型**：已下载 [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)，记下本地路径。下载方式见下文「下载模型」。
3. **Python**：>= 3.9，已安装 PyTorch、transformers 等（见 `python/setup.cfg`）

---

## 下载模型（DeepSeek-R1-Distill-Qwen-1.5B）

任选一种方式，将模型下载到本地后，用该目录路径作为 `--model` 参数。

### 方式 A：项目自带脚本（推荐）

确保已安装 `huggingface_hub`（可用项目 venv）：

```bash
cd /home/chenncy/llaisys
python3 -m venv .venv
.venv/bin/pip install huggingface_hub
.venv/bin/python scripts/download_model.py
```

默认会下载到 `llaisys/models/DeepSeek-R1-Distill-Qwen-1.5B`。指定目录：

```bash
.venv/bin/python scripts/download_model.py --dir /你的路径/DeepSeek-R1-Distill-Qwen-1.5B
```

### 方式 B：任意 Python 环境

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir='./models/DeepSeek-R1-Distill-Qwen-1.5B')
print('下载完成:', path)
"
```

### 方式 C：Hugging Face CLI

```bash
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 方式 D：Git + LFS（需先安装 git-lfs）

```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

---

## 一、编译并安装 LLAISYS

在项目根目录 `/home/chenncy/llaisys` 执行：

```bash
# 1. 编译 C++ 后端
xmake

# 2. 安装动态库（会复制到 python/llaisys_py/libllaisys/）
xmake install

# 3. 安装 Python 包（可编辑模式方便改代码）
pip install -e ./python/
```

若未安装 xmake，可先安装：

- **Linux**: `curl -fsSL https://xmake.io/get.sh | bash` 或包管理器
- **Windows**: 从 [xmake  releases](https://github.com/xmake-io/xmake/releases) 下载

---

## 二、安装服务端与客户端依赖

```bash
pip install fastapi uvicorn requests
```

（transformers / torch 已在 llaisys 的 install_requires 中，pip install 时会装）

---

## 三、启动聊天服务端

任选一种方式指定模型路径：

```bash
# 方式 A：命令行参数（推荐）
python -m llaisys_py.server --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B --port 8000

# 方式 B：环境变量
export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B
python -m llaisys_py.server --port 8000
```

可选参数：

- `--host 127.0.0.1`：监听地址（默认 127.0.0.1）
- `--port 8000`：端口（默认 8000）
- `--device cpu`：设备，目前用 `cpu` 即可（nvidia 需 Project #2 完成）

看到 “Model ready. Starting server...” 即表示服务已就绪。

---

## 四、使用聊天界面

### 方式 1：Web 页面（推荐）

浏览器打开：

**http://127.0.0.1:8000/chat**

在页面里输入内容发送，即可多轮对话。

### 方式 2：命令行客户端

**新开一个终端**，在项目根或任意目录执行：

```bash
python -m llaisys_py.server.chat_cli
```

默认连到 `http://127.0.0.1:8000`。输入内容回车发送，输入 `quit` 或 `q` 退出。

可选参数示例：

```bash
python -m llaisys_py.server.chat_cli --base-url http://127.0.0.1:8000 --max-tokens 128 --temperature 0.8 --top-k 50 --top-p 0.9
```

---

## 五、未安装 llaisys 时用 PYTHONPATH 运行（不依赖 pip install）

若没有执行 `pip install -e ./python/`（例如因网络超时装不上），可直接用 PYTHONPATH 运行，无需安装包。

**前提**：已执行 `xmake && xmake install`，且 `python/llaisys_py/libllaisys/` 下已有 `libllaisys.so`。

```bash
cd /home/chenncy/llaisys
export PYTHONPATH="/home/chenncy/llaisys/python:$PYTHONPATH"
.venv/bin/python -m llaisys_py.server --model /home/chenncy/llaisys/DeepSeek-R1-Distill-Qwen-1___5B --port 8000
```

或使用脚本（会自动设置 PYTHONPATH 并选用 .venv）：

```bash
chmod +x scripts/run_server.sh
./scripts/run_server.sh /home/chenncy/llaisys/DeepSeek-R1-Distill-Qwen-1___5B 8000
```

命令行聊天客户端同样用 PYTHONPATH：

```bash
export PYTHONPATH="/home/chenncy/llaisys/python:$PYTHONPATH"
.venv/bin/python -m llaisys_py.server.chat_cli
```

注意：`.venv` 里仍需能 import torch、transformers、fastapi、uvicorn（若缺可单独装：`.venv/bin/pip install torch transformers fastapi uvicorn`）。

---

## 六、常见问题

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: No module named 'llaisys_py'` | 执行 `pip install -e ./python/` 或设置 `PYTHONPATH=python（包名已改为 llaisys_py）` 并从项目根运行 |
| `xmake: command not found` | 安装 xmake，见上文 |
| 服务启动报错找不到 .so / .dll | 先 `xmake` 再 `xmake install`，保证动态库在 `python/llaisys_py/libllaisys/` |
| “MODEL_PATH not set or not a directory” | 用 `--model /path/to/模型目录` 或 `export MODEL_PATH=...` |
| 请求返回 503 | 多为模型未加载成功，检查 --model 路径是否包含 safetensors 等文件 |
| pip install 报 Read timed out | 网络慢，可加 `--default-timeout=300` 或换国内镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 为什么必须用 .venv/bin/python？ | 系统 Python 禁止直接装包（externally-managed-environment），只有虚拟环境里的 Python 才能看到在 venv 里安装的包；用系统 `python` 会报 No module named 'llaisys_py' |

---

## 七、pip 安装超时或失败时

若 `pip install -e ./python/` 因网络超时失败，可尝试：

```bash
# 延长超时 + 使用清华镜像
.venv/bin/pip install --default-timeout=300 -i https://pypi.tuna.tsinghua.edu.cn/simple -e ./python/
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi uvicorn requests
```

或直接不安装 llaisys 包，用「五、未安装 llaisys 时用 PYTHONPATH 运行」的方式启动服务（需已 xmake install）。

---

## 八、快速命令汇总

```bash
# 终端 1：编译安装（仅首次或改 C++ 后需要）
cd /home/chenncy/llaisys
xmake && xmake install
pip install -e ./python/
pip install fastapi uvicorn requests

# 终端 1：启动服务（把 /path/to/模型 换成实际路径）
python -m llaisys_py.server --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B

# 终端 2：命令行聊天
python -m llaisys_py.server.chat_cli
# 或浏览器打开 http://127.0.0.1:8000/chat
```
