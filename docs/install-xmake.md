# 在 Linux 服务器上安装 Xmake

## 方法一：官方安装脚本（推荐）

```bash
# 下载并运行安装脚本（会安装到 ~/.local/bin）
bash <(curl -fsSL https://raw.githubusercontent.com/xmake-io/xmake/master/scripts/get.sh)
```

若服务器没有 curl，可用 wget：

```bash
bash <(wget -qO- https://raw.githubusercontent.com/xmake-io/xmake/master/scripts/get.sh)
```

安装完成后，把 xmake 加入当前会话的 PATH：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

验证：

```bash
xmake --version
```

若每次登录都要用 xmake，可写入 `~/.bashrc`：

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 方法二：pip 安装（仅 xmake 本体，不包含 C++ 工具链）

xmake 也提供 PyPI 包，但编译 LLAISYS 还需要系统有 C++ 编译器（g++/clang）：

```bash
# 在 venv 里装
.venv/bin/pip install xmake
# 然后用 .venv/bin/xmake
```

若用系统 pip 需加 `--break-system-packages` 或改用 venv。

## 依赖：C++ 编译器

xmake 只是构建工具，实际编译需要编译器。Ubuntu/Debian 上：

```bash
sudo apt update
sudo apt install build-essential
```

CentOS/RHEL 上：

```bash
sudo yum groupinstall "Development Tools"
```

或

```bash
sudo dnf install gcc-c++
```
