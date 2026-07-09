# llaisys-server

Standalone server package for Project #3.

## Install order

1. Install core package:

```bash
python3 -m pip install -e /home/vankari/code/llaisys/python/llaisyscore --user --break-system-packages
```

2. Install server package:

```bash
python3 -m pip install -e /home/vankari/code/llaisys/python/server-project --user --break-system-packages
```

## Run server

```bash
cd /home/vankari/code/llaisys/python
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```
### attention! the --break-system-packages was enabled on public server ,it should be disabled in personal computer.