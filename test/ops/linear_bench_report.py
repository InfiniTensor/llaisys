"""
根据 linear_bench.py 生成的两份 JSON 对比并输出性能提升报告。
用法: python test/ops/linear_bench_report.py <基准.json> <优化后.json>
例如: python test/ops/linear_bench_report.py single.json multi.json
      python test/ops/linear_bench_report.py baseline.json optimized.json
"""
import json
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shape_key(r):
    return (tuple(r["out_shape"]), tuple(r["x_shape"]), tuple(r["w_shape"]), r["dtype"])


def main():
    if len(sys.argv) != 3:
        print("用法: python linear_bench_report.py <基准.json> <对比.json>", file=sys.stderr)
        print("示例: python test/ops/linear_bench_report.py single.json multi.json", file=sys.stderr)
        sys.exit(1)
    base_path = sys.argv[1]
    opt_path = sys.argv[2]
    base_name = base_path.replace(".json", "").split("/")[-1]
    opt_name = opt_path.replace(".json", "").split("/")[-1]

    base = load_json(base_path)
    opt = load_json(opt_path)

    base_results = {shape_key(r): r for r in base["results"]}
    opt_results = {shape_key(r): r for r in opt["results"]}

    # 按 base 的顺序输出，若 opt 无对应项则跳过
    lines = []
    lines.append("=" * 70)
    lines.append("Linear 性能对比报告")
    lines.append("=" * 70)
    lines.append(f"  基准: {base_path}  (e.g. 优化前 / 单线程)")
    lines.append(f"  对比: {opt_path}  (e.g. 优化后 / 多线程)")
    lines.append("")
    lines.append(f"{'shape':<28} {'dtype':<6} {'基准(ms)':>12} {'对比(ms)':>12} {'加速比':>10}")
    lines.append("-" * 70)

    for r in base["results"]:
        key = shape_key(r)
        if key not in opt_results:
            continue
        o = opt_results[key]
        base_ms = r["lla_ms"]
        opt_ms = o["lla_ms"]
        speedup = base_ms / opt_ms if opt_ms > 0 else 0
        shape_str = f"{r['out_shape']} @ {r['w_shape']}"
        lines.append(f"{shape_str:<28} {r['dtype']:<6} {base_ms:>12.2f} {opt_ms:>12.2f} {speedup:>10.2f}x")

    lines.append("-" * 70)
    lines.append("说明: 加速比 = 基准耗时 / 对比耗时，>1 表示对比版本更快。")
    lines.append("=" * 70)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
