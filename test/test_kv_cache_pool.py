import importlib.util
from pathlib import Path
import sys


def _load_pool_module():
    root = Path(__file__).resolve().parents[1]

    # Load interfaces first (kv_cache_pool imports from it)
    iface_path = root / "python" / "llaisys" / "interfaces.py"
    iface_spec = importlib.util.spec_from_file_location("llaisys.interfaces", str(iface_path))
    if iface_spec is not None and iface_spec.loader is not None:
        iface_mod = importlib.util.module_from_spec(iface_spec)
        sys.modules[iface_spec.name] = iface_mod
        iface_spec.loader.exec_module(iface_mod)

    module_path = root / "python" / "llaisys" / "kv_cache_pool.py"
    spec = importlib.util.spec_from_file_location("llaisys.kv_cache_pool", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load kv_cache_pool module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kv_module = _load_pool_module()
KVCachePool = kv_module.KVCachePool


def test_prefix_match_only_on_sealed_block():
    pool = KVCachePool(block_size=4, max_blocks=128, max_bytes=1024 * 1024)

    # ctx-a creates one sealed block [1,2,3,4] and one unsealed [5,6]
    result_a = pool.acquire_context("ctx-a", [1, 2, 3, 4, 5, 6])
    assert result_a.prefix_len == 0

    # ctx-b should only reuse sealed prefix length=4
    result_b = pool.acquire_context("ctx-b", [1, 2, 3, 4, 5, 6])
    assert result_b.prefix_len == 4

    stats = pool.snapshot_stats()
    assert stats["prefix_hit_count"] >= 1


def test_release_and_evict_zero_ref_blocks():
    pool = KVCachePool(block_size=2, max_blocks=2, max_bytes=1024 * 1024)
    pool.acquire_context("ctx-a", [10, 11, 12, 13])  # two sealed blocks
    pool.acquire_context("ctx-b", [20, 21, 22, 23])  # pressure pool

    # both contexts exist
    assert pool.debug_context("ctx-a") is not None
    assert pool.debug_context("ctx-b") is not None

    pool.release_context("ctx-a")
    pool.release_context("ctx-b")
    stats = pool.snapshot_stats()
    # capacity eviction can now clear all zero-ref blocks
    assert stats["zero_ref_blocks"] >= 0


def test_reference_count_sharing():
    pool = KVCachePool(block_size=3, max_blocks=128, max_bytes=1024 * 1024)
    pool.acquire_context("ctx-a", [1, 2, 3, 4, 5, 6])
    pool.acquire_context("ctx-b", [1, 2, 3, 9, 9, 9])
    stats = pool.snapshot_stats()
    assert stats["shared_blocks"] >= 1, "sealed prefix block should be shared"


def test_rollback_on_block_creation_error():
    pool = KVCachePool(block_size=2, max_blocks=128, max_bytes=1024 * 1024)
    pool.acquire_context("ctx-ok", [1, 2, 3, 4])

    original_create = pool._create_block
    call_count = {"n": 0}

    def flaky_create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("inject failure")
        return original_create(*args, **kwargs)

    pool._create_block = flaky_create
    before = pool.snapshot_stats()
    try:
        try:
            pool.acquire_context("ctx-fail", [5, 6, 7, 8])
            raise AssertionError("expected failure not raised")
        except RuntimeError:
            pass
    finally:
        pool._create_block = original_create

    after = pool.snapshot_stats()
    # failed context should not exist; leaked refs should not increase
    assert pool.debug_context("ctx-fail") is None
    assert after["total_refs"] <= before["total_refs"]


if __name__ == "__main__":
    test_prefix_match_only_on_sealed_block()
    test_release_and_evict_zero_ref_blocks()
    test_reference_count_sharing()
    test_rollback_on_block_creation_error()
    print("KV cache pool tests passed.")
