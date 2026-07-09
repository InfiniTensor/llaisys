import ctypes

import torch

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.models.kvcachepool import KVCachePool


class _DummyModel:
    def __init__(self):
        self.num_hidden_layers = 2
        self.num_key_value_heads = 2
        self.per_kvhead_dim = 4
        self.data_type = llaisys.DataType.F32
        self.device = llaisys.DeviceType.CPU
        self.device_id = 0
        self.max_position_embeddings = 64


def _zero_init_slot(slot, model):
    for i in range(model.num_hidden_layers):
        zero_tensor = torch.zeros(
            (slot.capacity, model.num_key_value_heads, model.per_kvhead_dim),
            dtype=torch.float32,
            device="cpu",
        ).contiguous()
        LIB_LLAISYS.tensorLoad(slot.kcache_array[i], zero_tensor.data_ptr())
        LIB_LLAISYS.tensorLoad(slot.vcache_array[i], zero_tensor.data_ptr())


def test_prepare_and_commit_session():
    model = _DummyModel()
    pool = KVCachePool(model, max_sessions=4)
    try:
        slot = pool.prepare_session("s1", [1, 2, 3], max_new_tokens=8)
        _zero_init_slot(slot, model)
        assert slot.past_len == 0

        pool.commit_session("s1", [1, 2, 3, 4])
        slot = pool.prepare_session("s1", [1, 2, 3, 4, 9], max_new_tokens=4)
        assert slot.past_len == 4
    finally:
        pool.clear()


def test_find_best_prefix_with_trie():
    model = _DummyModel()
    pool = KVCachePool(model, max_sessions=4)
    try:
        s1 = pool.prepare_session("s1", [10, 20, 30], max_new_tokens=8)
        s2 = pool.prepare_session("s2", [10, 21, 31], max_new_tokens=8)
        _zero_init_slot(s1, model)
        _zero_init_slot(s2, model)

        pool.commit_session("s1", [10, 20, 30, 40])
        pool.commit_session("s2", [10, 21, 31, 41])

        best = pool.find_best_prefix([10, 20, 99])
        assert best is not None
        assert best.session_id == "s1"
        assert best.matched_tokens == 2

        best_excluded = pool.find_best_prefix([10, 20, 99], exclude_session_id="s1")
        assert best_excluded is not None
        assert best_excluded.session_id == "s2"
        assert best_excluded.matched_tokens == 1
    finally:
        pool.clear()


def test_reset_and_clear():
    model = _DummyModel()
    pool = KVCachePool(model, max_sessions=4)

    s1 = pool.prepare_session("s1", [1, 2], max_new_tokens=8)
    s2 = pool.prepare_session("s2", [1, 3], max_new_tokens=8)
    _zero_init_slot(s1, model)
    _zero_init_slot(s2, model)

    pool.commit_session("s1", [1, 2, 4])
    pool.commit_session("s2", [1, 3, 5])

    pool.reset_session("s1")
    best = pool.find_best_prefix([1, 2, 9])
    assert best is not None
    assert best.session_id == "s2"
    assert best.matched_tokens == 1

    pool.clear()
    assert pool.find_best_prefix([1, 3]) is None


def test_modify_history_then_commit():
    model = _DummyModel()
    pool = KVCachePool(model, max_sessions=4)
    try:
        s1 = pool.prepare_session("s1", [1, 2, 3], max_new_tokens=8)
        s2 = pool.prepare_session("s2", [1, 7, 8], max_new_tokens=8)
        _zero_init_slot(s1, model)
        _zero_init_slot(s2, model)

        pool.commit_session("s1", [1, 2, 3, 4])
        pool.commit_session("s2", [1, 7, 8, 9])

        slot = pool.modify_session_tokens("s1", [1, 2, 8])
        assert slot.past_len == 2

        best_after_modify = pool.find_best_prefix([1, 2, 3, 99])
        assert best_after_modify is not None
        assert best_after_modify.session_id == "s1"
        assert best_after_modify.matched_tokens == 2

        pool.commit_session("s1", [1, 2, 8, 10])
        best_after_commit = pool.find_best_prefix([1, 2, 8, 11])
        assert best_after_commit is not None
        assert best_after_commit.session_id == "s1"
        assert best_after_commit.matched_tokens == 3
    finally:
        pool.clear()


if __name__ == "__main__":
    test_prepare_and_commit_session()
    test_find_best_prefix_with_trie()
    test_reset_and_clear()
    test_modify_history_then_commit()
    print("\033[92mTest passed!\033[0m\n")
