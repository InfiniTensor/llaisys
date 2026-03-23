"""接口定义 - 解耦调度器、服务、KVCache 池之间的依赖"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Sequence

if TYPE_CHECKING:
    from llaisys.kv_cache_pool import AcquireResult


class IKVCachePool(ABC):
    """KVCache 池接口

    调度器可以通过此接口查询 KV 状态，而不需要知道具体实现。
    """

    @property
    @abstractmethod
    def block_size(self) -> int:
        """每个 block 的 token 数量"""
        pass

    @abstractmethod
    def query_prefix_len(self, tokens: Sequence[int]) -> int:
        """查询前缀命中长度（只读，不修改状态）

        Args:
            tokens: 待查询的 token 序列

        Returns:
            命中的前缀长度（token 数量）
        """
        pass

    @abstractmethod
    def acquire_context(self, context_id: str, tokens: Sequence[int]) -> "AcquireResult":
        """获取/创建上下文，返回匹配的前缀长度

        Args:
            context_id: 上下文/会话 ID
            tokens: 当前请求的完整 token 序列

        Returns:
            AcquireResult，包含 context_id 和 prefix_len
        """
        pass

    @abstractmethod
    def update_context(self, context_id: str, tokens: Sequence[int]) -> None:
        """更新上下文的 token 序列（生成结束后调用）"""
        pass

    @abstractmethod
    def release_context(self, context_id: str) -> None:
        """释放上下文"""
        pass

    @abstractmethod
    def memory_pressure(self) -> float:
        """返回 KV 内存压力值 (0.0~1.0)

        取 used_blocks/max_blocks 和 used_bytes/max_bytes 的较大值。
        调度器可用此值做流控决策。
        """
        pass

    @abstractmethod
    def snapshot_stats(self) -> Dict[str, float]:
        """获取统计信息快照"""
        pass


class IInferenceService(ABC):
    """推理服务接口

    调度器依赖此接口进行任务分发和执行。
    """

    @abstractmethod
    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """非流式生成

        Args:
            payload: 请求参数（prompt, session_id, max_new_tokens 等）

        Returns:
            生成结果（response, session_id, usage 等）
        """
        pass

    @abstractmethod
    def stream(self, payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """流式生成

        Args:
            payload: 请求参数

        Yields:
            流式输出的每个 chunk（delta, done, session_id 等）
        """
        pass

    @abstractmethod
    def request_stop(self, session_id: str) -> bool:
        """请求停止生成

        Args:
            session_id: 要停止的会话 ID

        Returns:
            是否成功发送停止信号
        """
        pass

    @abstractmethod
    def kv_debug_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取 KV 调试快照

        Args:
            session_id: 可选，指定会话 ID

        Returns:
            调试信息字典
        """
        pass

    @property
    @abstractmethod
    def kv_pool(self) -> IKVCachePool:
        """暴露 KVCache 池给调度器查询"""
        pass

    def generate_packed_non_stream(
        self, payloads: Sequence[Dict[str, Any]]
    ) -> Optional[Sequence[Dict[str, Any]]]:
        """批量非流式生成（可选实现）

        Args:
            payloads: 多个请求参数

        Returns:
            批量生成结果，如果不支持则返回 None
        """
        return None

    def tokenize_for_routing(
        self, payload: Dict[str, Any]
    ) -> Optional[Sequence[int]]:
        """为 KV 感知路由进行轻量级 tokenize（可选实现）

        调度器可以调用此方法将请求转换为 token 序列，
        用于查询各 worker 的 KV 命中情况。

        Args:
            payload: 请求参数（prompt, messages 等）

        Returns:
            token ids 序列，如果无法 tokenize 则返回 None
        """
        return None

    def prepare_batch(
        self, payloads: Sequence[Dict[str, Any]]
    ) -> Optional[Any]:
        """准备���式批处理：prefill 所有序列，返回 BatchState（可选实现）

        Args:
            payloads: 多个请求参数

        Returns:
            BatchState 对象，如果不支持则返回 None
        """
        return None

    def step_batch(self, state: Any) -> Optional[Sequence[Any]]:
        """执行一步批量 decode，返回每个序列的 StepResult（可选实现）

        Args:
            state: prepare_batch 返回的 BatchState

        Returns:
            StepResult 列表，如果不支持则返回 None
        """
        return None

    def finalize_sequence(self, state: Any, seq_index: int) -> None:
        """完成单个序列：保存消息历史，清理状态（可选实现）

        Args:
            state: BatchState 对象
            seq_index: 序列在 batch 中的索引
        """
        pass
