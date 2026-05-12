"""
BGE-M3 嵌入模块 — Dense + ColBERT Token 向量

P2 功能：
- encode_dense: 批量生成 dense 向量（存储用）
- encode_colbert: 生成 token 级向量（查询时精排用）
- colbert_score: ColBERT late interaction 打分

懒加载 + 用完可卸载。环境变量 ROLLING_MEMORY_KEEP_MODEL=1 保持常驻。
"""
import threading

import numpy as np


class EmbeddingEngine:
    """BGE-M3 懒加载单例，线程安全"""

    _instance = None
    _lock = threading.Lock()
    _model = None

    DIM = 1024  # BGE-M3 dense 输出维度

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            import sys
            print("[Rolling Memory] 加载 BGE-M3 ...", file=sys.stderr, flush=True)
            self._model = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=True,
                device="cuda",
                use_safetensors=True,
            )
            print("[Rolling Memory] BGE-M3 加载完成", file=sys.stderr, flush=True)
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Dense 编码（存储用）──────────────────────────────

    def encode_dense(self, texts: list[str], batch_size: int = 24,
                     max_length: int = 2000) -> np.ndarray:
        """批量生成 dense 向量，返回 (N, 1024) float32 L2归一化"""
        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        out = self.model.encode(
            texts, batch_size=batch_size, max_length=max_length,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        mat = np.array(out["dense_vecs"], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms

    # ── ColBERT 编码（查询时精排用）─────────────────────

    def encode_colbert(self, texts: list[str], batch_size: int = 8,
                       max_length: int = 2000) -> list[np.ndarray]:
        """生成 token 级向量，返回 list[(seq_len, 1024)] float32。
        仅在查询时对 top-K 候选调用，不用于存储。"""
        if not texts:
            return []
        out = self.model.encode(
            texts, batch_size=batch_size, max_length=max_length,
            return_dense=False, return_sparse=False, return_colbert_vecs=True,
        )
        return [np.array(v, dtype=np.float32) for v in out["colbert_vecs"]]

    # ── ColBERT 打分 ──────────────────────────────────

    @staticmethod
    def colbert_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        """ColBERT late interaction: 对每个 query token 取 max similarity，求平均。
        query_tokens: (Q, 1024), doc_tokens: (D, 1024)"""
        # 归一化
        q = query_tokens / (np.linalg.norm(query_tokens, axis=1, keepdims=True) + 1e-9)
        d = doc_tokens / (np.linalg.norm(doc_tokens, axis=1, keepdims=True) + 1e-9)
        # similarity matrix: (Q, D)
        sim = q @ d.T
        # max-sim over doc tokens for each query token, then mean
        return float(sim.max(axis=1).mean())

    # ── 工具方法 ──────────────────────────────────────

    @staticmethod
    def vec_to_blob(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_vec(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def unload(self):
        """卸载模型释放 GPU 显存"""
        import gc, sys
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("[Rolling Memory] BGE-M3 已卸载", file=sys.stderr, flush=True)


def get_engine() -> EmbeddingEngine:
    return EmbeddingEngine()
