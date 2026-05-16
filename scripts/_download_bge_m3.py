"""预下载 BGE-M3 嵌入模型（后台执行，一次性）."""
import sys
import time
from pathlib import Path

print("开始下载 BAAI/bge-m3 模型...")
print("预计大小：约 1.1GB，时间取决于网速")
print()

try:
    from FlagEmbedding import BGEM3FlagModel
    t0 = time.time()
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        device="cpu",  # 下载时用 CPU 即可
    )
    elapsed = time.time() - t0
    # 测试 encode
    test_emb = model.encode(["测试文本"], batch_size=1, max_length=64,
                             return_dense=True, return_sparse=False, return_colbert_vecs=False)
    dim = len(test_emb["dense_vecs"][0])
    print(f"✅ BGE-M3 下载并加载成功（{elapsed:.1f}s），向量维度={dim}")
    Path("/tmp/bge_m3_ready.flag").write_text("ok")
except Exception as e:
    print(f"❌ 下载失败：{e}")
    sys.exit(1)
