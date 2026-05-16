"""最小化 KB 状态检查脚本（无 loguru/quantmind 导入）."""
import sys
import time
t0 = time.time()

try:
    import chromadb
    client = chromadb.PersistentClient(path=".cache/chromadb")
    colls = client.list_collections()
    print(f"ChromaDB OK ({time.time()-t0:.1f}s)")
    print(f"Collections: {len(colls)}")
    for c in colls:
        print(f"  {c.name}: {c.count()} docs")
    if not colls:
        print("  (空 - 无数据)")
except Exception as e:
    print(f"ERROR: {e}")
