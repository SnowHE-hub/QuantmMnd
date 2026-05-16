import os
print("HF_ENDPOINT:", os.getenv("HF_ENDPOINT", "(not set)"))
print("HF_HUB_URL:", os.getenv("HF_HUB_URL", "(not set)"))

# 测试 HuggingFace 连通性
import urllib.request
import time
for url in ["https://huggingface.co", "https://hf-mirror.com"]:
    try:
        t0 = time.time()
        r = urllib.request.urlopen(url, timeout=5)
        print(f"{url}: OK ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"{url}: FAIL ({e})")
