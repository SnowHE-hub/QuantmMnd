"""1a：token 轮换验证。新 token 通 + 旧 token(历史中)死。全程不打印任何 token 字符。

输出仅：api / status / 记录数 / 各 token 用 sha256[:8] 标识（非 token 本身）。
"""
import hashlib
import subprocess
import sys
import re
from pathlib import Path

REPO = Path("/home/lenovo/projects/quantmind")
sys.path.insert(0, str(REPO))


def _h(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()[:8]


def _read_current_token() -> str:
    for f in (REPO / "api_key.txt", REPO / ".env"):
        if f.exists():
            for line in f.read_text().splitlines():
                if "TUSHARE_TOKEN" in line:
                    m = re.search(r"[0-9a-fA-F]{30,}", line)
                    if m:
                        return m.group(0)
    raise SystemExit("未找到当前 token")


def _history_tokens() -> set[str]:
    """从 git 历史(9 个曾含 token 的文件)提取所有 40+ hex token，去重。"""
    files = ["HANDOVER.md", "README.md", "scripts/data_pipeline/patch_adj_close.py",
             "scripts/data_pipeline/prefetch_bulk_with_hi.py", "scripts/data_pipeline/setup_api_config.sh",
             "scripts/data_pipeline/step1_build_alpha_universe.py", "scripts/data_pipeline/step2_download_prices.py",
             "scripts/data_pipeline/step4_download_fundamentals.py", "scripts/verify_tokens.py"]
    toks = set()
    blob = subprocess.run(["git", "-C", str(REPO), "log", "--all", "-p", "--", *files],
                          capture_output=True, text=True).stdout
    for m in re.finditer(r"[0-9a-f]{40,}", blob):
        toks.add(m.group(0))
    return toks


def _test_token(tok: str) -> tuple[bool, str, int]:
    """调 stock_basic。返回 (ok, msg, n_rows)。鉴权失败→ok=False。"""
    try:
        import tushare as ts
        api = ts.pro_api(tok)
        df = api.stock_basic(exchange="", list_status="L", fields="ts_code,name")
        return True, "OK", 0 if df is None else len(df)
    except Exception as e:
        return False, str(e)[:90], 0


def main():
    cur = _read_current_token()
    print(f"=== 1a token 轮换验证 (token 以 sha256[:8] 标识，不打印明文) ===")
    ok, msg, n = _test_token(cur)
    print(f"[新 token {_h(cur)}] stock_basic → {'通过 ✅' if ok else '失败 ❌'}  rows={n}  msg={msg if not ok else ''}")
    new_ok = ok

    hist = _history_tokens() - {cur}
    print(f"\n历史中 distinct 旧 token: {len(hist)} 个")
    all_dead = True
    for tok in sorted(hist):
        ok, msg, n = _test_token(tok)
        status = "仍可用 ⚠⚠" if ok else "已死 ✅"
        if ok:
            all_dead = False
        print(f"[旧 token {_h(tok)}] stock_basic → {status}  {('rows='+str(n)) if ok else msg}")

    print(f"\n=== 结论: 新token通={new_ok}  全部旧token死={all_dead} → "
          f"{'PASS ✅' if (new_ok and all_dead) else 'FAIL ❌(停下,联系Tushare客服)'} ===")
    sys.exit(0 if (new_ok and all_dead) else 1)


if __name__ == "__main__":
    main()
