"""临时验证脚本：确认 BoardModelRouter 不再降级 fallback."""
import sys
import logging
sys.path.insert(0, ".")
logging.disable(logging.CRITICAL)

from quantmind.models.factor_model import FactorModel
from quantmind.models.board_router import BoardModelRouter

print("=" * 60)
print("1. 模型文件 direction 验证")
print("=" * 60)
all_dir_ok = True
for name in ["lgbm_v6_main", "lgbm_v6_gem", "lgbm_v6_star"]:
    m = FactorModel.load(f"models/{name}.pkl")
    direction = getattr(m, "direction", "N/A")
    ic_mean   = getattr(m, "ic_mean",   "N/A")
    ok = direction == 1
    all_dir_ok = all_dir_ok and ok
    flag = "✅" if ok else "❌"
    print(f"  {flag} {name}: direction={direction}  ic_mean={ic_mean}")

print()
print("=" * 60)
print("2. BoardModelRouter.status()")
print("=" * 60)
router = BoardModelRouter()
for board, status in router.status().items():
    print(f"  {board}: {status}")

print()
print("=" * 60)
print("3. BoardModelRouter.get_routing_status() — 路由决策")
print("=" * 60)
routing = router.get_routing_status()
all_dedicated = True
for board, info in routing.items():
    is_fb  = info["is_fallback"]
    d      = info["direction"]
    reason = info["reason"]
    flag   = "⚠ FALLBACK" if is_fb else "✅ 专用模型"
    all_dedicated = all_dedicated and not is_fb
    print(f"  {board}: {flag} | direction={d} | {reason}")

print()
if all_dir_ok and all_dedicated:
    print("✅ 全部通过：三个板块模型 direction=+1，BoardModelRouter 不再降级 fallback")
else:
    if not all_dir_ok:
        print("❌ 存在 direction≠+1 的模型")
    if not all_dedicated:
        print("❌ BoardModelRouter 仍在使用 fallback")
    sys.exit(1)
