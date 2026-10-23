"""公告分类器（T-049 采用 MacBERT）：降级路径、规则类型并入、与 TitleClass 字段兼容。"""
from __future__ import annotations

import pytest

from quantmind.line_a.evidence import ann_classifier as ac
from quantmind.line_a.evidence.ann_types import classify_title


def test_degraded_without_model_matches_rule():
    c = ac.TitleClassifier(None, None)
    titles = ["关于收到中国证监会立案告知书的公告", "关于为子公司提供担保的进展公告", "关于召开2026年第一次临时股东大会的通知"]
    out = c.classify(titles)
    assert c.degraded
    for t, o in zip(titles, out, strict=True):
        r = classify_title(t)
        assert (o.event_type, o.importance, o.stage, o.source) == (r.event_type, r.importance, r.stage, "rule")


def test_model_types_merge_with_rule_types(monkeypatch):
    c = ac.TitleClassifier(None, {"repurchase": 0.5})
    fake = {"关于终止回购股份方案的公告": ("repurchase",), "关于为子公司提供担保的公告": (), "关于召开业绩说明会的公告": ()}
    monkeypatch.setattr(c, "model_types", lambda titles, batch=256: [fake[t] for t in titles])
    rep, gua, meet = c.classify(list(fake))
    assert rep.types == ("repurchase",) and rep.event_type == "repurchase" and rep.stage == "terminated"
    assert rep.importance == "watch"                    # 终止的回购升级为 watch（与规则版一致）
    assert rep.source == "model:macbert" and rep.type_rule == "M:macbert"
    assert gua.event_type == "guarantee" and gua.source == "model:macbert+rule"     # X4 以外的类型沿用规则
    assert meet.event_type == "other" and meet.importance == "routine"


def test_primary_type_is_most_important(monkeypatch):
    c = ac.TitleClassifier(None, None)
    monkeypatch.setattr(c, "model_types", lambda titles, batch=256: [("repurchase", "investigation")])
    (o,) = c.classify(["关于公司被立案调查及回购进展的公告"])
    assert o.event_type == "investigation" and o.importance == "major" and set(o.types) == {"repurchase", "investigation"}


@pytest.mark.skipif(not ac.ADOPTION.exists(), reason="no adoption config")
def test_adopted_model_loads_and_classifies():
    c = ac.TitleClassifier.from_adoption()
    if not (c.model_dir and (c.model_dir / "config.json").exists()):
        pytest.skip("model dir not on this machine")
    a, b = c.classify(["关于收到中国证券监督管理委员会立案告知书的公告", "关于对深圳证券交易所问询函的回复公告"])
    assert c.degraded is None
    assert "investigation" in a.types and a.importance == "major"
    assert "regulatory_letter" not in b.types           # 指南 v2：对函件的回复不算收到监管函
