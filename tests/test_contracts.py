"""contracts 层单测：HorizonRegistry / ModelRegistry / Recommendation+Outcome schema."""
import pytest

from quantmind.contracts import (
    HorizonRegistry, ModelRecord, RecommendationContract, OutcomeContract)
from quantmind.contracts import model_registry as MR


# ---- HorizonRegistry ----
def test_horizon_single_source():
    assert HorizonRegistry.get("short").days == 12
    assert HorizonRegistry.get("robust").days == 21
    assert HorizonRegistry.get("long").days == 63
    assert HorizonRegistry.by_days(63).name == "long"


def test_horizon_product_lines_exclude_robust():
    names = {h.name for h in HorizonRegistry.product_lines()}
    assert names == {"short", "long"}            # robust 仅稳健性验证，非产品线
    assert HorizonRegistry.get("robust").is_product_line is False


def test_horizon_rebalance_step():
    assert HorizonRegistry.get("short").rebalance_step == 3   # ceil(12/5)
    assert HorizonRegistry.get("long").rebalance_step == 13   # ceil(63/5)


def test_horizon_unknown_raises():
    with pytest.raises(KeyError):
        HorizonRegistry.get("medium")


# ---- ModelRegistry ----
def test_model_register_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(MR, "REGISTRY_PATH", tmp_path / "reg.json")
    rec = ModelRecord(model_id="t1", label="forward_return_12d", horizon="short",
                      feature_set="full_35_16_158", data_version="v6",
                      gate_status="research_candidate_pending_nav")
    MR.register(rec)
    got = MR.get("t1")
    assert got.gate_status == "research_candidate_pending_nav"
    assert got.horizon == "short"


def test_model_bad_gate_status_raises():
    with pytest.raises(ValueError):
        ModelRecord(model_id="x", label="l", horizon="short",
                    feature_set="f", data_version="v6", gate_status="bogus")


def test_set_gate_status(tmp_path, monkeypatch):
    monkeypatch.setattr(MR, "REGISTRY_PATH", tmp_path / "reg.json")
    MR.register(ModelRecord(model_id="t2", label="l", horizon="short",
                            feature_set="f", data_version="v6"))
    MR.set_gate_status("t2", "production")
    assert MR.get("t2").gate_status == "production"


def test_register_from_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(MR, "REGISTRY_PATH", tmp_path / "reg.json")
    manifest = {"horizon": "12d", "label": "forward_return_12d", "feature_set": "full_35_16_158",
                "data_version": "v6", "git_sha": "abc123",
                "extra": {"model": "ridge", "train_seconds": 208, "refit": "quarterly"}}
    rec = MR.register_from_manifest(manifest)
    assert rec is not None and rec.horizon == "short"
    assert rec.gate_status == "research_candidate"   # 自动注册默认不抬级
    # 非模型 manifest → 不注册
    assert MR.register_from_manifest({"horizon": "weekly", "extra": {}}) is None


# ---- RecommendationContract / OutcomeContract ----
def test_recommendation_validate_ok():
    r = RecommendationContract(
        as_of="2026-06-01", ticker="000001.SZ", horizon_name="short", horizon_days=12,
        label_col="forward_return_12d", model_id="ridge_full_12d_v6_seed",
        feature_set_id="full_35_16_158", score_raw=0.1, score_neutralized=0.08,
        rank=1, weight=0.05, liquidity_bucket="b0", cost_model_id="wf_costs_v1",
        gate_status="research_candidate_pending_nav").validate()
    assert r.to_dict()["ticker"] == "000001.SZ"
    assert RecommendationContract.from_dict(r.to_dict()) == r


def test_recommendation_horizon_mismatch_raises():
    with pytest.raises(ValueError):
        RecommendationContract(
            as_of="2026-06-01", ticker="x", horizon_name="short", horizon_days=63,  # 不匹配
            label_col="forward_return_12d", model_id="m", feature_set_id="f",
            score_raw=0.0, score_neutralized=0.0, rank=1, weight=0.05,
            liquidity_bucket="b0", cost_model_id="c", gate_status="research_candidate").validate()


def test_outcome_validate():
    o = OutcomeContract(as_of="2026-06-01", ticker="000001.SZ", horizon_days=12,
                        model_id="m", entry_price=10.0, exit_price=10.5,
                        actual_return_12d=0.05, fill_status="filled").validate()
    assert o.to_dict()["net_return"] is None
    with pytest.raises(ValueError):
        OutcomeContract(as_of="x", ticker="y", horizon_days=12, model_id="m",
                        entry_price=1, exit_price=1, fill_status="bogus").validate()
