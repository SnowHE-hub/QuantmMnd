"""quantmind.line_a.evidence.ann_types — 公告标题的规则分类（03_final_plan_v2 第 5 节旁路与 F5；P2 第一版）.

只用标题做两件事：
- 事件类型（回购、减持、重组、立案调查……）；
- 阶段（预案 → 进展 → 实施 → 完成，或终止 / 撤回 / 解除；更正和补充单独标出）。
规则逐条带编号写在 RULES 里，命中哪条就记哪条，便于审计和以后与分类器对比（实验 X4，P4）。
分类只决定**研究优先级**，不进入任何组合排名。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (规则编号, 事件类型, 正则, 默认重要性)；按顺序匹配，第一条命中的类型生效。重要性：major / watch / routine
RULES: list[tuple[str, str, str, str]] = [
    ("T01", "investigation", r"立案|立案调查|立案告知", "major"),
    ("T02", "penalty", r"行政处罚|处罚决定|市场禁入|纪律处分|公开谴责", "major"),
    ("T03", "delisting_risk", r"终止上市|退市风险|退市整理|面值退市|可能被终止", "major"),
    ("T04", "st_change", r"风险警示", "major"),
    ("T05", "restructuring", r"重大资产重组|发行股份(?:及支付现金)?购买资产|重组上市|吸收合并", "major"),
    ("T06", "control_change", r"控制权.*变更|控股股东.*变更|实际控制人.*变更|表决权委托", "major"),
    ("T07", "clarification", r"澄清", "major"),
    ("T08", "regulatory_letter", r"问询函|关注函|监管函|警示函|监管工作函|督促函", "watch"),
    ("T09", "litigation", r"诉讼|仲裁", "watch"),
    ("T10", "guarantee", r"担保", "watch"),
    ("T11", "pledge", r"质押|解除质押|平仓", "watch"),
    ("T12", "holder_reduction", r"减持", "watch"),
    ("T13", "holder_increase", r"增持", "routine"),
    ("T14", "repurchase", r"回购", "routine"),
    ("T15", "equity_incentive", r"股权激励|限制性股票|股票期权|员工持股", "routine"),
    ("T16", "private_placement", r"向特定对象发行|非公开发行|定向增发|配股", "routine"),
    ("T17", "convertible", r"可转换公司债券|可转债", "routine"),
    ("T18", "unlock", r"限售股.*上市流通|解除限售|解禁", "watch"),
    ("T19", "performance_forecast", r"业绩预告|业绩预增|业绩预减|预亏|预盈", "watch"),
    ("T20", "performance_express", r"业绩快报", "watch"),
    ("T21", "periodic_report", r"年度报告|半年度报告|季度报告", "routine"),
    ("T22", "dividend", r"利润分配|权益分派|分红", "routine"),
    ("T23", "suspension", r"停牌|复牌", "watch"),
    ("T24", "abnormal_volatility", r"异常波动|严重异常", "watch"),
    ("T25", "major_contract", r"重大合同|中标|签订.*合同", "routine"),
    ("T26", "acquisition", r"收购|出售资产|购买资产|对外投资", "routine"),
    ("T27", "name_change", r"变更证券简称|变更公司名称", "routine"),
    ("T28", "shareholder_meeting", r"股东大会|股东会", "routine"),
    ("T29", "board_meeting", r"董事会.*决议|监事会.*决议", "routine"),
]
# 阶段规则：(规则编号, 阶段, 正则)；按顺序匹配
STAGE_RULES: list[tuple[str, str, str]] = [
    ("S01", "correction", r"更正|补充公告|补充更正|修订稿|更新后"),
    ("S02", "withdrawn", r"撤回|撤销申请"),
    ("S03", "terminated", r"终止|取消|放弃|不再|提前结束"),
    ("S04", "lifted", r"解除|撤销.*风险警示|复牌|恢复"),
    ("S05", "completed", r"完成|结果暨|实施结果|届满|期限届满|已完成|结束"),
    ("S06", "implemented", r"实施|权益分派实施|实施公告|开始|首次"),
    ("S07", "progress", r"进展|进度|延期|调整"),
    ("S08", "plan", r"预案|方案|计划|拟|草案|提示性公告|预披露"),
]
NOT_EVENT = r"摘要$|英文版|英文|半年度报告摘要|年度报告摘要|说明会|投资者关系活动"


@dataclass(frozen=True)
class TitleClass:
    event_type: str
    stage: str
    importance: str
    type_rule: str
    stage_rule: str
    is_summary_or_meta: bool


def classify_title(title: str) -> TitleClass:
    t = (title or "").strip()
    etype, imp, trule = "other", "routine", ""
    for rid, typ, pat, im in RULES:
        if re.search(pat, t):
            etype, imp, trule = typ, im, rid
            break
    stage, srule = "single", ""
    for rid, st, pat in STAGE_RULES:
        if re.search(pat, t):
            stage, srule = st, rid
            break
    # 终止 / 撤回类事件本身升级重要性（例如终止重组、终止回购）
    if stage in ("terminated", "withdrawn") and imp == "routine":
        imp = "watch"
    return TitleClass(etype, stage, imp, trule, srule, bool(re.search(NOT_EVENT, t)))


_PERIOD = re.compile(r"(20\d{2})\s*(年度|年半年度|半年度|年第[一二三四1-4]季度|年前三季度|年一季度|年三季度|年)")
# 只对标题里通常写明期间的事件类型生效；回购、增减持等标题常不带年份，强行区分会把一条链切碎
SUBJECT_TYPES = {"dividend", "periodic_report", "performance_forecast", "performance_express", "equity_incentive"}


def subject_tag(title: str, event_type: str) -> str:
    """事项标识：同一类事件里区分不同事项（例如 2023年度分红 与 特别分红、两期回购）。只对 SUBJECT_TYPES 生效。"""
    if event_type not in SUBJECT_TYPES:
        return ""
    t = title or ""
    tags = []
    if "特别" in t:
        tags.append("特别")
    t2 = re.sub(r"(20\d{2})\s*年\s*年度", r"\g<1>年度", t)          # "2022年年度" → "2022年度"
    m = _PERIOD.search(t2)
    if m:
        g = m.group(2)
        tags.append(m.group(1) + ("" if g == "年" else g if g == "年度" else g.lstrip("年")))
    if event_type == "equity_incentive":
        k = re.search(r"(20\d{2})\s*年.*?(限制性股票|股票期权|员工持股)", t)
        if k:
            tags = [k.group(1) + k.group(2)]
    return "-".join(tags)
