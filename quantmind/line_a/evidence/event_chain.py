"""quantmind.line_a.evidence.event_chain — 事件状态机（03_final_plan_v2 第 5 节 F5(b)3；P2 第一版）.

同一只股票、同一类事件的公告串成链：
    plan（预案 / 计划）→ progress（进展，可多次）→ implemented（实施）→ completed（完成）
    任一阶段 → terminated（终止）/ withdrawn（撤回）   —— 终态
    lifted（解除）：针对"状态类"事件（停牌、风险警示、质押）把当前状态解除 —— 终态
    correction（更正 / 补充）：不改变阶段，挂在链上最近一条公告后面（supersedes），保留原公告
    single：没有阶段信息的单条公告（例如问询函），自成一条链
链进入终态后，同类事件的下一条非更正公告开新链。
链的键是（股票, 事件类型, 事项标识）：事项标识来自标题里的期间或事项（例如"2023年度"、"特别分红"），
这样同一家公司同时进行的年度分红和特别分红不会被串成一条链（2026-10-21 第一次真实切片发现的问题）。
状态按公告的可用时间排序计算，所以任何时点的状态只用该时点之前已发布的公告（时点正确）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

ORDER = {"plan": 0, "progress": 1, "implemented": 2, "completed": 3}
TERMINAL = {"completed", "terminated", "withdrawn", "lifted"}


@dataclass
class Ann:
    ann_id: str
    ticker: str
    event_type: str
    stage: str
    available_at: str          # ISO 时间（或 YYYYMMDD），可比较大小
    title: str = ""
    subject: str = ""          # 事项标识（ann_types.subject_tag）；空 = 不区分


@dataclass
class Chain:
    chain_id: str
    ticker: str
    event_type: str
    state: str
    opened_at: str
    updated_at: str
    subject: str = ""
    ann_ids: list[str] = field(default_factory=list)
    corrections: list[tuple[str, str]] = field(default_factory=list)    # (更正公告, 被更正的公告)
    anomalies: list[str] = field(default_factory=list)                  # 阶段倒退等，留给校验器

    @property
    def closed(self) -> bool:
        return self.state in TERMINAL


def build_chains(anns: list[Ann]) -> list[Chain]:
    chains: list[Chain] = []
    open_: dict[tuple[str, str, str], Chain] = {}
    n: dict[tuple[str, str, str], int] = {}
    for a in sorted(anns, key=lambda x: (x.available_at, x.ann_id)):
        k = (a.ticker, a.event_type, a.subject)
        cur = open_.get(k)
        if a.stage == "correction":
            if cur is None:                       # 更正的对象链已关闭：挂到最近一条同类链
                cur = next((c for c in reversed(chains) if (c.ticker, c.event_type, c.subject) == k), None)
            if cur is not None:
                cur.corrections.append((a.ann_id, cur.ann_ids[-1] if cur.ann_ids else ""))
                cur.updated_at = a.available_at
                continue
            a = Ann(a.ann_id, a.ticker, a.event_type, "single", a.available_at, a.title, a.subject)
        if a.stage == "single" or cur is None or cur.closed:
            n[k] = n.get(k, 0) + 1
            sid = f"{a.ticker}:{a.event_type}" + (f":{a.subject}" if a.subject else "") + f":{n[k]}"
            cur = Chain(sid, a.ticker, a.event_type, a.stage if a.stage != "single" else "single", a.available_at,
                        a.available_at, a.subject, [a.ann_id])
            chains.append(cur)
            if a.stage == "single" or a.stage in TERMINAL:
                open_.pop(k, None)
            else:
                open_[k] = cur
            continue
        if a.stage in ORDER and cur.state in ORDER and ORDER[a.stage] < ORDER[cur.state]:
            cur.anomalies.append(f"stage went back {cur.state} -> {a.stage} at {a.ann_id}")
        cur.state = a.stage
        cur.updated_at = a.available_at
        cur.ann_ids.append(a.ann_id)
        if cur.closed:
            open_.pop(k, None)
    return chains


def state_as_of(anns: list[Ann], as_of: str) -> list[Chain]:
    """as_of 时点（含）之前已发布的公告构成的链；未关闭的链就是"进行中的事件"。"""
    return build_chains([a for a in anns if a.available_at <= as_of])
