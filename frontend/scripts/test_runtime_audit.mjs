// 接口契约与数值合理性测试（经 5173 的代理调用 8780）。每项都有断言，任何一项不通过就退出码 1。
// 原版本只打印状态码和字段，从不判定。
// 用法：node scripts/test_runtime_audit.mjs
const API = 'http://localhost:5173/api';
const results = [];
const check = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? '  PASS' : '  FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};
const get = async (p) => {
  const r = await fetch(API + p);
  return { status: r.status, body: r.headers.get('content-type')?.includes('json') ? await r.json() : await r.text() };
};
const finite = (x) => typeof x === 'number' && Number.isFinite(x);

const h = await get('/health');
check('health ok', h.status === 200 && h.body.ok === true && /^\d{4}-\d{2}-\d{2}$/.test(h.body.latest_run), JSON.stringify(h.body));
const d = h.body.latest_run;
const dates = (await get('/dates')).body;
check('dates 有序且包含 latest_run', dates.runs.includes(d) && [...dates.runs].sort().join() === dates.runs.join());

const f = await get(`/formal/${d}`);
check('formal 三档齐全', f.status === 200 && ['aum_1e5', 'aum_1e6', 'aum_1e7'].every((k) => f.body.tiers?.[k]));
for (const [k, t] of Object.entries(f.body.tiers || {})) {
  const bad = t.orders_next_open.filter((o) => !(o.est_shares >= 0) || Math.abs(o.est_shares * o.ref_close - o.est_value) > 0.01);
  check(`formal ${k}: 股数 × 参考价 = 金额，股数 ≥ 0`, bad.length === 0, `${t.orders_next_open.length} 笔`);
  const spend = t.orders_next_open.filter((o) => o.side === 'buy').reduce((s, o) => s + o.est_value, 0);
  check(`formal ${k}: 买入金额不超过现金`, spend <= t.cash + 1e-6, `${spend.toFixed(0)} ≤ ${t.cash}`);
}

const sc = await get(`/scores/${d}?model=ens_20d&top=50`);
check('scores 按 ENS 降序、rank 从 1 开始', sc.body.every((x, i) => x.rank === i + 1 && (i === 0 || sc.body[i - 1].ENS >= x.ENS)));
const one = await get(`/scores/${d}?model=ens_20d&ticker=${sc.body[10].ticker}`);
check('scores?ticker= 返回同一只与正确排名', one.body.length === 1 && one.body[0].rank === 11);

const c = await get(`/consensus/${d}?ticker=${sc.body[0].ticker}`);
const pcts = ['xgb_pct', 'cat_pct', 'tabm_pct', 'seqfusion_pct', 'ENS_pct'].map((k) => c.body.row?.[k]);
check('consensus 百分位在 [0,100]', pcts.every((v) => finite(v) && v >= 0 && v <= 100), pcts.map((v) => v?.toFixed(1)).join(' '));

const runs = dates.runs;
const fw = await get(`/forward/${runs[0]}?source=f3top&top=20&horizons=1,5`);
const okRows = fw.body.rows.filter((r) => r.status === 'filled');
check('forward: T+1 入场日晚于信号日', fw.body.meta.entry_date > runs[0]);
check('forward: 单日收益在 ±35% 内', okRows.every((r) => finite(r.ret_1) && Math.abs(r.ret_1) < 0.35), `${okRows.length} 只成交`);
check('forward: 超额 = 收益 − 研究池等权', okRows.every((r) => Math.abs(r.excess_pool_1 - (r.ret_1 - fw.body.benchmarks['1'].pool_ew)) < 1e-12));
const fwLatest = await get(`/forward/${d}?source=f3top&top=5`);
check('forward: 最新日没有未来收益', fwLatest.body.rows.every((r) => r.ret_1 === null));

const rt = await get(`/redteam/${runs[0]}?source=f4&horizon=5&version=v2`);
const g = rt.body.by_group?.excess_pool;
check('redteam: 给出标记组和未标记组（基准跑输率）', rt.status === 200 && g && 'base_rate' in g && 'lift' in g);
check('redteam: 缺少的规则输入如实列出', Array.isArray(rt.body.missing_inputs));

const fn = await get(`/funnel/${d}`);
const L = Object.fromEntries(fn.body.layers.map((l) => [l.layer, l.n]));
check('funnel: F1 ≥ F2 > 0，F7 = 各状态之和', L.F1 >= L.F2 && L.F2 > 0 &&
  (!L.F7 || fn.body.layers.find((l) => l.layer === 'F7').n === Object.entries(fn.body.layers.find((l) => l.layer === 'F7').detail).filter(([k]) => !['每案例平均', '已核实比例'].includes(k)).reduce((s, [, v]) => s + v, 0)));

const ic = await get('/forward_ic?horizon=5');
check('forward_ic: IC 在 [-1,1]，未到期的为 null', ic.body.rows.every((r) => r.ic === null || (finite(r.ic) && Math.abs(r.ic) <= 1)), `mean ${ic.body.mean_ic?.toFixed(3)}`);

const cases = (await get(`/research/${d}`)).body.cases || [];
const rt0 = cases[0]?.ticker;
const claims = rt0 ? (await get(`/research/${d}/${rt0}`)).body.claims || [] : [];
if (claims.length) {
  const tr = await get(`/claim/${d}/${rt0}/${claims[0].ref}`);
  check('claim: 溯源返回同一条声明与同一结论', tr.status === 200 && tr.body.claim.text === claims[0].text && tr.body.check_at_run.status === claims[0].check);
} else {
  console.log('  SKIP  claim（这只股票当天没有研究）');
}

const bad = await get('/forward/2026-10-23?tickers=$(id)');
check('输入校验：非法股票代码 400', bad.status === 400);
const miss = runs.find((x) => x < '2026-10-14');
if (miss) check('早期运行日缺少的产物返回 404 而不是 500', [404, 200].includes((await get(`/bypass/${miss}`)).status));

// 模拟账户账本
const accs = (await get('/ledger/accounts')).body;
check('ledger: 三类账户都在（T-052 / 正式 / 影子）', ['t052', 'formal', 'shadow'].every((k) => accs.some((a) => a.kind === k)), `${accs.length} 个账户`);
check('ledger: T-052 与正式账户会计核对全部通过', accs.filter((a) => a.kind !== 'shadow').every((a) => a.checks_passed === true));
for (const a of accs.filter((x) => x.kind === 't052')) {
  const L = (await get(`/ledger/${a.account}`)).body;
  const last = L.daily[L.daily.length - 1];
  const ok = L.daily.every((r) => Math.abs(r.cash + r.market_value - r.equity) < 0.01 &&
    Math.abs(r.equity - L.meta.aum - r.realized_cum - r.unrealized - r.dividends_cum) < 0.01);
  check(`ledger ${a.account}: 每天 总资产 = 现金 + 市值 = 本金 + 已实现 + 未实现 + 分红`, ok, `末日 ${last.equity.toFixed(2)}`);
  check(`ledger ${a.account}: 与 T-052 存档净值逐日一致`, L.meta.checks.t052_nav_reproduced_max_abs < 0.01);
}
const tr = (await get('/ledger/t052_W6_100000/trades')).body.trades;
check('ledger: 佣金 ≥ 5 元、印花税只在卖出、过户费 0.001%',
  tr.every((t) => t.commission >= 5 - 1e-9 && Math.abs(t.transfer - t.gross * 1e-5) < 1e-6 && (t.side === 'buy' ? t.stamp === 0 : Math.abs(t.stamp - t.gross * 5e-4) < 1e-6)), `${tr.length} 笔`);
const badAcc = await get('/ledger/..%2Fetc');
check('ledger: 非法账户名被拒绝', [400, 404].includes(badAcc.status));

const passed = results.filter(Boolean).length;
console.log(`\n${passed}/${results.length} passed`);
process.exit(passed === results.length ? 0 : 1);
