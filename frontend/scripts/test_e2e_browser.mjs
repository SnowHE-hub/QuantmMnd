// 端到端巡检（Edge 无头浏览器 + CDP）：每一项都和后端接口的真实数值逐一核对，任何一项不符就失败（退出码 1）。
// 原版本只记录“按钮点到了没有、页面有没有报错”，功能结果从不断言。
// 用法：先启动后端（8780）和前端（5173），再 node scripts/test_e2e_browser.mjs
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const DEBUG_PORT = 9223;
const APP = 'http://localhost:5173';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const api = async (p) => {
  const r = await fetch(`${APP}/api${p}`);
  if (!r.ok) throw new Error(`api ${p} -> ${r.status}`);
  return r.json();
};

const results = [];
const check = (name, ok, detail = '') => {
  results.push({ name, ok: Boolean(ok), detail });
  console.log(`${ok ? '  PASS' : '  FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

async function main() {
  if (!fs.existsSync(EDGE_PATH)) throw new Error(`Edge not found at ${EDGE_PATH}`);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'edge-e2e-'));
  const edge = spawn(
    EDGE_PATH,
    ['--headless=new', `--remote-debugging-port=${DEBUG_PORT}`, '--window-size=1600,1000', '--disable-gpu', '--no-first-run', `--user-data-dir=${profile}`, 'about:blank'],
    { stdio: 'ignore' }
  );
  const cleanup = () => {
    try { edge.kill(); } catch {}
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
  };
  process.on('exit', cleanup);

  let targets = null;
  for (let i = 0; i < 25 && !targets; i++) {
    await sleep(400);
    try {
      const r = await fetch(`http://localhost:${DEBUG_PORT}/json`);
      if (r.ok) targets = await r.json();
    } catch {}
  }
  if (!targets?.length) throw new Error('no CDP target');
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 1;
  const pending = new Map();
  const exceptions = [];
  const consoleErrors = [];
  const badResponses = [];
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) {
      const { res, rej } = pending.get(m.id);
      pending.delete(m.id);
      m.error ? rej(m.error) : res(m.result);
    } else if (m.method === 'Runtime.exceptionThrown') {
      exceptions.push(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text);
    } else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
      consoleErrors.push((m.params.args || []).map((a) => a.value || a.description || '').join(' '));
    } else if (m.method === 'Network.responseReceived') {
      const r = m.params.response;
      if (r.url.includes('/api/') && r.status >= 500) badResponses.push(`${r.status} ${r.url}`);
    }
  };
  const send = (method, params = {}) => new Promise((res, rej) => { const i = id++; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expr) => (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result?.value;
  const waitFor = async (expr, ms = 20000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (await ev(`Boolean(${expr})`)) return true;
      await sleep(300);
    }
    return false;
  };
  const clickText = (txt) => ev(`(() => { const b = [...document.querySelectorAll('button, [role=button], div, span')].find(x => x.children.length < 4 && x.innerText && x.innerText.trim().startsWith(${JSON.stringify(txt)})); if (b) { b.click(); return true; } return false; })()`);
  const text = (sel) => ev(`document.querySelector(${JSON.stringify(sel)})?.innerText ?? null`);

  await send('Page.enable');
  await send('Runtime.enable');
  await send('Network.enable');

  // ---------------------------------------------------------------- 后端真实值
  const health = await api('/health');
  const d = health.latest_run;
  const dates = await api('/dates');
  const formal = await api(`/formal/${d}`);
  const funnel = await api(`/funnel/${d}`);
  const ic = await api('/forward_ic?horizon=5');
  const rtV2Hist = await api(`/redteam/${dates.runs[0]}?source=f4&horizon=5&version=v2`);
  console.log(`latest_run = ${d}; runs = ${dates.runs.length}`);

  await send('Page.navigate', { url: APP + '/' });
  check('1. 页面挂载并选中最新运行日', await waitFor(`document.body.innerText.includes(${JSON.stringify(d)})`));

  // 2. 委托表行数 = 正式日报 10 万档委托数
  await waitFor(`document.querySelector('[data-testid="orders-next-open"] tbody')?.children.length === ${formal.tiers.aum_1e5.orders_next_open.length}`);
  const nOrders = await ev(`document.querySelector('[data-testid="orders-next-open"] tbody').children.length`);
  const want = formal.tiers.aum_1e5.orders_next_open.length;
  check('2. 委托表行数 = 正式日报 10 万档委托数', nOrders === want, `${nOrders} vs ${want}`);

  // 3. 事后 IC 摘要 = /api/forward_ic
  await waitFor(`document.querySelector('[data-testid="ic-summary"]')`);
  const icText = await text('[data-testid="ic-summary"]');
  check('3. 事后 IC 摘要与接口一致', icText?.includes(ic.mean_ic.toFixed(3)) && icText.includes(`已到期 ${ic.n_dates} 个`), icText);

  // 4. 推演沙盒：科创板加一档，股数合法、现金按成交额 + 费用减少
  const rules = await api('/rules/execution');
  await clickText('100 万元档');
  await sleep(600);
  await clickText('开启反事实推演沙盒');
  await waitFor(`document.querySelector('[data-testid="sandbox-panel"]')`);
  const sb = await ev(`(async () => {
    const btn = [...document.querySelectorAll('[data-testid^="step-up-"]')].find(b => /step-up-68[89]/.test(b.dataset.testid)) || document.querySelector('[data-testid^="step-up-"]');
    const t = btn.dataset.testid.replace('step-up-', '');
    const num = (s) => Number(String(s).replace(/[^0-9.-]/g, ''));
    const sharesOf = () => num(document.querySelector('[data-testid="shares-' + t + '"]').childNodes[0].textContent);
    const cash = () => num(document.querySelector('[data-testid="sandbox-cash-after"]').innerText);
    const s0 = sharesOf(), c0 = cash();
    btn.click();
    await new Promise(r => setTimeout(r, 300));
    const row = document.querySelector('[data-testid="shares-' + t + '"]').closest('tr');
    const price = num(row.children[5].innerText);   // 0 推演 1 方向 2 代码 3 板块 4 股数 5 参考价
    return { t, s0, s1: sharesOf(), c0, c1: cash(), price };
  })()`);
  const lot = sb.t.startsWith('688') || sb.t.startsWith('689') ? rules.lot.STAR : rules.lot.MAIN;
  const legal = sb.s1 >= lot.min && (sb.s1 - lot.min) % lot.step === 0;
  const dShares = sb.s1 - sb.s0;
  const gross = dShares * sb.price;
  check('4a. 沙盒加一档后股数符合最小申报单位', legal && dShares > 0, JSON.stringify(sb));
  check('4b. 沙盒现金减少 ≈ 新增成交额（误差 < 成交额 0.2% + 1 元，含佣金与过户费）', sb.c0 - sb.c1 >= gross - 1 && sb.c0 - sb.c1 <= gross * 1.002 + 1, `${sb.c0 - sb.c1} vs ${gross}`);

  // 5. 拓扑页：漏斗数量 = /api/funnel
  await clickText('全景沙盘与拓扑');
  await waitFor(`document.querySelector('[data-testid="funnel-F1"]')`);
  const f1 = await text('[data-testid="funnel-F1"]');
  const f7 = await text('[data-testid="funnel-F7"]');
  const by = Object.fromEntries(funnel.layers.map((l) => [l.layer, l.n]));
  check('5. 漏斗 F1 / F7 数量与接口一致', Number(f1?.replace(/,/g, '')) === by.F1 && (!by.F7 || Number(f7?.replace(/,/g, '')) === by.F7), `${f1} / ${f7}`);

  // 6. 个股页：声明核验溯源的证据编号和结论来自 /api/claim
  await clickText('个股深度透视');
  await waitFor(`[...document.querySelectorAll('span,div')].some(x => /^A1:c\\d+$/.test(x.innerText?.trim()))`);
  const ref = await ev(`(() => { const el = [...document.querySelectorAll('span,div')].find(x => /^A1:c\\d+$/.test(x.innerText?.trim())); el.click(); return el.innerText.trim(); })()`);
  await waitFor(`document.querySelector('[data-testid="claim-trace"]')?.innerText.includes('运行文件')`);
  const traceText = await text('[data-testid="claim-trace"]');
  const ticker = await ev(`(document.body.innerText.match(/\\d{6}\\.(SH|SZ|BJ)/) || [])[0]`);
  let traceApi = null;
  try { traceApi = await api(`/claim/${d}/${ticker}/${ref}`); } catch (e) { traceApi = null; }
  const eid = traceApi?.cited?.[0]?.ref?.evidence_id;
  check('6. 溯源面板显示的证据编号与校验结论来自接口', traceApi && traceText.includes(traceApi.check_now.status) && (!eid || traceText.includes(eid)), `${ticker} ${ref} ${eid ?? '(无直接证据)'}`);
  check('6b. 溯源面板不再出现前端拼的示意代码', !/def verify_|tushare_ast_v2|SELECT title/.test(traceText));

  // 7. 历史日期：红队基准跑输率与接口一致
  await clickText('核心投研工作台');
  await ev(`(() => { const s = document.querySelector('select'); const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(s, ${JSON.stringify(dates.runs[0])}); s.dispatchEvent(new Event('change', { bubbles: true })); return true; })()`);
  await waitFor(`document.querySelector('[data-testid="redteam-v2"]')?.innerText.includes('%') && !document.querySelector('[data-testid="redteam-v2"]').innerText.includes('…')`);
  const v2 = await text('[data-testid="redteam-v2"]');
  const base = rtV2Hist.by_group.excess_pool.base_rate;
  check('7. 历史日红队 v2 的基准跑输率与接口一致', base === null ? v2.includes('—') : v2.includes(`${Math.round(base * 100)}%`), v2);

  // 8. 命令执行台：任意命令被拒绝，白名单任务真实执行并返回退出码
  const lr = await ev(`(async () => {
    const bad = await fetch('/api/live_run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ command: 'echo pwned' }) });
    const inj = await fetch('/api/live_run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ task: 'research_ticker', date: '2026-10-23', ticker: "600519.SH'; rm -rf ~" }) });
    const ok = await fetch('/api/live_run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ task: 'catalog_scan' }) });
    return { bad: bad.status, inj: inj.status, ok: ok.status, body: await ok.text() };
  })()`);
  check('8a. 任意命令与注入参数被拒绝', lr.bad === 400 && lr.inj === 400, `${lr.bad} / ${lr.inj}`);
  check('8b. 白名单任务真实执行并返回退出码 0', lr.ok === 200 && /\[exit:0\]/.test(lr.body) && lr.body.includes('files:'), lr.body.slice(-120).replace(/\n/g, ' '));

  // 10. 模拟账户与交割单：数字与 /api/ledger 一致、会计恒等式在页面上成立、红涨绿跌、日历与曲线联动、切换账本、K 线买卖点
  const acc = 't052_W6_100000';
  const L = await api(`/ledger/${acc}`);
  const lastDay = L.daily[L.daily.length - 1];
  await ev(`document.querySelector('[data-testid="nav-brokerage"]').click(), true`);
  await waitFor(`document.querySelector('[data-testid="hdr-cash"]')?.innerText.includes('¥')`);
  const yuanOf = (s) => Number(String(s).replace(/[^0-9.]/g, '')) * (/[−-]/.test(String(s)) ? -1 : 1);
  const hdr = await ev(`({ eq: document.querySelector('[data-testid="hdr-equity"]').innerText, cash: document.querySelector('[data-testid="hdr-cash"]').innerText,
    mv: document.querySelector('[data-testid="hdr-mv"]').innerText, pnl: document.querySelector('[data-testid="hdr-total-pnl"]').innerText,
    pnlCls: document.querySelector('[data-testid="hdr-total-pnl"]').className, un: document.querySelector('[data-testid="hdr-unrealized"]').innerText,
    unCls: document.querySelector('[data-testid="hdr-unrealized"]').className, checks: document.querySelector('[data-testid="checks"]')?.innerText })`);
  check('10a. 账户总资产与接口一致（精确到分）', Math.abs(yuanOf(hdr.eq) - lastDay.equity) < 0.006, `${hdr.eq} vs ${lastDay.equity.toFixed(2)}`);
  check('10b. 页面上 总资产 = 可用现金 + 持仓市值（误差 ≤ 1 分）', Math.abs(yuanOf(hdr.cash) + yuanOf(hdr.mv) - yuanOf(hdr.eq)) <= 0.011, `${hdr.cash} + ${hdr.mv} = ${hdr.eq}`);
  check('10c. 红涨绿跌：盈利红色、亏损绿色',
    (lastDay.equity > L.meta.aum ? /rose/.test(hdr.pnlCls) : /emerald/.test(hdr.pnlCls)) && (lastDay.unrealized < 0 ? /emerald/.test(hdr.unCls) : /rose/.test(hdr.unCls)),
    `总盈亏 ${hdr.pnl}（${hdr.pnlCls.match(/text-\w+-\d+/)?.[0]}）· 浮盈 ${hdr.un}（${hdr.unCls.match(/text-\w+-\d+/)?.[0]}）`);
  check('10d. 会计核对结果显示为通过', L.meta.checks.passed && /会计核对通过/.test(hdr.checks || ''));

  // 日历点击 → 持仓回放
  const pick = L.daily.filter((r) => r.n_trades > 0).slice(-3)[0];
  const pd = pick.date.slice(0, 10);
  await ev(`(() => { const b = [...document.querySelectorAll('[data-testid="pnl-calendar"] button')].find(x => x.innerText.trim() === ${JSON.stringify(pd.slice(0, 7))}); b && b.click(); return true; })()`);
  await waitFor(`document.querySelector('[data-testid="cal-${pd}"]')`);
  const cellPnl = await ev(`Number(document.querySelector('[data-testid="cal-${pd}"]').dataset.pnl)`);
  await ev(`document.querySelector('[data-testid="cal-${pd}"]').click(), true`);
  await waitFor(`document.querySelector('[data-testid="positions-date"]')?.innerText.startsWith(${JSON.stringify(pd)})`);
  const posApi = await api(`/ledger/${acc}/positions?date=${pd}`);
  const nRows = await ev(`document.querySelectorAll('[data-testid^="hold-"]').length`);
  check('10e. 日历格子的当日盈亏 = 接口 daily_pnl', Math.abs(cellPnl - pick.daily_pnl) < 1e-6, `${pd}: ${cellPnl.toFixed(2)}`);
  check('10f. 点击日历后持仓表回放到那一天（行数与接口一致）', nRows === posApi.positions.length, `${nRows} 行`);
  const mvSum = posApi.positions.reduce((s, p) => s + p.shares * p.close, 0);
  check('10g. 回放日：现金 + Σ 股数 × 收盘价 = 总资产', Math.abs(posApi.day.cash + mvSum - posApi.day.equity) < 0.01, `${posApi.day.cash.toFixed(2)} + ${mvSum.toFixed(2)} = ${posApi.day.equity.toFixed(2)}`);

  // 资金曲线点击某一天 → 持仓日期跟着变（用真实鼠标事件点在画布上）
  const ci = Math.floor(L.daily.length * 0.4);
  const cd = L.daily[ci].date.slice(0, 10);
  const box = await ev(`(() => { const c = document.querySelector('[data-testid="equity-curve"] canvas'); c.scrollIntoView({ block: 'center' }); const r = c.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; })()`);
  await sleep(300);
  const box2 = await ev(`(() => { const r = document.querySelector('[data-testid="equity-curve"] canvas').getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; })()`);
  const px = box2.x + 70 + ((ci + 0.5) * (box2.w - 90)) / L.daily.length;
  const py = box2.y + 30 + box2.h * 0.25;
  for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) {
    await send('Input.dispatchMouseEvent', { type, x: px, y: py, button: 'left', clickCount: 1 });
  }
  const curveOk = await waitFor(`document.querySelector('[data-testid="positions-date"]')?.innerText.startsWith(${JSON.stringify(cd)})`, 5000);
  check('10h. 点击资金曲线上的某一天，持仓表联动回放', curveOk, `${cd} → ${await text('[data-testid="positions-date"]')}`);

  // K 线买卖点
  const shownDate = (await text('[data-testid="positions-date"]')).slice(0, 10);
  const held = (await api(`/ledger/${acc}/positions?date=${shownDate}`)).positions[0]?.ticker;   // 当前显示的那一天的第一只持仓
  if (held) {
    await ev(`document.querySelector('[data-testid="hold-${held}"]')?.click(), true`);
    const sameDay = await waitFor(`document.querySelector('[data-testid="replay-trades"]')`, 8000);
    const tk = await api(`/ledger/${acc}/ticker/${held}`);
    const rtText = await text('[data-testid="replay-trades"]');
    check('10i. K 线复盘列出这只股票的全部 B / S 成交', sameDay && rtText.includes(`成交（${tk.trades.length} 笔）`) && /B 买入/.test(rtText), `${held} ${tk.trades.length} 笔`);
    await ev(`document.querySelector('[data-testid="replay-modal"]').parentElement.click(), true`);
  }

  // 切换账本：影子账本只有权重；正式账户是建仓预览
  const setAcc = (v) => ev(`(() => { const s = document.querySelector('[data-testid="account-select"]'); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, ${JSON.stringify(v)}); s.dispatchEvent(new Event('change', { bubbles: true })); return true; })()`);
  await setAcc('shadow_ens_native_20d__all_boards');
  const shadowOk = await waitFor(`document.querySelector('[data-testid="positions-panel"]')?.innerText.includes('按目标权重记账')`);
  await setAcc('formal_aum_1e5');
  const formalOk = await waitFor(`document.querySelector('[data-testid="formal-preview"]')?.innerText.includes('建仓预览')`);
  check('10j. 切换账本：影子账本显示权重口径、正式账户显示建仓预览', shadowOk && formalOk);

  // 11. 每一页的图表都真正画出来（画布上有非背景像素）。echarts-for-react 3.0.6 在 React 19 下首次挂载是空白，已换成自己的封装
  const painted = `(() => [...document.querySelectorAll('main canvas')].map(c => { try { const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 0; i < d.length; i += 64) if (d[i + 3] > 0) n++; return n; } catch { return -1; } }))()`;
  const pages = [['nav-stock_terminal', '个股 K 线'], ['nav-research_reader', '研报页 · T-052 网格', 'T-052 模拟盘网格成果'], ['nav-research_reader', '研报页 · 影子账本', '前瞻影子账本'],
    ['nav-data_observatory', '数据资产树状图'], ['nav-cockpit', '工作台（事后 IC）']];
  for (const [nav, name, sub] of pages) {
    await ev(`document.querySelector('[data-testid="${nav}"]').click(), true`);
    if (sub) { await sleep(500); await clickText(sub); }
    await waitFor(`document.querySelectorAll('main canvas').length > 0`, 15000);
    await sleep(1500);
    const px = (await ev(painted)) || [];
    check(`11. ${name}：图表已绘制`, px.length > 0 && px.every((n) => n > 50), `${px.length} 张画布，像素 ${px.join('/')}`);
  }

  // 9. 运行时健康：没有未捕获异常、没有 5xx；控制台错误只允许“资源 404”（某些日期没有某类产物）
  const unexpected = consoleErrors.filter((x) => !/Failed to load resource|404/.test(x));
  check('9. 无未捕获异常、无 5xx、无非 404 的控制台错误', exceptions.length === 0 && badResponses.length === 0 && unexpected.length === 0,
    JSON.stringify({ exceptions: exceptions.slice(0, 2), badResponses: badResponses.slice(0, 3), unexpected: unexpected.slice(0, 2) }));

  ws.close();
  cleanup();
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error('E2E fatal:', e);
  process.exit(1);
});
