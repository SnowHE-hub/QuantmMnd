// 开发服务器上的"受控命令执行台"（POST /api/live_run）。
// 原实现把浏览器传来的整段 shell 命令交给 `wsl bash -c` 执行，开发服务器又监听所有网卡（host: true），
// 同一网络里任何人都能在 WSL 里执行任意命令；参数（日期、股票代码）也直接拼进 shell 字符串。
// 现在：只接受白名单里的任务编号；参数用正则校验；用 `wsl -e` 以参数数组启动（不经过 shell）；
// 只接受本机来源的请求；客户端断开或超时就结束子进程；按 UTF-8 流式解码。
import { spawn, ChildProcess } from 'child_process';
import type { IncomingMessage, ServerResponse } from 'http';

const PY = '/home/lenovo/miniforge3/envs/qm-gpu/bin/python';
const REPO = '/home/lenovo/projects/qm-t035';
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const TICKER = /^\d{6}\.(SH|SZ|BJ)$/;
const TIMEOUT_MS = 120_000;

type Args = { date?: string; ticker?: string };
type Task = { argv: (a: Required<Args>) => string[]; needs: (keyof Args)[] };

export const TASKS: Record<string, Task> = {
  // 当天盘后流水线的日志尾部
  pipeline_status: {
    needs: ['date'],
    argv: ({ date }) => ['tail', '-n', '40', `/home/lenovo/t030_scratch/logs/daily/${date.replace(/-/g, '')}/day_${date}.log`],
  },
  health_audit: {
    needs: [],
    argv: () => [
      'env', `PYTHONPATH=${REPO}`, PY, '-c',
      "from quantmind.line_a.api_server import app; from quantmind.line_a import api_analytics as A; " +
        "m = A.market(); print('api routes:', len(app.routes)); print('price panel last bar:', m['dates'][-1].date())",
    ],
  },
  // 单股时点数据工具自检：真的取一次财报、预告与快报，打印每张表的条数
  research_ticker: {
    needs: ['date', 'ticker'],
    argv: ({ date, ticker }) => [
      'env', `PYTHONPATH=${REPO}`, PY, '-c',
      'import sys; from pathlib import Path; from quantmind.line_a.harness.tools_data import DataTools; ' +
        "t, d = sys.argv[1], sys.argv[2]; dt = DataTools(Path('/home/lenovo/projects/quantmind/data/lake'), d); " +
        "s = dt.financial_snapshot(t); print('financial tables:', {k: len(v) for k, v in s['tables'].items()}); " +
        "f = dt.forecasts_and_express(t); print('forecast/express:', {k: len(v) for k, v in f.items() if isinstance(v, list)})",
      ticker, date.replace(/-/g, ''),
    ],
  },
  catalog_scan: {
    needs: [],
    argv: () => [
      PY, '-c',
      "import json; d = json.load(open('/home/lenovo/projects/quantmind/data/line_a_v2/dsh_out/t55/data_catalog.json')); " +
        "t = d['totals']; print('files:', t['files'], 'bytes:', t['bytes'], '(%.1f GB)' % (t['bytes'] / 1e9)); print('generated_at:', d['generated_at'])",
    ],
  },
};

const isLocal = (req: IncomingMessage) => {
  const addr = req.socket.remoteAddress || '';
  const origin = String(req.headers.origin || '');
  const okAddr = addr === '127.0.0.1' || addr === '::1' || addr === '::ffff:127.0.0.1';
  const okOrigin = !origin || /^http:\/\/(localhost|127\.0\.0\.1):\d+$/.test(origin);
  return okAddr && okOrigin;
};

export function liveRunnerPlugin() {
  return {
    name: 'live-runner-plugin',
    configureServer(server: { middlewares: { use: (p: string, h: (req: IncomingMessage, res: ServerResponse, next: () => void) => void) => void } }) {
      server.middlewares.use('/api/live_run', (req, res, next) => {
        if (req.method !== 'POST') return next();
        const reject = (code: number, msg: string) => {
          res.statusCode = code;
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(JSON.stringify({ error: msg }));
        };
        if (!isLocal(req)) return reject(403, 'only local requests');
        let body = '';
        req.setEncoding('utf8');
        req.on('data', (c: string) => {
          body += c;
          if (body.length > 4096) req.destroy();
        });
        req.on('end', () => {
          let parsed: { task?: string } & Args;
          try {
            parsed = JSON.parse(body || '{}');
          } catch {
            return reject(400, 'bad json');
          }
          const task = TASKS[parsed.task || ''];
          if (!task) return reject(400, `unknown task (allowed: ${Object.keys(TASKS).join(', ')})`);
          const args: Required<Args> = { date: parsed.date || '', ticker: parsed.ticker || '' };
          if (task.needs.includes('date') && !DATE.test(args.date)) return reject(400, 'date must be YYYY-MM-DD');
          if (task.needs.includes('ticker') && !TICKER.test(args.ticker)) return reject(400, 'ticker must look like 600519.SH');

          const argv = task.argv(args);
          res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-cache' });
          res.write(`[LiveRunner] ${parsed.task}: ${argv.map((a) => (/\s/.test(a) ? JSON.stringify(a) : a)).join(' ')}\n\n`);
          let child: ChildProcess;
          try {
            child = spawn('wsl.exe', ['-d', 'Ubuntu', '-e', ...argv], { shell: false, windowsHide: true });
          } catch (e) {
            res.end(`[LiveRunner] spawn failed: ${(e as Error).message}\n[exit:-1]\n`);
            return;
          }
          const timer = setTimeout(() => {
            res.write(`\n[LiveRunner] 超过 ${TIMEOUT_MS / 1000} 秒，已终止\n`);
            child.kill();
          }, TIMEOUT_MS);
          child.stdout?.setEncoding('utf8');
          child.stderr?.setEncoding('utf8');
          child.stdout?.on('data', (d: string) => res.write(d));
          child.stderr?.on('data', (d: string) => res.write(d));
          child.on('error', (err) => res.write(`\n[LiveRunner] 进程错误: ${err.message}\n`));
          child.on('close', (code) => {
            clearTimeout(timer);
            res.end(`\n[exit:${code ?? -1}]\n`);
          });
          res.on('close', () => {
            // 浏览器关掉或中断：不留孤儿进程
            if (child.exitCode === null) child.kill();
          });
        });
      });
    },
  };
}
