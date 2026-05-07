import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";

// Throw-away local dashboard for perf worker clones.
// IMPORTANT: keep this cheap. Old version read whole logs every ~1.5s and made
// pi sad. This version reads only the last few KB and refreshes slowly.

const TASKS = [
  { id: "moe-single-token", title: "MoE single-token", branch: "perf/moe-single-token" },
  { id: "reuse-mask-rope", title: "RoPE/mask reuse", branch: "perf/reuse-mask-rope" },
  { id: "cache-skeleton", title: "KV/CCA cache skeleton", branch: "perf/cache-skeleton" },
  { id: "prefill-decode-loop", title: "Cached prefill/decode", branch: "perf/prefill-decode-loop" },
];

const WIDGET = "zaya-perf-workers";
let timer: NodeJS.Timeout | undefined;
let lastCtx: ExtensionContext | undefined;
let expanded = false;
let linesPerTask = 3;
let intervalMs = 6000;

type State = "queued" | "running" | "done" | "failed" | "unknown";

type TaskView = {
  id: string;
  title: string;
  branch: string;
  log: string;
  pidFile: string;
  pid?: number;
  state: State;
  git: string;
  age: string;
  summary: string;
  tail: string[];
};

function safeReadSmall(file: string, maxBytes = 4096): string | undefined {
  try {
    const st = fs.statSync(file);
    const len = Math.min(st.size, maxBytes);
    const fd = fs.openSync(file, "r");
    try {
      const buf = Buffer.alloc(len);
      fs.readSync(fd, buf, 0, len, Math.max(0, st.size - len));
      return buf.toString("utf8");
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return undefined;
  }
}

function safeReadTiny(file: string): string | undefined {
  try {
    return fs.readFileSync(file, "utf8");
  } catch {
    return undefined;
  }
}

function truncate(s: string, n: number): string {
  const one = s.replace(/\s+/g, " ").trim();
  return one.length > n ? one.slice(0, n - 1) + "…" : one;
}

function fileAge(file: string): string {
  try {
    const ms = Date.now() - fs.statSync(file).mtimeMs;
    if (ms < 5_000) return "now";
    if (ms < 60_000) return `${Math.floor(ms / 1000)}s ago`;
    if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
    return `${Math.floor(ms / 3_600_000)}h ago`;
  } catch {
    return "never";
  }
}

function prettyLogLine(line: string): string {
  // pi --mode json writes JSON lines; extract the useful text when obvious.
  try {
    const obj = JSON.parse(line);
    const text = obj.text ?? obj.message ?? obj.content ?? obj.delta ?? obj.type;
    if (typeof text === "string") return text;
    if (obj.type) return `${obj.type} ${JSON.stringify(obj).slice(0, 180)}`;
  } catch {
    // raw log line
  }
  return line.replace(/\\n/g, " ");
}

function tail(file: string, maxLines: number): string[] {
  if (!fs.existsSync(file)) return ["waiting for worker.log…"];
  const text = safeReadSmall(file, expanded ? 32 * 1024 : 4 * 1024) ?? "";
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return ["worker.log exists but is empty…"];
  return lines.slice(-maxLines).map((line) => truncate(prettyLogLine(line), 110));
}

function pidState(pidFile: string, logFile: string): { state: State; pid?: number } {
  const pidText = safeReadTiny(pidFile)?.trim();
  if (!pidText) return fs.existsSync(logFile) ? { state: "unknown" } : { state: "queued" };
  const pid = Number(pidText);
  if (!pid) return { state: "unknown" };
  try {
    process.kill(pid, 0);
    return { state: "running", pid };
  } catch {
    const logTail = safeReadSmall(logFile, 16 * 1024) ?? "";
    if (/Traceback|Error:|failed|FAIL|Command exited with code [1-9]/i.test(logTail)) return { state: "failed", pid };
    return { state: "done", pid };
  }
}

function gitLine(repo: string): string {
  const head = safeReadTiny(path.join(repo, ".git", "HEAD"))?.trim();
  if (!head) return "not cloned";
  if (head.startsWith("ref:")) return head.replace("ref: refs/heads/", "");
  return head.slice(0, 8);
}

function summarize(lines: string[], state: State): string {
  const interesting = [...lines].reverse().find((l) => /commit|py_compile|validation|error|failed|pull request|github.com|done/i.test(l));
  if (interesting) return truncate(interesting, 90);
  if (state === "queued") return "waiting to launch";
  if (state === "running") return truncate(lines[lines.length - 1] ?? "working…", 90);
  if (state === "done") return "finished; inspect branch/PR";
  if (state === "failed") return truncate(lines.join(" ") || "worker failed", 90);
  return truncate(lines[lines.length - 1] ?? "no activity yet", 90);
}

function taskView(cwd: string, task: (typeof TASKS)[number]): TaskView {
  const root = path.join(cwd, "worktrees");
  const repo = path.join(root, task.id);
  const log = path.join(repo, "worker.log");
  const pidFile = path.join(repo, "worker.pid");
  const ps = pidState(pidFile, log);
  const t = tail(log, expanded ? linesPerTask : 1);
  return {
    id: task.id,
    title: task.title,
    branch: task.branch,
    log,
    pidFile,
    pid: ps.pid,
    state: ps.state,
    git: gitLine(repo),
    age: fileAge(log),
    summary: summarize(t, ps.state),
    tail: t,
  };
}

function icon(state: State): string {
  return state === "running" ? "◉" : state === "done" ? "✓" : state === "failed" ? "✗" : state === "queued" ? "○" : "?";
}

function stateText(v: TaskView): string {
  if (v.state === "running") return `Running${v.pid ? ` (PID: ${v.pid})` : ""}`;
  if (v.state === "done") return `Done${v.pid ? ` (PID: ${v.pid})` : ""}`;
  if (v.state === "failed") return `Failed${v.pid ? ` (PID: ${v.pid})` : ""}`;
  if (v.state === "queued") return "Queued / not launched";
  return "Unknown";
}

function buildLines(cwd: string, views: TaskView[]): string[] {
  const running = views.filter((v) => v.state === "running").length;
  const done = views.filter((v) => v.state === "done").length;
  const failed = views.filter((v) => v.state === "failed").length;
  const out: string[] = [];
  out.push(`▸ ZAYA perf agents  ${running} running · ${done} done · ${failed} failed   ${new Date().toLocaleTimeString()}`);
  out.push(`  tailing ./worktrees/*/worker.log every ${Math.round(intervalMs / 1000)}s   /perf-workers-expand · /perf-workers-off`);
  out.push("");
  for (const v of views) {
    out.push(`▸ ${icon(v.state)} ${v.title}   ${v.branch}`);
    out.push(`  └ ${stateText(v)} · git:${v.git} · log:${v.age}`);
    out.push(`    ${v.summary}`);
    if (expanded) {
      out.push(`    tail ${path.relative(cwd, v.log)}:`);
      for (const line of v.tail) out.push(`      ${line}`);
    }
    out.push("");
  }
  return out.slice(0, 70);
}

function refresh(ctx: ExtensionContext) {
  lastCtx = ctx;
  const views = TASKS.map((t) => taskView(ctx.cwd, t));
  ctx.ui.setWidget(WIDGET, buildLines(ctx.cwd, views), { placement: "belowEditor" });
  const running = views.filter((v) => v.state === "running").length;
  const failed = views.filter((v) => v.state === "failed").length;
  ctx.ui.setStatus(WIDGET, failed ? `perf agents ${running} running, ${failed} failed` : `perf agents ${running} running`);
}

function start(ctx: ExtensionContext) {
  lastCtx = ctx;
  if (timer) clearInterval(timer);
  refresh(ctx);
  timer = setInterval(() => lastCtx && refresh(lastCtx), intervalMs);
}

function stop(ctx: ExtensionContext) {
  if (timer) clearInterval(timer);
  timer = undefined;
  ctx.ui.setWidget(WIDGET, undefined, { placement: "belowEditor" });
  ctx.ui.setStatus(WIDGET, undefined as any);
}

export default function (pi: ExtensionAPI) {
  // No auto-start. Manually run /perf-workers so this extension cannot slow down
  // normal pi sessions just because ./worktrees exists.
  pi.on("session_shutdown", async (_event, ctx) => stop(ctx));

  pi.registerCommand("perf-workers", {
    description: "Show/refresh the cheap ZAYA perf worker dashboard",
    handler: async (_args, ctx) => {
      expanded = false;
      start(ctx);
      ctx.ui.notify("Showing compact ZAYA perf worker dashboard.", "info");
    },
  });

  pi.registerCommand("perf-workers-expand", {
    description: "Show expanded tail-f style worker logs in the dashboard",
    handler: async (args, ctx) => {
      const n = Number((args || "").trim());
      if (Number.isFinite(n) && n > 0) linesPerTask = Math.min(12, Math.floor(n));
      expanded = true;
      start(ctx);
      ctx.ui.notify(`Showing expanded worker log tails (${linesPerTask} lines/task).`, "info");
    },
  });

  pi.registerCommand("perf-workers-off", {
    description: "Hide the ZAYA perf worker dashboard",
    handler: async (_args, ctx) => {
      stop(ctx);
      ctx.ui.notify("Hidden ZAYA perf worker dashboard.", "info");
    },
  });
}
