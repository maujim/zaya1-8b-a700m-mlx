import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";

// Cheap throw-away one-screen dashboard for perf worker clones.
// Reads only tail chunks from ./worktrees/*/worker.log.

const TASKS = [
  { id: "moe-single-token", title: "MoE" },
  { id: "reuse-mask-rope", title: "RoPE" },
  { id: "cache-skeleton", title: "CacheSkel" },
  { id: "prefill-decode-loop", title: "PrefillDecode" },
];

const WIDGET = "zaya-perf-workers";
let timer: NodeJS.Timeout | undefined;
let lastCtx: ExtensionContext | undefined;
let expandedTask: string | undefined;
let intervalMs = 8000;

type State = "queued" | "running" | "done" | "failed" | "unknown";

type TaskView = {
  id: string;
  title: string;
  log: string;
  pid?: number;
  state: State;
  git: string;
  age: string;
  summary: string;
  tail: string[];
};

function readSmall(file: string, maxBytes = 4096): string | undefined {
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

function readTiny(file: string): string | undefined {
  try { return fs.readFileSync(file, "utf8"); } catch { return undefined; }
}

function trunc(s: string, n: number): string {
  const one = s.replace(/\s+/g, " ").trim();
  return one.length > n ? one.slice(0, n - 1) + "…" : one;
}

function age(file: string): string {
  try {
    const ms = Date.now() - fs.statSync(file).mtimeMs;
    if (ms < 60_000) return `${Math.max(0, Math.floor(ms / 1000))}s`;
    if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
    return `${Math.floor(ms / 3_600_000)}h`;
  } catch { return "-"; }
}

function pretty(line: string): string {
  try {
    const obj = JSON.parse(line);
    const text = obj.text ?? obj.message ?? obj.content ?? obj.delta ?? obj.type;
    if (typeof text === "string") return text;
    if (obj.type) return obj.type;
  } catch {}
  return line.replace(/\\n/g, " ");
}

function tail(file: string, n: number): string[] {
  if (!fs.existsSync(file)) return ["waiting for log"];
  const lines = (readSmall(file, 12 * 1024) ?? "").split(/\r?\n/).filter(Boolean);
  if (!lines.length) return ["empty log"];
  return lines.slice(-n).map((l) => trunc(pretty(l), 92));
}

function pidState(pidFile: string, log: string): { state: State; pid?: number } {
  const pid = Number(readTiny(pidFile)?.trim());
  if (!pid) return fs.existsSync(log) ? { state: "unknown" } : { state: "queued" };
  try {
    process.kill(pid, 0);
    return { state: "running", pid };
  } catch {
    const t = readSmall(log, 8192) ?? "";
    return /Traceback|Error:|failed|FAIL|Command exited with code [1-9]/i.test(t)
      ? { state: "failed", pid }
      : { state: "done", pid };
  }
}

function gitLine(repo: string): string {
  const head = readTiny(path.join(repo, ".git", "HEAD"))?.trim();
  if (!head) return "not-cloned";
  if (head.startsWith("ref:")) return head.replace("ref: refs/heads/", "").replace("perf/", "");
  return head.slice(0, 7);
}

function view(cwd: string, task: (typeof TASKS)[number]): TaskView {
  const repo = path.join(cwd, "worktrees", task.id);
  const log = path.join(repo, "worker.log");
  const ps = pidState(path.join(repo, "worker.pid"), log);
  const t = tail(log, expandedTask === task.id ? 5 : 1);
  const interesting = [...t].reverse().find((l) => /commit|py_compile|validation|error|failed|pull request|github.com|done/i.test(l));
  return {
    id: task.id,
    title: task.title,
    log,
    pid: ps.pid,
    state: ps.state,
    git: gitLine(repo),
    age: age(log),
    summary: trunc(interesting ?? t[t.length - 1] ?? "waiting", 72),
    tail: t,
  };
}

function icon(s: State): string {
  return s === "running" ? "◉" : s === "done" ? "✓" : s === "failed" ? "✗" : s === "queued" ? "○" : "?";
}

function build(cwd: string, views: TaskView[]): string[] {
  const running = views.filter((v) => v.state === "running").length;
  const done = views.filter((v) => v.state === "done").length;
  const failed = views.filter((v) => v.state === "failed").length;
  const out: string[] = [];
  out.push(`▸ ZAYA perf agents  ${running} running · ${done} done · ${failed} failed · ${new Date().toLocaleTimeString()}`);
  out.push(`  /perf-workers-expand <task>  /perf-workers-off`);
  for (const v of views) {
    out.push(`${icon(v.state)} ${v.title.padEnd(13)} ${v.git.padEnd(18)} ${v.age.padStart(4)}  ${v.summary}`);
  }
  if (expandedTask) {
    const v = views.find((x) => x.id === expandedTask);
    if (v) {
      out.push("");
      out.push(`tail ${v.id}/worker.log:`);
      for (const line of v.tail) out.push(`  ${line}`);
    }
  }
  return out.slice(0, 14);
}

function refresh(ctx: ExtensionContext) {
  lastCtx = ctx;
  const views = TASKS.map((t) => view(ctx.cwd, t));
  ctx.ui.setWidget(WIDGET, build(ctx.cwd, views));
  const running = views.filter((v) => v.state === "running").length;
  const failed = views.filter((v) => v.state === "failed").length;
  ctx.ui.setStatus(WIDGET, failed ? `perf ${running} running ${failed} failed` : `perf ${running} running`);
}

function start(ctx: ExtensionContext) {
  if (timer) clearInterval(timer);
  refresh(ctx);
  timer = setInterval(() => lastCtx && refresh(lastCtx), intervalMs);
}

function stop(ctx: ExtensionContext) {
  if (timer) clearInterval(timer);
  timer = undefined;
  ctx.ui.setWidget(WIDGET, undefined);
  ctx.ui.setStatus(WIDGET, undefined as any);
}

export default function (pi: ExtensionAPI) {
  pi.on("session_shutdown", async (_event, ctx) => stop(ctx));

  pi.registerCommand("perf-workers", {
    description: "Show compact ZAYA perf worker dashboard",
    handler: async (_args, ctx) => {
      expandedTask = undefined;
      start(ctx);
      ctx.ui.notify("Showing compact perf worker dashboard.", "info");
    },
  });

  pi.registerCommand("perf-workers-expand", {
    description: "Expand one task tail: /perf-workers-expand moe-single-token",
    handler: async (args, ctx) => {
      const requested = (args || "").trim();
      expandedTask = TASKS.find((t) => t.id === requested || t.title.toLowerCase() === requested.toLowerCase())?.id ?? TASKS[0].id;
      start(ctx);
      ctx.ui.notify(`Expanded ${expandedTask}.`, "info");
    },
  });

  pi.registerCommand("perf-workers-off", {
    description: "Hide ZAYA perf worker dashboard",
    handler: async (_args, ctx) => {
      stop(ctx);
      ctx.ui.notify("Hidden perf worker dashboard.", "info");
    },
  });
}
