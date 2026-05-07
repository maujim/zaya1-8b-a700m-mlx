import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";

// Absolutely throw-away local dashboard for the four perf worker clones.
// It does NOT attach to worker stdout. Workers write worker.log; this is a
// tiny tail -f-ish UI that rereads those files every couple seconds.

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

type State = "queued" | "running" | "done" | "failed" | "unknown";

type TaskView = {
  id: string;
  title: string;
  branch: string;
  repo: string;
  log: string;
  pidFile: string;
  pid?: number;
  state: State;
  git: string;
  age: string;
  summary: string;
  tail: string[];
};

function safeRead(file: string): string | undefined {
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

function tail(file: string, maxLines: number): string[] {
  try {
    if (!fs.existsSync(file)) return ["waiting for worker.log…"];
    const text = fs.readFileSync(file, "utf8");
    const lines = text.split(/\r?\n/).filter(Boolean);
    if (lines.length === 0) return ["worker.log exists but is empty…"];
    return lines.slice(-maxLines).map((line) => truncate(prettyLogLine(line), 120));
  } catch (e: any) {
    return [`tail failed: ${e?.message ?? e}`];
  }
}

function prettyLogLine(line: string): string {
  // pi --mode json often writes JSON. Pull out useful bits if obvious, but keep
  // raw-ish fallback so this survives format changes.
  try {
    const obj = JSON.parse(line);
    const text = obj.text ?? obj.message ?? obj.content ?? obj.delta ?? obj.type;
    if (typeof text === "string") return text;
    if (obj.type) return `${obj.type} ${JSON.stringify(obj).slice(0, 300)}`;
  } catch {
    // not JSON, fine
  }
  return line.replace(/\\n/g, " ");
}

function pidState(pidFile: string, logFile: string): { state: State; pid?: number } {
  const pidText = safeRead(pidFile)?.trim();
  if (!pidText) {
    if (fs.existsSync(logFile)) return { state: "unknown" };
    return { state: "queued" };
  }
  const pid = Number(pidText);
  if (!pid) return { state: "unknown" };
  try {
    process.kill(pid, 0);
    return { state: "running", pid };
  } catch {
    const log = safeRead(logFile) ?? "";
    if (/Traceback|Error:|failed|FAIL|Command exited with code [1-9]/i.test(log)) return { state: "failed", pid };
    return { state: "done", pid };
  }
}

function gitLine(repo: string): string {
  try {
    const head = safeRead(path.join(repo, ".git", "HEAD"))?.trim();
    let branch = "detached";
    if (head?.startsWith("ref:")) branch = head.replace("ref: refs/heads/", "");
    else if (head) branch = head.slice(0, 8);
    const hasIndex = fs.existsSync(path.join(repo, ".git"));
    return hasIndex ? branch : "not cloned";
  } catch {
    return "not cloned";
  }
}

function summarize(lines: string[], state: State): string {
  const joined = lines.join(" ");
  const interesting = [...lines].reverse().find((l) =>
    /commit|committed|py_compile|pytest|validation|error|failed|created pull request|https:\/\/github.com|diff|modified|done/i.test(l),
  );
  if (interesting) return truncate(interesting, 96);
  if (state === "queued") return "waiting to launch";
  if (state === "running") return truncate(lines[lines.length - 1] ?? "working…", 96);
  if (state === "done") return "finished; inspect branch/PR";
  if (state === "failed") return truncate(joined || "worker failed", 96);
  return truncate(lines[lines.length - 1] ?? "no activity yet", 96);
}

function taskView(cwd: string, task: (typeof TASKS)[number]): TaskView {
  const root = path.join(cwd, "worktrees");
  const repo = path.join(root, task.id);
  // Preferred manual-clone layout. Fallbacks are only for my earlier false start.
  const log = fs.existsSync(path.join(repo, "worker.log"))
    ? path.join(repo, "worker.log")
    : path.join(root, "logs", `${task.id}.log`);
  const pidFile = fs.existsSync(path.join(repo, "worker.pid"))
    ? path.join(repo, "worker.pid")
    : path.join(root, "logs", `${task.id}.pid`);
  const ps = pidState(pidFile, log);
  const t = tail(log, expanded ? linesPerTask : 1);
  return {
    id: task.id,
    title: task.title,
    branch: task.branch,
    repo,
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
  if (state === "running") return "◉";
  if (state === "done") return "✓";
  if (state === "failed") return "✗";
  if (state === "queued") return "○";
  return "?";
}

function stateText(v: TaskView): string {
  if (v.state === "running") return `Running in background${v.pid ? ` (PID: ${v.pid})` : ""}`;
  if (v.state === "done") return `Done${v.pid ? ` (PID: ${v.pid})` : ""}`;
  if (v.state === "failed") return `Failed${v.pid ? ` (PID: ${v.pid})` : ""}`;
  if (v.state === "queued") return "Queued / not launched";
  return "Unknown state";
}

function buildLines(cwd: string): string[] {
  const views = TASKS.map((t) => taskView(cwd, t));
  const running = views.filter((v) => v.state === "running").length;
  const done = views.filter((v) => v.state === "done").length;
  const failed = views.filter((v) => v.state === "failed").length;

  const out: string[] = [];
  out.push(`▸ ZAYA perf agents  ${running} running · ${done} done · ${failed} failed   ${new Date().toLocaleTimeString()}`);
  out.push(`  watching ./worktrees/*/worker.log (tail -f style)   /perf-workers-expand · /perf-workers-off`);
  out.push("");

  for (const v of views) {
    // Screenshot-ish card header: pale block in real TUI? We only have text lines,
    // so use fat unicode borders and indentation.
    out.push(`▸ ${icon(v.state)} ${v.title}   ${v.branch}`);
    out.push(`  └ ${stateText(v)} · git:${v.git} · log:${v.age}`);
    out.push(`    ${v.summary}`);
    if (expanded) {
      out.push(`    tail ${path.relative(cwd, v.log)}:`);
      for (const line of v.tail) out.push(`      ${line}`);
    }
    out.push("");
  }

  return out.slice(0, 90);
}

function refresh(ctx: ExtensionContext) {
  lastCtx = ctx;
  const lines = buildLines(ctx.cwd);
  ctx.ui.setWidget(WIDGET, lines, { placement: "belowEditor" });

  const views = TASKS.map((t) => taskView(ctx.cwd, t));
  const running = views.filter((v) => v.state === "running").length;
  const failed = views.filter((v) => v.state === "failed").length;
  ctx.ui.setStatus(WIDGET, failed ? `perf agents ${running} running, ${failed} failed` : `perf agents ${running} running`);
}

function start(ctx: ExtensionContext) {
  lastCtx = ctx;
  if (timer) clearInterval(timer);
  refresh(ctx);
  timer = setInterval(() => {
    if (lastCtx) refresh(lastCtx);
  }, 1500);
}

function stop(ctx: ExtensionContext) {
  if (timer) clearInterval(timer);
  timer = undefined;
  ctx.ui.setWidget(WIDGET, undefined, { placement: "belowEditor" });
  ctx.ui.setStatus(WIDGET, undefined as any);
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    if (fs.existsSync(path.join(ctx.cwd, "worktrees"))) start(ctx);
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    stop(ctx);
  });

  pi.registerCommand("perf-workers", {
    description: "Show/refresh the ZAYA perf worker dashboard",
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
      if (Number.isFinite(n) && n > 0) linesPerTask = Math.min(20, Math.floor(n));
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
