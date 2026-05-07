import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";

// Disgusting tiny throw-away dashboard for the parallel pi CLI workers.
// It watches ./worktrees/** logs/pids and splats a tiny tail into a pi widget.

const TASKS = ["moe-single-token", "reuse-mask-rope", "cache-skeleton", "prefill-decode-loop"];
const WIDGET = "zaya-perf-workers";
let timer: NodeJS.Timeout | undefined;
let lastCtx: ExtensionContext | undefined;
let linesPerTask = 5;

function tail(file: string, maxLines: number): string[] {
  try {
    if (!fs.existsSync(file)) return ["(no log yet)"];
    const text = fs.readFileSync(file, "utf8");
    const lines = text.split(/\r?\n/).filter(Boolean);
    return lines.slice(-maxLines).map((line) => line.length > 140 ? line.slice(0, 137) + "..." : line);
  } catch (e: any) {
    return [`(tail failed: ${e?.message ?? e})`];
  }
}

function pidAlive(pidFile: string): string {
  try {
    if (!fs.existsSync(pidFile)) return "no-pid";
    const pid = Number(fs.readFileSync(pidFile, "utf8").trim());
    if (!pid) return "bad-pid";
    try {
      process.kill(pid, 0);
      return `running:${pid}`;
    } catch {
      return `done:${pid}`;
    }
  } catch (e: any) {
    return `pid?:${e?.message ?? e}`;
  }
}

function branchAndHead(repo: string): string {
  try {
    const head = fs.readFileSync(path.join(repo, ".git", "HEAD"), "utf8").trim();
    if (head.startsWith("ref:")) return head.replace("ref: refs/heads/", "");
    return head.slice(0, 8);
  } catch {
    return "no-git";
  }
}

function buildLines(cwd: string): string[] {
  const root = path.join(cwd, "worktrees");
  const out: string[] = [];
  out.push(`ZAYA perf workers — ${new Date().toLocaleTimeString()} — /perf-workers-off to hide`);
  for (const task of TASKS) {
    const repo = path.join(root, task);
    const logA = path.join(root, "logs", `${task}.log`);
    const logB = path.join(repo, "worker.log");
    const pidA = path.join(root, "logs", `${task}.pid`);
    const pidB = path.join(repo, "worker.pid");
    const log = fs.existsSync(logB) ? logB : logA;
    const pid = fs.existsSync(pidB) ? pidB : pidA;
    out.push("");
    out.push(`◆ ${task} [${branchAndHead(repo)}] [${pidAlive(pid)}]`);
    for (const line of tail(log, linesPerTask)) out.push(`  ${line}`);
  }
  return out.slice(0, 80);
}

function refresh(ctx: ExtensionContext) {
  lastCtx = ctx;
  ctx.ui.setWidget(WIDGET, buildLines(ctx.cwd), { placement: "belowEditor" });
  const running = TASKS.filter((task) => {
    const root = path.join(ctx.cwd, "worktrees");
    const pid = fs.existsSync(path.join(root, task, "worker.pid"))
      ? path.join(root, task, "worker.pid")
      : path.join(root, "logs", `${task}.pid`);
    return pidAlive(pid).startsWith("running:");
  }).length;
  ctx.ui.setStatus(WIDGET, `perf workers ${running}/${TASKS.length}`);
}

function start(ctx: ExtensionContext) {
  lastCtx = ctx;
  if (timer) clearInterval(timer);
  refresh(ctx);
  timer = setInterval(() => {
    if (lastCtx) refresh(lastCtx);
  }, 2000);
}

function stop(ctx: ExtensionContext) {
  if (timer) clearInterval(timer);
  timer = undefined;
  ctx.ui.setWidget(WIDGET, undefined, { placement: "belowEditor" });
  ctx.ui.setStatus(WIDGET, undefined as any);
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    // Auto-start if worktrees exists, otherwise stay quiet.
    if (fs.existsSync(path.join(ctx.cwd, "worktrees"))) start(ctx);
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    stop(ctx);
  });

  pi.registerCommand("perf-workers", {
    description: "Show/refresh the throw-away ZAYA perf worker log widget",
    handler: async (args, ctx) => {
      const n = Number((args || "").trim());
      if (Number.isFinite(n) && n > 0) linesPerTask = Math.min(20, Math.floor(n));
      start(ctx);
      ctx.ui.notify(`Showing ZAYA perf worker logs (${linesPerTask} lines/task).`, "info");
    },
  });

  pi.registerCommand("perf-workers-off", {
    description: "Hide the throw-away ZAYA perf worker log widget",
    handler: async (_args, ctx) => {
      stop(ctx);
      ctx.ui.notify("Hidden ZAYA perf worker logs.", "info");
    },
  });
}
