import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.registerProvider("zaya-mlx", {
    name: "ZAYA MLX (local)",
    baseUrl: "http://127.0.0.1:8123/v1",
    apiKey: "zaya-local",
    api: "openai-completions",
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsUsageInStreaming: false,
      maxTokensField: "max_tokens",
    },
    models: [
      {
        id: "zaya-mlx",
        name: "ZAYA1-8B MLX local",
        reasoning: false,
        input: ["text"],
        contextWindow: 8192,
        maxTokens: 2048,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      },
    ],
  });

  pi.registerCommand("zaya-mlx-server", {
    description: "Show how to start the local ZAYA MLX OpenAI-compatible server",
    handler: async (_args, ctx) => {
      ctx.ui.notify(
        "Start server with: uv run python scripts/server_zaya_mlx.py --port 8123, then select zaya-mlx/zaya-mlx",
        "info",
      );
    },
  });
}
