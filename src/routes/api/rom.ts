import { createFileRoute } from "@tanstack/react-router";

const ROM =
  process.env.ROM_PATH ||
  "/workspace/attachments/Pokemon - Yellow Version (UE) [C][!].gbc";

export const Route = createFileRoute("/api/rom")({
  server: {
    handlers: {
      GET: async () => {
        const { existsSync } = await import("node:fs");
        const { readFile } = await import("node:fs/promises");
        if (!existsSync(ROM)) {
          return new Response("Cartridge not on this machine", { status: 404 });
        }
        const bytes = await readFile(ROM);
        return new Response(bytes, {
          headers: {
            "content-type": "application/octet-stream",
            "cache-control": "no-store",
          },
        });
      },
    },
  },
});
