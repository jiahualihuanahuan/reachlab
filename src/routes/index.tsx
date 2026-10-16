import { createFileRoute } from "@tanstack/react-router";
import { YellowLab } from "@/components/yellow-lab";

export const Route = createFileRoute("/")({
  component: YellowLab,
});
