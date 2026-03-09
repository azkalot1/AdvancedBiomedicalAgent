import { ViewerShell } from "@/src/components/viewer-shell";
import { loadRunIndexes } from "@/src/lib/server/runs";

export const dynamic = "force-dynamic";

export default async function HomePage(): Promise<React.ReactElement> {
  const initialRuns = await loadRunIndexes();
  return <ViewerShell initialRuns={initialRuns} />;
}
