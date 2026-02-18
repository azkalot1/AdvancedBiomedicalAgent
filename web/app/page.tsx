import { WorkbenchShell } from "@/components/workbench/workbench-shell";
import { loadInitialWorkbenchData } from "@/lib/server/initial-data";

export const dynamic = "force-dynamic";

export default async function HomePage(): Promise<React.ReactElement> {
  const initialData = await loadInitialWorkbenchData();
  return <WorkbenchShell initialData={initialData} />;
}
