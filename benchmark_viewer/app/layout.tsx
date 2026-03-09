import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Benchmark Viewer",
  description: "Local benchmark results explorer for AdvancedBiomedicalAgent"
};

export default function RootLayout({ children }: { children: React.ReactNode }): React.ReactElement {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
