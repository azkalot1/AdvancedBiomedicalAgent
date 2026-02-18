import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#131826",
          raised: "#1b2233",
          overlay: "#253047",
          edge: "#33415f"
        },
        accent: {
          blue: "#4b92ff",
          cyan: "#2dd4bf",
          ink: "#0f172a"
        },
        status: {
          success: "#22c55e",
          running: "#f59e0b",
          error: "#ef4444",
          queued: "#94a3b8"
        }
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"]
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(75,146,255,0.24), 0 12px 40px rgba(17,24,39,0.45)"
      }
    }
  },
  plugins: []
};

export default config;
