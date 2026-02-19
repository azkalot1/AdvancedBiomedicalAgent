import type { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import fs from "node:fs";
import path from "node:path";
import { Pool, type PoolConfig } from "pg";

function normalizeEnv(value: string | undefined): string {
  return (value ?? "").trim().replace(/^['"]|['"]$/g, "");
}

function readEnvVarFromFile(filePath: string, key: string): string {
  if (!fs.existsSync(filePath)) {
    return "";
  }

  const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!match) {
      continue;
    }
    if (match[1] !== key) {
      continue;
    }
    return normalizeEnv(match[2]);
  }
  return "";
}

function getEnvValue(key: string): string {
  const direct = normalizeEnv(process.env[key]);
  if (direct) {
    return direct;
  }

  const candidates = [
    path.resolve(process.cwd(), ".env.local"),
    path.resolve(process.cwd(), ".env"),
    path.resolve(process.cwd(), "..", ".env.local"),
    path.resolve(process.cwd(), "..", ".env")
  ];
  for (const candidate of candidates) {
    const value = readEnvVarFromFile(candidate, key);
    if (value) {
      process.env[key] = value;
      return value;
    }
  }
  return "";
}

function buildPoolConfig(): PoolConfig {
  const databaseUrl = getEnvValue("DATABASE_URL") || getEnvValue("APP_DATABASE_URL");
  if (!databaseUrl) {
    throw new Error("Missing DATABASE_URL (or APP_DATABASE_URL) for NextAuth.");
  }

  let parsed: URL;
  try {
    parsed = new URL(databaseUrl);
  } catch {
    throw new Error("Invalid DATABASE_URL format. Expected postgres://user:pass@host:port/dbname");
  }

  const database = parsed.pathname.replace(/^\//, "");
  if (!database) {
    throw new Error("Invalid DATABASE_URL: missing database name.");
  }

  const user = decodeURIComponent(parsed.username || "");
  const password = decodeURIComponent(parsed.password || getEnvValue("PGPASSWORD") || "");

  if (!user) {
    throw new Error("Invalid DATABASE_URL: missing username.");
  }

  return {
    host: parsed.hostname,
    port: parsed.port ? Number(parsed.port) : 5432,
    database,
    user,
    password,
    max: 5,
    ssl: process.env.NODE_ENV === "production" ? { rejectUnauthorized: false } : false
  };
}

let pool: Pool | null = null;

function getPool(): Pool {
  if (pool) {
    return pool;
  }
  pool = new Pool(buildPoolConfig());
  return pool;
}

type UserRow = {
  id: number;
  email: string;
  name: string;
  password_hash: string;
  role: string;
  is_active: boolean;
};

function logAuthReject(reason: string, email: string): void {
  if (process.env.NODE_ENV === "production") {
    return;
  }
  console.warn(`[auth] credentials rejected: ${reason} (email=${email})`);
}

export const authOptions: NextAuthOptions = {
  secret: process.env.NEXTAUTH_SECRET ?? "dev-nextauth-secret-change-me",
  providers: [
    CredentialsProvider({
      name: "Sign In",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        try {
          const pool = getPool();
          const email = credentials.email.toLowerCase().trim();
          const result = await pool.query<UserRow>(
            "SELECT id, email, name, password_hash, role, is_active FROM app_users WHERE email = $1",
            [email]
          );
          const user = result.rows[0];

          if (!user) {
            logAuthReject("user_not_found", email);
            return null;
          }

          if (!user.is_active) {
            logAuthReject("user_inactive", email);
            return null;
          }

          const validPassword = await bcrypt.compare(credentials.password, user.password_hash);
          if (!validPassword) {
            logAuthReject("password_mismatch", email);
            return null;
          }

          await pool.query("UPDATE app_users SET last_login = NOW() WHERE id = $1", [user.id]);

          return {
            id: String(user.id),
            email: user.email,
            name: user.name,
            role: user.role
          };
        } catch (error) {
          console.error("Credentials authorize error", error);
          return null;
        }
      }
    })
  ],
  session: {
    strategy: "jwt",
    maxAge: 7 * 24 * 60 * 60
  },
  pages: {
    signIn: "/login"
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.role = user.role;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = String(token.id ?? "");
        session.user.role = String(token.role ?? "user");
      }
      return session;
    }
  }
};
