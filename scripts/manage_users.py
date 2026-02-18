#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.extras import RealDictCursor


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

APP_DATABASE_URL = (
    os.environ.get("APP_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
    or "postgresql://postgres:postgres@localhost:5432/coscientist_app"
)

APP_USERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_users (
    id              SERIAL PRIMARY KEY,
    email           VARCHAR(255) UNIQUE NOT NULL,
    name            VARCHAR(255) NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    role            VARCHAR(50) DEFAULT 'user',
    is_active       BOOLEAN DEFAULT true,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login      TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_app_users_email ON app_users(email);
"""


DEMO_USERS: list[dict[str, str]] = [
    {"email": "admin@coscientist.com", "name": "demo_admin_user", "role": "admin"},
    {"email": "user_1@coscientist.com", "name": "user_1", "role": "user"},
    {"email": "user_2@coscientist.com", "name": "user_2", "role": "user"},
    {"email": "user_3@coscientist.com", "name": "user_3", "role": "user"},
    {"email": "user_4@coscientist.com", "name": "user_4", "role": "user"},
    {"email": "user_5@coscientist.com", "name": "user_5", "role": "user"},
    {"email": "user_6@coscientist.com", "name": "user_6", "role": "user"},
    {"email": "user_7@coscientist.com", "name": "user_7", "role": "user"},
    {"email": "user_8@coscientist.com", "name": "user_8", "role": "user"},
    {"email": "user_9@coscientist.com", "name": "user_9", "role": "user"},
]


def get_conn():
    try:
        return psycopg2.connect(APP_DATABASE_URL, cursor_factory=RealDictCursor)
    except psycopg2.OperationalError as exc:
        _raise_connection_help(exc, APP_DATABASE_URL)
        raise


def _raise_connection_help(exc: psycopg2.OperationalError, db_url: str) -> None:
    parsed = urlparse(db_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    db_name = parsed.path.lstrip("/") or "postgres"
    db_user = parsed.username or "(empty)"
    msg = str(exc).strip()
    debug = msg or repr(exc)
    if "Connection refused" in msg:
        print("❌ Could not connect to app database server.")
        print(f"   Target: {host}:{port}/{db_name} as {db_user}")
        print("   Server is not running or not listening on this port.")
        print("   If using local Postgres service, start it and retry.")
        print("   If using docker-compose.dev.yml, run:")
        print("   docker compose -f docker-compose.dev.yml up -d postgres-app")
    elif "password authentication failed" in msg.lower():
        print("❌ Postgres authentication failed for app database connection.")
        print(f"   Target: {host}:{port}/{db_name} as {db_user}")
        print("   The username/password in APP_DATABASE_URL (or DATABASE_URL) is incorrect for this server.")
        db_user_fallback = os.environ.get("DB_USER", "").strip()
        db_pass_fallback = os.environ.get("DB_PASSWORD", "").strip()
        if db_user_fallback and db_pass_fallback:
            print("   You can reuse your data DB credentials on the same server, for example:")
            print(
                "   APP_DATABASE_URL="
                f"postgresql://{db_user_fallback}:{db_pass_fallback}@{host}:{port}/coscientist_app"
            )
            print(
                "   DATABASE_URL="
                f"postgresql://{db_user_fallback}:{db_pass_fallback}@{host}:{port}/coscientist_app"
            )
        print("   If you need DB creation with admin privileges, set APP_DATABASE_ADMIN_URL separately.")
    else:
        print("❌ Could not connect to app database.")
        print(f"   Target: {host}:{port}/{db_name} as {db_user}")
        print(f"   Error: {debug}")


def _parsed_db_info(db_url: str) -> dict[str, Any]:
    parsed = urlparse(db_url)
    db_name = parsed.path.lstrip("/") or "postgres"
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": db_name,
        "user": parsed.username or "postgres",
        "password": parsed.password or "postgres",
    }


def _sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _run_sudo_postgres_sql(sql_text: str, dbname: str | None = None) -> bool:
    if shutil.which("sudo") is None or shutil.which("psql") is None:
        return False

    cmd = ["sudo", "-u", "postgres", "psql", "-v", "ON_ERROR_STOP=1"]
    if dbname:
        cmd += ["-d", dbname]

    print("Attempting privileged PostgreSQL command via sudo (you may be prompted for password)...")
    result = subprocess.run(cmd, input=sql_text, text=True)
    return result.returncode == 0


def ensure_database() -> None:
    target = _parsed_db_info(APP_DATABASE_URL)
    admin_url = os.environ.get("APP_DATABASE_ADMIN_URL", "").strip()
    if admin_url:
        admin = _parsed_db_info(admin_url)
    else:
        admin = {**target, "dbname": os.environ.get("APP_DATABASE_ADMIN_DB", "postgres")}

    try:
        conn = psycopg2.connect(
            host=admin["host"],
            port=admin["port"],
            dbname=admin["dbname"],
            user=admin["user"],
            password=admin["password"],
        )
    except psycopg2.OperationalError as exc:
        _raise_connection_help(exc, APP_DATABASE_URL)
        raise

    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target["dbname"],))
        if cur.fetchone():
            print(f"Database exists: {target['dbname']}")
            return

        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target["dbname"])))
        print(f"Created database: {target['dbname']}")
    except psycopg2.Error as exc:
        msg = str(exc).lower()
        if "permission denied to create database" in msg:
            print(f"⚠️  No CREATE DATABASE permission for user '{target['user']}'.")
            if _run_sudo_postgres_sql(f"CREATE DATABASE {_sql_ident(target['dbname'])};"):
                print(f"✅ Created database '{target['dbname']}' via sudo postgres user.")
                return
            print(f"   Checking whether '{target['dbname']}' already exists and is accessible...")
            try:
                check_conn = psycopg2.connect(
                    host=target["host"],
                    port=target["port"],
                    dbname=target["dbname"],
                    user=target["user"],
                    password=target["password"],
                )
                check_conn.close()
                print(f"Database already accessible: {target['dbname']}")
                return
            except psycopg2.Error:
                print(f"❌ Failed to ensure database '{target['dbname']}': {exc}")
                print("   Create the database once with an admin account, then re-run users setup.")
                print("   Option A: set APP_DATABASE_ADMIN_URL to a superuser DSN.")
                print(
                    "   Option B: run manually (example): "
                    f"createdb -h {target['host']} -p {target['port']} -U postgres {target['dbname']}"
                )
                raise

        print(f"❌ Failed to ensure database '{target['dbname']}': {exc}")
        raise
    finally:
        cur.close()
        conn.close()


def _admin_db_url() -> str:
    return os.environ.get("APP_DATABASE_ADMIN_URL", "").strip()


def _ensure_schema_privileges_with_admin(target: dict[str, Any]) -> bool:
    admin_url = _admin_db_url()
    if not admin_url:
        return False

    admin = _parsed_db_info(admin_url)
    if not admin.get("dbname"):
        admin["dbname"] = target["dbname"]

    try:
        conn = psycopg2.connect(
            host=admin["host"],
            port=admin["port"],
            dbname=target["dbname"],
            user=admin["user"],
            password=admin["password"],
        )
    except psycopg2.Error as exc:
        print(f"⚠️  Could not connect with APP_DATABASE_ADMIN_URL to grant schema permissions: {exc}")
        return False

    conn.autocommit = True
    cur = conn.cursor()
    try:
        role = sql.Identifier(target["user"])
        dbname_ident = sql.Identifier(target["dbname"])
        cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(dbname_ident, role))
        cur.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(role))
        cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {}").format(role))
        cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {}").format(role))
        cur.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {}").format(role))
        cur.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {}").format(role))
        print(f"✅ Granted schema/table privileges on '{target['dbname']}' to '{target['user']}' via APP_DATABASE_ADMIN_URL")
        return True
    except psycopg2.Error as exc:
        print(f"⚠️  Failed granting schema privileges with APP_DATABASE_ADMIN_URL: {exc}")
        return False
    finally:
        cur.close()
        conn.close()


def _ensure_schema_privileges_with_sudo(target: dict[str, Any]) -> bool:
    role = _sql_ident(target["user"])
    dbname = _sql_ident(target["dbname"])
    sql_text = f"""
GRANT CONNECT ON DATABASE {dbname} TO {role};
GRANT USAGE, CREATE ON SCHEMA public TO {role};
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {role};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {role};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {role};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {role};
"""
    ok = _run_sudo_postgres_sql(sql_text, dbname=target["dbname"])
    if ok:
        print(f"✅ Granted schema/table privileges on '{target['dbname']}' to '{target['user']}' via sudo postgres user")
    return ok


def _print_schema_permission_fix_hint(target: dict[str, Any]) -> None:
    print("❌ Current app DB user cannot create tables in schema public.")
    print("   Fix once with admin account, then retry make users-setup.")
    print("   You can also rerun this command in a terminal with sudo access so it can auto-apply grants.")
    print("   Example (psql as admin):")
    print(f"   \\c {target['dbname']}")
    print(f"   GRANT USAGE, CREATE ON SCHEMA public TO {target['user']};")
    print(f"   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {target['user']};")
    print(f"   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {target['user']};")
    print(f"   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {target['user']};")
    print(f"   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {target['user']};")


def generate_password() -> str:
    chars = string.ascii_letters + string.digits
    part1 = "".join(secrets.choice(chars) for _ in range(4))
    part2 = "".join(secrets.choice(chars) for _ in range(4))
    part3 = "".join(secrets.choice(chars) for _ in range(4))
    return f"{part1}-{part2}-{part3}"


def hash_password(password: str) -> str:
    try:
        import bcrypt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("bcrypt is required for password operations. Run: pip install -e .") from exc
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def ensure_schema() -> None:
    ensure_database()

    target = _parsed_db_info(APP_DATABASE_URL)
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(APP_USERS_SCHEMA_SQL)
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        msg = str(exc).lower()
        if "permission denied for schema public" in msg:
            fixed = _ensure_schema_privileges_with_admin(target)
            if not fixed:
                fixed = _ensure_schema_privileges_with_sudo(target)
            if fixed:
                cur.execute(APP_USERS_SCHEMA_SQL)
                conn.commit()
            else:
                _print_schema_permission_fix_hint(target)
                raise
        else:
            raise
    finally:
        cur.close()
        conn.close()


def _write_credentials(credentials: list[dict[str, str]], output_path: Path) -> None:
    lines: list[str] = [
        "AI Co-Scientist - Demo Credentials",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
    ]
    for cred in credentials:
        lines.extend(
            [
                f"Name:     {cred['name']}",
                f"Email:    {cred['email']}",
                f"Password: {cred['password']}",
                f"Role:     {cred['role']}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines))


def cmd_seed(args: argparse.Namespace) -> None:
    ensure_schema()

    conn = get_conn()
    cur = conn.cursor()

    print("\nSeeding demo users...\n")
    print(f"{'Email':<35} {'Name':<20} {'Password':<18} {'Role'}")
    print("-" * 95)

    credentials: list[dict[str, str]] = []
    for user in DEMO_USERS:
        password = generate_password()
        password_hash = hash_password(password)

        cur.execute(
            """
            INSERT INTO app_users (email, name, password_hash, role)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET
                name = EXCLUDED.name,
                password_hash = EXCLUDED.password_hash,
                role = EXCLUDED.role,
                is_active = true
            RETURNING id
            """,
            (
                user["email"].lower().strip(),
                user["name"],
                password_hash,
                user["role"],
            ),
        )

        print(f"{user['email']:<35} {user['name']:<20} {password:<18} {user['role']}")
        credentials.append({**user, "password": password})

    conn.commit()
    cur.close()
    conn.close()

    output_path = Path(args.output)
    _write_credentials(credentials, output_path)
    print(f"\nSeeded {len(DEMO_USERS)} users.")
    print(f"Credentials saved to {output_path}")
    print("Delete credentials.txt after securely sharing passwords.\n")


def cmd_add(args: argparse.Namespace) -> None:
    ensure_schema()

    password = args.password or generate_password()
    password_hash = hash_password(password)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO app_users (email, name, password_hash, role)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (email) DO NOTHING
        RETURNING id
        """,
        (
            args.email.lower().strip(),
            args.name.strip(),
            password_hash,
            args.role,
        ),
    )
    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    if not result:
        print(f"User already exists: {args.email}")
        return

    print("\nUser created:")
    print(f"  Email:    {args.email}")
    print(f"  Name:     {args.name}")
    print(f"  Password: {password}")
    print(f"  Role:     {args.role}\n")


def cmd_list(_args: argparse.Namespace) -> None:
    ensure_schema()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, email, name, role, is_active, created_at, last_login
        FROM app_users
        ORDER BY created_at
        """
    )
    users = cur.fetchall()
    cur.close()
    conn.close()

    if not users:
        print("\nNo users found.\n")
        return

    print(f"\n{'ID':<5} {'Email':<35} {'Name':<20} {'Role':<8} {'Active':<8} {'Last Login'}")
    print("-" * 110)
    active_count = 0
    for user in users:
        is_active = bool(user["is_active"])
        active_count += 1 if is_active else 0
        last_login = user["last_login"].strftime("%Y-%m-%d %H:%M") if user["last_login"] else "never"
        active = "yes" if is_active else "no"
        print(
            f"{user['id']:<5} {user['email']:<35} {user['name']:<20} "
            f"{user['role']:<8} {active:<8} {last_login}"
        )
    print(f"\nTotal: {len(users)} users ({active_count} active)\n")


def cmd_reset_password(args: argparse.Namespace) -> None:
    ensure_schema()

    password = args.password or generate_password()
    password_hash = hash_password(password)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE app_users
        SET password_hash = %s
        WHERE email = %s
        RETURNING id, name
        """,
        (password_hash, args.email.lower().strip()),
    )
    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    if not result:
        print(f"User not found: {args.email}")
        return

    print(f"\nPassword reset for {result['name']} ({args.email})")
    print(f"  New password: {password}\n")


def _toggle_user_active(email: str, active: bool) -> dict[str, Any] | None:
    ensure_schema()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE app_users
        SET is_active = %s
        WHERE email = %s
        RETURNING id, name, email
        """,
        (active, email.lower().strip()),
    )
    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return result


def cmd_deactivate(args: argparse.Namespace) -> None:
    result = _toggle_user_active(args.email, active=False)
    if not result:
        print(f"User not found: {args.email}")
        return
    print(f"Deactivated: {result['name']} ({result['email']})")


def cmd_activate(args: argparse.Namespace) -> None:
    result = _toggle_user_active(args.email, active=True)
    if not result:
        print(f"User not found: {args.email}")
        return
    print(f"Activated: {result['name']} ({result['email']})")


def cmd_remove(args: argparse.Namespace) -> None:
    ensure_schema()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM app_users WHERE email = %s RETURNING id, name, email",
        (args.email.lower().strip(),),
    )
    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    if not result:
        print(f"User not found: {args.email}")
        return
    print(f"Removed: {result['name']} ({result['email']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage AI Co-Scientist users")
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="Seed demo users with generated passwords")
    p_seed.add_argument("--output", default="credentials.txt", help="Credentials output file (default: credentials.txt)")

    p_add = sub.add_parser("add", help="Add a single user")
    p_add.add_argument("--email", required=True)
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--password", help="Optional password; autogenerated if omitted")
    p_add.add_argument("--role", default="user", choices=["user", "admin"])

    sub.add_parser("list", help="List all users")

    p_reset = sub.add_parser("reset-password", help="Reset a user's password")
    p_reset.add_argument("--email", required=True)
    p_reset.add_argument("--password", help="Optional password; autogenerated if omitted")

    p_deactivate = sub.add_parser("deactivate", help="Deactivate a user")
    p_deactivate.add_argument("--email", required=True)

    p_activate = sub.add_parser("activate", help="Activate a user")
    p_activate.add_argument("--email", required=True)

    p_remove = sub.add_parser("remove", help="Remove a user")
    p_remove.add_argument("--email", required=True)

    sub.add_parser("init-db", help="Create app database if missing")
    sub.add_parser("init-schema", help="Create app_users table if missing")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    command_map = {
        "seed": cmd_seed,
        "add": cmd_add,
        "list": cmd_list,
        "reset-password": cmd_reset_password,
        "deactivate": cmd_deactivate,
        "activate": cmd_activate,
        "remove": cmd_remove,
        "init-db": lambda _args: ensure_database(),
        "init-schema": lambda _args: ensure_schema(),
    }
    try:
        command_map[args.command](args)
    except Exception as exc:
        detail = str(exc).strip() or repr(exc)
        print(f"Command failed: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
