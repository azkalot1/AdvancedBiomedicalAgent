#!/usr/bin/env python3
"""
Extract database schema and sample data for PostgreSQL database.

Creates a simple text file with table definitions and sample rows.
"""
from __future__ import annotations

from pathlib import Path

import psycopg2
import psycopg2.extras

# Handle imports for both direct execution and module import
try:
    from .config import DatabaseConfig, get_connection
except ImportError:
    from config import DatabaseConfig, get_connection



def extract_schema_and_examples(
    config: DatabaseConfig, output_file: Path, sample_rows: int = 2, table_names_suffix: str = ""
) -> None:
    """
    Extract table definitions and sample rows to a text file.

    Args:
        config: Database configuration
        output_file: Path to save the text file
        sample_rows: Number of sample rows to extract per table
    """
    print(f"🔍 Extracting table definitions and {sample_rows} sample rows per table...")
    print(f"📁 Output file: {output_file}")

    try:
        with get_connection(config) as con:
            with con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get all tables in the database
                cur.execute(
                    """
                    SELECT
                        schemaname,
                        tablename
                    FROM pg_tables
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                    ORDER BY schemaname, tablename
                """
                )
                tables = cur.fetchall()

                # Get all views in the database
                cur.execute(
                    """
                    SELECT
                        schemaname,
                        viewname as tablename
                    FROM pg_views
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY schemaname, viewname
                """
                )
                views = cur.fetchall()

                # Combine tables and views
                all_objects = list(tables) + list(views)

                print(f"📊 Found {len(tables)} tables and {len(views)} views")

                # Process each table/view
                output_lines = []
                for i, obj in enumerate(all_objects, 1):
                    schema_name = obj['schemaname']
                    table_name = obj['tablename']
                    if table_names_suffix and table_names_suffix not in table_name:
                        continue
                    full_name = f"{schema_name}.{table_name}"

                    print(f"   [{i}/{len(all_objects)}] Processing {full_name}...")

                    # Add table header
                    output_lines.append(f"\n{'='*60}")
                    output_lines.append(f"TABLE: {full_name}")
                    output_lines.append(f"{'='*60}")

                    # Get column information
                    cur.execute(
                        """
                        SELECT
                            column_name,
                            data_type,
                            is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position
                    """,
                        (schema_name, table_name),
                    )

                    columns = cur.fetchall()

                    # Add column definitions
                    output_lines.append("\nCOLUMNS:")
                    for col in columns:
                        nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                        output_lines.append(f"  {col['column_name']}: {col['data_type']} {nullable}")

                    # Get sample data
                    try:
                        cur.execute(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT {sample_rows}')
                        sample_rows_data = cur.fetchall()

                        if sample_rows_data:
                            output_lines.append(f"\nSAMPLE DATA ({len(sample_rows_data)} rows):")
                            for row_idx, row in enumerate(sample_rows_data, 1):
                                output_lines.append(f"\nRow {row_idx}:")
                                for key, value in row.items():
                                    # Handle non-serializable types
                                    if isinstance(value, (bytes, memoryview)):
                                        value_str = f"<binary data: {len(value)} bytes>"
                                    elif hasattr(value, 'isoformat'):  # datetime objects
                                        value_str = value.isoformat()
                                    elif value is None:
                                        value_str = "NULL"
                                    else:
                                        value_str = str(value)
                                    output_lines.append(f"  {key}: {value_str}")
                        else:
                            output_lines.append("\nSAMPLE DATA: No data found")

                    except Exception as e:
                        output_lines.append(f"\nSAMPLE DATA: Error retrieving data: {str(e)}")

                # Write to file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output_lines))

                print("✅ Schema extraction completed successfully!")
                print(f"📊 Extracted {len(all_objects)} objects")
                print(f"💾 Saved to: {output_file}")
                print(f"📏 File size: {output_file.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        print(f"❌ Failed to extract schema: {e}")
        raise


def main():
    """Main function for standalone execution."""
    try:
        from .config import DEFAULT_CONFIG
    except ImportError:
        # Handle direct execution
        from config import DEFAULT_CONFIG

    import argparse

    parser = argparse.ArgumentParser(description="Extract database schema and sample data")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("database_schema.txt"),
        help="Output text file path (default: database_schema.txt)",
    )
    parser.add_argument("--sample-rows", "-s", type=int, default=2, help="Number of sample rows per table (default: 2)")
    parser.add_argument(
        "--table-names-suffix", "-t", type=str, default="", help="suffix to subset of tables to extract (default: '')"
    )
    args = parser.parse_args()

    print("🔍 Database Schema Extraction")
    print("=" * 50)
    print(f"🔗 Connected to: {DEFAULT_CONFIG.host}:{DEFAULT_CONFIG.port}/{DEFAULT_CONFIG.database}")
    print(f"👤 User: {DEFAULT_CONFIG.user}")
    print(f"📁 Output: {args.output}")
    print(f"📊 Sample rows: {args.sample_rows}")

    try:
        extract_schema_and_examples(DEFAULT_CONFIG, args.output, args.sample_rows, args.table_names_suffix)
        print("\n🎉 Schema extraction completed successfully!")
    except Exception as e:
        print(f"\n❌ Schema extraction failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
