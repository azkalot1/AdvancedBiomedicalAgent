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
    config: DatabaseConfig, output_file: Path, sample_rows: int = 2, table_names_suffix: str = "",
    max_value_length: int = 150, include_summary_stats: bool = True
) -> None:
    """
    Extract table definitions and sample rows to a text file.

    Args:
        config: Database configuration
        output_file: Path to save the text file
        sample_rows: Number of sample rows to extract per table
        table_names_suffix: suffix to subset of tables to extract (default: '')
        max_value_length: Maximum length for displayed values (default: 150 chars)
        include_summary_stats: Include data summary statistics (default: True)
    """
    print(f"🔍 Extracting table definitions and {sample_rows} sample rows per table...")
    print(f"📁 Output file: {output_file}")

    def format_value(value, max_length: int = max_value_length) -> str:
        """Format a value for display with optional truncation."""
        if isinstance(value, (bytes, memoryview)):
            return f"<binary data: {len(value)} bytes>"
        elif hasattr(value, 'isoformat'):  # datetime objects
            value_str = value.isoformat()
        elif value is None:
            return "NULL"
        else:
            value_str = str(value)
        
        # Truncate if too long
        if len(value_str) > max_length:
            return f"{value_str[:max_length]}... [TRUNCATED: {len(value_str)} chars total]"
        return value_str

    def get_column_stats(column_data: list) -> dict:
        """Calculate basic statistics for a column."""
        null_count = sum(1 for v in column_data if v is None)
        non_null_count = len(column_data) - null_count
        
        stats = {
            "null_count": null_count,
            "non_null_count": non_null_count,
            "null_percentage": (null_count / len(column_data) * 100) if column_data else 0
        }
        
        # Type-specific stats
        numeric_values = []
        for v in column_data:
            if v is not None and isinstance(v, (int, float)):
                numeric_values.append(v)
        
        if numeric_values:
            stats["min"] = min(numeric_values)
            stats["max"] = max(numeric_values)
            stats["avg"] = sum(numeric_values) / len(numeric_values)
        
        return stats

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
                        # Get total row count
                        try:
                            cur.execute(f'SELECT COUNT(*) as cnt FROM "{schema_name}"."{table_name}"')
                            total_rows = cur.fetchone()['cnt']
                            output_lines.append(f"\nTOTAL ROWS: {total_rows}")
                        except Exception as e:
                            output_lines.append(f"\nTOTAL ROWS: Error retrieving count: {str(e)}")

                        cur.execute(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT {sample_rows}')
                        sample_rows_data = cur.fetchall()

                        if sample_rows_data:
                            output_lines.append(f"\nSAMPLE DATA ({len(sample_rows_data)} rows):")
                            
                            # Collect all values per column for stats
                            column_data_map = {col['column_name']: [] for col in columns}
                            
                            for row_idx, row in enumerate(sample_rows_data, 1):
                                output_lines.append(f"\nRow {row_idx}:")
                                for key, value in row.items():
                                    value_str = format_value(value)
                                    output_lines.append(f"  {key}: {value_str}")
                                    if key in column_data_map:
                                        column_data_map[key].append(value)
                            
                            # Add summary statistics if enabled
                            if include_summary_stats:
                                output_lines.append("\n" + "-" * 40)
                                output_lines.append("SUMMARY STATISTICS (from sample):")
                                for col_name, col_values in column_data_map.items():
                                    stats = get_column_stats(col_values)
                                    output_lines.append(f"\n  {col_name}:")
                                    output_lines.append(f"    Non-NULL: {stats['non_null_count']}/{len(col_values)} ({100-stats['null_percentage']:.1f}%)")
                                    if 'min' in stats:
                                        output_lines.append(f"    Range: {stats['min']} to {stats['max']}")
                                        output_lines.append(f"    Avg: {stats['avg']:.2f}")
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
    parser.add_argument(
        "--max-value-length", "-m", type=int, default=150,
        help="Maximum length for displayed values before truncation (default: 150 chars)"
    )
    parser.add_argument(
        "--no-summary-stats", action="store_true",
        help="Disable summary statistics in output (default: False)"
    )
    args = parser.parse_args()

    print("🔍 Database Schema Extraction")
    print("=" * 50)
    print(f"🔗 Connected to: {DEFAULT_CONFIG.host}:{DEFAULT_CONFIG.port}/{DEFAULT_CONFIG.database}")
    print(f"👤 User: {DEFAULT_CONFIG.user}")
    print(f"📁 Output: {args.output}")
    print(f"📊 Sample rows: {args.sample_rows}")
    print(f"📏 Max value length: {args.max_value_length} chars")
    print(f"📈 Summary stats: {not args.no_summary_stats}")

    try:
        extract_schema_and_examples(
            DEFAULT_CONFIG,
            args.output,
            args.sample_rows,
            args.table_names_suffix,
            args.max_value_length,
            not args.no_summary_stats
        )
        print("\n🎉 Schema extraction completed successfully!")
    except Exception as e:
        print(f"\n❌ Schema extraction failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
