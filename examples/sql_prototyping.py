#!/usr/bin/env python3
"""
SQL Prototyping Script for BiomedAgent

A simple, interactive way to execute SQL queries against the biomedical database.
Perfect for rapid prototyping, data exploration, and debugging.

Usage:
    python examples/sql_prototyping.py
    python examples/sql_prototyping.py "SELECT COUNT(*) FROM ctgov_studies"
    python examples/sql_prototyping.py --file my_query.sql
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import psycopg2
    import psycopg2.extras
    from bioagent.data.ingest.config import DEFAULT_CONFIG
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the project root and have installed dependencies:")
    print("  cd /path/to/AdvancedBiomedicalAgent")
    print("  pip install -e .")
    sys.exit(1)


class SQLPrototyper:
    """Simple SQL query executor with formatting options."""
    
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """Connect to the database."""
        try:
            self.conn = psycopg2.connect(
                host=DEFAULT_CONFIG.host,
                port=DEFAULT_CONFIG.port,
                dbname=DEFAULT_CONFIG.database,
                user=DEFAULT_CONFIG.user,
                password=DEFAULT_CONFIG.password
            )
            print(f"✅ Connected to {DEFAULT_CONFIG.database} at {DEFAULT_CONFIG.host}:{DEFAULT_CONFIG.port}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            sys.exit(1)
    
    def execute(self, query: str, format_type: str = "table", limit: int = 100) -> None:
        """Execute a query and format the results."""
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Add LIMIT for SELECT queries if not present
                if query.strip().upper().startswith('SELECT') and 'LIMIT' not in query.upper():
                    query = f"{query.rstrip(';')} LIMIT {limit};"
                
                print(f"🔍 Executing query...")
                cursor.execute(query)
                
                # Handle non-SELECT queries
                if cursor.description is None:
                    self.conn.commit()
                    print(f"✅ Query executed. Rows affected: {cursor.rowcount}")
                    return
                
                results = cursor.fetchall()
                
                if not results:
                    print("📭 No results returned.")
                    return
                
                print(f"📊 Returned {len(results)} row(s)")
                print("-" * 60)
                
                if format_type == "json":
                    self._format_json(results)
                elif format_type == "csv":
                    self._format_csv(results)
                else:
                    self._format_table(results)
                    
        except Exception as e:
            print(f"❌ Query error: {e}")
            self.conn.rollback()
    
    def _format_table(self, results: list) -> None:
        """Format results as a table."""
        if not results:
            return
        
        headers = list(results[0].keys())
        
        # Calculate column widths
        col_widths = []
        for header in headers:
            max_width = len(header)
            for row in results[:20]:  # Sample first 20 rows
                cell_value = str(row[header]) if row[header] is not None else "NULL"
                max_width = max(max_width, min(len(cell_value), 50))
            col_widths.append(max_width)
        
        # Print header
        header_row = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
        print(header_row)
        print("-" * len(header_row))
        
        # Print rows
        for row in results:
            row_cells = []
            for i, header in enumerate(headers):
                cell_value = str(row[header]) if row[header] is not None else "NULL"
                # Truncate long values
                if len(cell_value) > col_widths[i]:
                    cell_value = cell_value[:col_widths[i]-3] + "..."
                row_cells.append(cell_value.ljust(col_widths[i]))
            print(" | ".join(row_cells))
    
    def _format_json(self, results: list) -> None:
        """Format results as JSON."""
        print(json.dumps([dict(row) for row in results], indent=2, default=str))
    
    def _format_csv(self, results: list) -> None:
        """Format results as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        if results:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows([dict(row) for row in results])
        print(output.getvalue())
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n🔍 SQL Prototyping - Interactive Mode")
        print("=" * 50)
        print("Enter SQL queries (type 'quit' or 'exit' to stop)")
        print("Commands:")
        print("  \\d [table]     - Describe table structure")
        print("  \\dt            - List all tables")
        print("  \\q             - Quit")
        print("  help           - Show example queries")
        print("-" * 50)
        
        while True:
            try:
                query = input("\nSQL> ").strip()
                
                if query.lower() in ['quit', 'exit', '\\q']:
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if query.startswith('\\dt'):
                    query = """
                        SELECT table_name, table_type 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        ORDER BY table_name
                    """
                elif query.startswith('\\d '):
                    table_name = query[3:].strip()
                    query = f"""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = '{table_name}'
                        ORDER BY ordinal_position
                    """
                elif query.startswith('\\d') and len(query) > 2:
                    table_name = query[2:].strip()
                    query = f"""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = '{table_name}'
                        ORDER BY ordinal_position
                    """
                
                if query:
                    self.execute(query)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
    
    def _show_help(self):
        """Show example queries."""
        print("\n📚 Example Queries:")
        print("-" * 30)
        
        examples = [
            ("Database Overview", "SELECT COUNT(*) FROM ctgov_studies"),
            ("List Tables", "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"),
            ("Drug Search", "SELECT name, formula FROM drugcentral_drugs WHERE name ILIKE '%aspirin%' LIMIT 5"),
            ("Trial Phases", "SELECT phase, COUNT(*) FROM ctgov_studies GROUP BY phase ORDER BY count DESC"),
            ("Recent Trials", "SELECT nct_id, brief_title, start_date FROM ctgov_studies WHERE start_date IS NOT NULL ORDER BY start_date DESC LIMIT 5"),
            ("Vector Columns", "SELECT table_name, column_name FROM information_schema.columns WHERE data_type = 'tsvector'"),
            ("Index Status", "SELECT schemaname, tablename, indexname FROM pg_indexes WHERE indexname LIKE '%gin%'")
        ]
        
        for title, query in examples:
            print(f"\n• {title}:")
            print(f"  {query}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SQL Prototyping for BiomedAgent")
    parser.add_argument("query", nargs="?", help="SQL query to execute")
    parser.add_argument("--file", "-f", help="Execute SQL from file")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table", help="Output format")
    parser.add_argument("--limit", type=int, default=100, help="Row limit for SELECT queries")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    prototyper = SQLPrototyper()
    
    try:
        if args.interactive or (not args.query and not args.file):
            prototyper.interactive_mode()
        elif args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return
            query = file_path.read_text()
            prototyper.execute(query, args.format, args.limit)
        elif args.query:
            prototyper.execute(args.query, args.format, args.limit)
    finally:
        prototyper.close()


if __name__ == "__main__":
    main()

