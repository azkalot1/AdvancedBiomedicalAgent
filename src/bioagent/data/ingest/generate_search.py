from typing import Any

import psycopg2

# Assume these are in your project's environment
from curebench.data.ingest.config import DEFAULT_CONFIG, DatabaseConfig

# --- Configuration (Unchanged) ---
TABLE_VECTOR_CONFIG: list[dict[str, Any]] = [
    # ... (Your configuration list remains exactly the same)
    {"table_name": "ctgov_brief_summaries", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_browse_conditions", "vector_column": "term_vector", "source_columns": ["downcase_mesh_term"]},
    {"table_name": "ctgov_browse_interventions", "vector_column": "term_vector", "source_columns": ["downcase_mesh_term"]},
    {"table_name": "ctgov_conditions", "vector_column": "name_vector", "source_columns": ["downcase_name"]},
    {"table_name": "ctgov_design_groups", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_design_outcomes", "vector_column": "search_vector", "source_columns": ["description", "measure"]},
    {"table_name": "ctgov_detailed_descriptions", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_interventions", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_outcome_measurements", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_outcomes", "vector_column": "description_vector", "source_columns": ["description"]},
    {"table_name": "ctgov_result_groups", "vector_column": "search_vector", "source_columns": ["title", "description"]},
]


def check_column_exists(cursor, table_name: str, column_name: str) -> bool:
    """Checks if a column already exists in a table using a synchronous cursor."""
    query = """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s;
    """
    cursor.execute(query, (table_name, column_name))
    return cursor.fetchone() is not None


def create_full_text_search_indexes_sync(db_config: DatabaseConfig):
    """
    Connects to the database synchronously and creates tsvector columns and GIN indexes.
    """
    print("Starting synchronous full-text search indexing process...")
    conn = None
    try:
        # Establish a synchronous connection
        conn = psycopg2.connect(
            host=db_config.host, port=db_config.port, dbname=db_config.database, user=db_config.user, password=db_config.password
        )

        for config in TABLE_VECTOR_CONFIG:
            table = config["table_name"]
            vector_col = config["vector_column"]
            source_cols = config["source_columns"]
            index_name = f"idx_gin_{table}_{vector_col}"

            # Use a cursor for all operations on this table
            with conn.cursor() as cursor:
                print(f"\n--- Processing table: {table} ---")

                # 1. Add the tsvector column if it doesn't exist
                if not check_column_exists(cursor, table, vector_col):
                    print(f"  Adding column '{vector_col}'...")
                    alter_query = f"ALTER TABLE {table} ADD COLUMN {vector_col} tsvector;"
                    cursor.execute(alter_query)
                    print(f"  Column '{vector_col}' added.")
                else:
                    print(f"  Column '{vector_col}' already exists, skipping.")

                # 2. Populate the tsvector column
                print(f"  Populating '{vector_col}' from column(s): {', '.join(source_cols)}...")
                concatenated_sources = " || ' ' || ".join([f"coalesce({col}, '')" for col in source_cols])
                update_query = f"UPDATE {table} SET {vector_col} = to_tsvector('english', {concatenated_sources});"
                cursor.execute(update_query)
                print(f"  Column '{vector_col}' populated. This may take a while to commit.")

                # 3. Create the GIN index if it doesn't exist
                print(f"  Creating GIN index '{index_name}'...")
                index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} USING GIN({vector_col});"
                cursor.execute(index_query)
                print(f"  Index '{index_name}' created or already exists.")

            # Commit the transaction for the current table
            print(f"  Committing changes for table '{table}'...")
            conn.commit()
            print(f"  Changes for '{table}' committed successfully.")

        print("\n--- Full-text search indexing process completed successfully! ---")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"\n--- An error occurred: {error} ---")
        if conn:
            print("  Rolling back the last transaction...")
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")


# --- How to Run This Script ---
if __name__ == "__main__":
    # To run this script, execute it from your terminal.
    # It will connect, perform the operations, and wait for each to finish.
    create_full_text_search_indexes_sync(DEFAULT_CONFIG)
