#!/usr/bin/env python3
"""
BiomedAgent Search Examples - Standalone Python Scripts

This file contains ready-to-run examples showing how to:
1. Execute SQL queries directly
2. Perform drug/FDA searches  
3. Search clinical trials
4. Debug and inspect results

Usage:
    python examples/search_examples.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import psycopg2
    import psycopg2.extras
    from bioagent.data.ingest.config import DEFAULT_CONFIG
    from bioagent.data.app.openfda_and_dailymed_searches import (
        unified_search_async, UnifiedSearchInput
    )
    from bioagent.data.app.clinical_trial_searches import (
        clinical_trials_search_async, ClinicalTrialsSearchInput, SearchKind
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the project root and have installed dependencies:")
    print("  cd /path/to/AdvancedBiomedicalAgent")
    print("  pip install -e .")
    sys.exit(1)


def execute_sql_query(query: str, limit: int = 10) -> list[dict]:
    """
    Execute a SQL query and return results as list of dictionaries.
    
    Example:
        results = execute_sql_query("SELECT COUNT(*) as total FROM ctgov_studies")
        print(f"Total studies: {results[0]['total']}")
    """
    print(f"🔍 Executing: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    try:
        conn = psycopg2.connect(
            host=DEFAULT_CONFIG.host,
            port=DEFAULT_CONFIG.port,
            dbname=DEFAULT_CONFIG.database,
            user=DEFAULT_CONFIG.user,
            password=DEFAULT_CONFIG.password
        )
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Add LIMIT if not present for SELECT queries
            if query.strip().upper().startswith('SELECT') and 'LIMIT' not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit};"
            
            cursor.execute(query)
            
            if cursor.description is None:
                # Non-SELECT query
                conn.commit()
                return [{"rows_affected": cursor.rowcount}]
            
            results = [dict(row) for row in cursor.fetchall()]
            print(f"✅ Returned {len(results)} row(s)")
            return results
            
    except Exception as e:
        print(f"❌ SQL Error: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()


async def search_drugs_example():
    """Example: Search for drug information."""
    print("\n" + "="*60)
    print("🔍 DRUG SEARCH EXAMPLE")
    print("="*60)
    
    # Example 1: Basic property lookup
    print("\n1. Basic drug property lookup:")
    search_input = UnifiedSearchInput(drug_names=['Lisinopril', 'Metformin'])
    results = await unified_search_async(DEFAULT_CONFIG, search_input)
    
    print(f"Status: {results.status}")
    if results.results:
        for drug in results.results:
            print(f"  • {drug.product_name}")
            if drug.properties:
                props = drug.properties
                print(f"    Formula: {props.formula or 'N/A'}")
                print(f"    SMILES: {props.smiles[:50] + '...' if props.smiles and len(props.smiles) > 50 else props.smiles or 'N/A'}")
    
    # Example 2: Search specific sections
    print("\n2. Search specific sections (warnings, dosage):")
    search_input = UnifiedSearchInput(
        drug_names=['Warfarin'],
        section_queries=['warnings', 'dosage and administration'],
        sections_per_query=3,
        result_limit=5
    )
    results = await unified_search_async(DEFAULT_CONFIG, search_input)
    
    if results.results:
        for drug in results.results:
            print(f"  • {drug.product_name} - Found {len(drug.sections)} sections:")
            for section in drug.sections[:3]:  # Show first 3
                print(f"    - [{section.source}] {section.section_name}")
                print(f"      {section.text[:100]}...")
    
    # Example 3: Keyword discovery
    print("\n3. Keyword discovery (find drugs containing 'heart failure'):")
    search_input = UnifiedSearchInput(
        keyword_query=['heart failure', 'cardiac'],
        top_n_drugs=3
    )
    results = await unified_search_async(DEFAULT_CONFIG, search_input)
    
    if results.results:
        print(f"  Found {len(results.results)} drugs mentioning heart failure/cardiac:")
        for drug in results.results:
            print(f"    • {drug.product_name} ({len(drug.sections)} relevant sections)")


async def search_trials_example():
    """Example: Search clinical trials."""
    print("\n" + "="*60)
    print("🧬 CLINICAL TRIALS SEARCH EXAMPLE")
    print("="*60)
    
    # Example 1: Search by condition
    print("\n1. Search by condition (diabetes):")
    search_input = ClinicalTrialsSearchInput(
        kind=SearchKind.condition,
        query="diabetes",
        limit=3
    )
    results = await clinical_trials_search_async(DEFAULT_CONFIG, search_input)
    
    print(f"Status: {results.status}")
    if results.hits:
        for i, hit in enumerate(results.hits, 1):
            print(f"  {i}. {hit.nct_id} (Score: {hit.score:.3f})")
            # Show first line of summary
            first_line = hit.summary.split('\n')[0]
            print(f"     {first_line}")
    
    # Example 2: Combo search
    print("\n2. Combo search (diabetes + metformin):")
    search_input = ClinicalTrialsSearchInput(
        kind=SearchKind.combo,
        condition_query="diabetes",
        intervention_query="metformin",
        limit=3
    )
    results = await clinical_trials_search_async(DEFAULT_CONFIG, search_input)
    
    if results.hits:
        for i, hit in enumerate(results.hits, 1):
            print(f"  {i}. {hit.nct_id} (Score: {hit.score:.3f})")
            # Extract brief title from summary
            lines = hit.summary.split('\n')
            title_line = lines[0] if lines else "No title"
            print(f"     {title_line}")
    
    # Example 3: Auto search with detailed results
    print("\n3. Auto search with full details:")
    search_input = ClinicalTrialsSearchInput(
        kind=SearchKind.auto,
        query="breast cancer immunotherapy",
        limit=2,
        return_json=True
    )
    results = await clinical_trials_search_async(DEFAULT_CONFIG, search_input)
    
    if results.hits:
        for i, hit in enumerate(results.hits, 1):
            print(f"  {i}. {hit.nct_id}")
            if hit.study_json:
                metadata = hit.study_json.get('metadata', {})
                print(f"     Phase: {metadata.get('phase', 'N/A')}")
                print(f"     Status: {metadata.get('overall_status', 'N/A')}")
                print(f"     Enrollment: {metadata.get('enrollment', 'N/A')}")


def sql_examples():
    """Example SQL queries for database exploration."""
    print("\n" + "="*60)
    print("🗄️  SQL QUERY EXAMPLES")
    print("="*60)
    
    # Example 1: Basic counts
    print("\n1. Database overview:")
    queries = [
        "SELECT COUNT(*) as total_studies FROM ctgov_studies",
        "SELECT COUNT(*) as total_drugs FROM drugcentral_drugs", 
        "SELECT COUNT(*) as total_sections FROM sections",
        "SELECT COUNT(*) as total_dailymed_products FROM dailymed_products"
    ]
    
    for query in queries:
        results = execute_sql_query(query)
        if results:
            key = list(results[0].keys())[0]
            print(f"  {key.replace('_', ' ').title()}: {results[0][key]:,}")
    
    # Example 2: Clinical trials by phase
    print("\n2. Clinical trials by phase:")
    results = execute_sql_query("""
        SELECT phase, COUNT(*) as count 
        FROM ctgov_studies 
        WHERE phase IS NOT NULL 
        GROUP BY phase 
        ORDER BY count DESC
    """)
    for row in results:
        print(f"  {row['phase']}: {row['count']:,} trials")
    
    # Example 3: Most common conditions
    print("\n3. Most common conditions in trials:")
    results = execute_sql_query("""
        SELECT downcase_name as condition, COUNT(*) as trial_count
        FROM ctgov_conditions c
        JOIN ctgov_studies s ON c.nct_id = s.nct_id
        GROUP BY downcase_name
        ORDER BY trial_count DESC
        LIMIT 5
    """)
    for row in results:
        print(f"  {row['condition']}: {row['trial_count']:,} trials")
    
    # Example 4: Drug formulations
    print("\n4. Sample drug formulations:")
    results = execute_sql_query("""
        SELECT name, formula, smiles
        FROM drugcentral_drugs 
        WHERE formula IS NOT NULL AND smiles IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 3
    """)
    for row in results:
        smiles_preview = row['smiles'][:50] + "..." if len(row['smiles']) > 50 else row['smiles']
        print(f"  {row['name']}: {row['formula']}")
        print(f"    SMILES: {smiles_preview}")
    
    # Example 5: Search index status
    print("\n5. Search index status:")
    results = execute_sql_query("""
        SELECT schemaname, tablename, indexname, indexdef
        FROM pg_indexes 
        WHERE indexname LIKE '%gin%' OR indexname LIKE '%vector%'
        ORDER BY tablename
        LIMIT 5
    """)
    for row in results:
        print(f"  {row['tablename']}.{row['indexname']}")


def debug_search_functionality():
    """Debug and inspect search functionality."""
    print("\n" + "="*60)
    print("🔧 DEBUG & INSPECTION EXAMPLES")
    print("="*60)
    
    # Check vector columns
    print("\n1. Vector columns in database:")
    results = execute_sql_query("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE data_type = 'tsvector'
        ORDER BY table_name, column_name
    """)
    for row in results:
        print(f"  {row['table_name']}.{row['column_name']}")
    
    # Check GIN indexes
    print("\n2. GIN indexes for full-text search:")
    results = execute_sql_query("""
        SELECT schemaname, tablename, indexname
        FROM pg_indexes
        WHERE indexdef LIKE '%gin%'
        ORDER BY tablename
    """)
    for row in results:
        print(f"  {row['tablename']}: {row['indexname']}")
    
    # Sample search functionality
    print("\n3. Test tsvector search:")
    results = execute_sql_query("""
        SELECT name, ts_rank(name_vector, to_tsquery('english', 'aspirin')) as rank
        FROM drugcentral_drugs
        WHERE name_vector @@ to_tsquery('english', 'aspirin')
        ORDER BY rank DESC
        LIMIT 3
    """)
    for row in results:
        print(f"  {row['name']}: rank {row['rank']:.4f}")


async def main():
    """Run all examples."""
    print("🔍 BiomedAgent Search Examples")
    print("=" * 60)
    print("This script demonstrates various ways to search and query the biomedical database.")
    
    try:
        # Test database connection first
        print("\n🔌 Testing database connection...")
        test_results = execute_sql_query("SELECT 1 as test")
        if not test_results:
            print("❌ Cannot connect to database. Check your configuration.")
            return
        print("✅ Database connection successful!")
        
        # Run SQL examples
        sql_examples()
        
        # Run search examples
        await search_drugs_example()
        await search_trials_example()
        
        # Debug functionality
        debug_search_functionality()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("\nTo run individual examples:")
        print("  python examples/search_examples.py")
        print("\nTo use the CLI interface:")
        print("  biomedagent-db search-examples")
        print("  biomedagent-db search-drugs --help")
        print("  biomedagent-db search-trials --help")
        print("  biomedagent-db sql-query --help")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
