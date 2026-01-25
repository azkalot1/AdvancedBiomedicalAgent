#!/usr/bin/env python3
"""
BindingDB ingestion for PostgreSQL.

Downloads BindingDB TSV file and imports raw binding data into standalone BindingDB tables.
Does NOT match to clinical trials - that will be done by a separate matching script.

This is a self-contained module following the pattern of build_openfda.py:
- Tables created programmatically (no separate SQL file)
- Single entry point: ingest_bindingdb_full()
- Can be called from ingest_all_postgres.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path
import sys
import psycopg2
from psycopg2.extras import execute_batch
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Handle imports for both direct execution and module import
try:
    from .config import DatabaseConfig, get_connection
except ImportError:
    from config import DatabaseConfig, get_connection

# BindingDB download URL
BINDINGDB_TSV_URL = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202510_tsv.zip"


def _download_tsv(dest: Path) -> Path:
    """Download BindingDB TSV file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading BindingDB TSV from: {BINDINGDB_TSV_URL}")
    print(f"📁 Destination: {dest}")
    print()
    
    try:
        # Add user-agent header to avoid issues
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }
        
        with requests.get(BINDINGDB_TSV_URL, stream=True, timeout=300, headers=headers) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(dest, "wb") as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading BindingDB") as pbar:
                        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
        
        # Verify file size
        file_size = dest.stat().st_size
        if file_size < 100_000_000:  # Should be > 100 MB
            raise RuntimeError(f"Downloaded file is too small ({file_size} bytes). Download may have failed.")
        
        print(f"✅ Downloaded {file_size / 1e6:.1f} MB to {dest}")
        return dest
        
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        print()
        print("💡 Manual download:")
        print("   1. Visit: https://www.bindingdb.org/bind/downloads/")
        print("   2. Download: BindingDB_All_202510_tsv.zip")
        print(f"   3. Save to: {dest}")
        raise


def _extract_tsv(zip_path: Path, extract_dir: Path) -> Path:
    """Extract TSV file from ZIP archive."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find TSV file(s) in the archive
        tsv_files = [f for f in zf.namelist() if f.endswith('.tsv')]
        if not tsv_files:
            raise RuntimeError("No .tsv file found in the archive")
        
        tsv_file = tsv_files[0]  # Take the first TSV file
        print(f"📄 Found TSV file: {tsv_file}")
        
        # Extract with progress
        info = zf.getinfo(tsv_file)
        extracted_path = extract_dir / info.filename
        
        with zf.open(tsv_file) as source, open(extracted_path, 'wb') as target:
            with tqdm(total=info.file_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                while True:
                    chunk = source.read(1 << 20)  # 1 MB chunks
                    if not chunk:
                        break
                    target.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Extracted {extracted_path.stat().st_size / 1e9:.2f} GB to {extracted_path}")
        return extracted_path


def _create_tables(config: DatabaseConfig, force_recreate: bool = False) -> None:
    """Create standalone BindingDB tables programmatically (following OpenFDA pattern)."""
    if force_recreate:
        print("🗑️  Force recreating BindingDB tables...")
    else:
        print("🔧 Creating BindingDB tables...")
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            try:
                # Drop existing tables if force_recreate is True
                if force_recreate:
                    print("🗑️  Dropping existing BindingDB tables...")
                    cur.execute("DROP TABLE IF EXISTS bindingdb_activities CASCADE")
                    cur.execute("DROP TABLE IF EXISTS bindingdb_molecules CASCADE") 
                    cur.execute("DROP TABLE IF EXISTS bindingdb_targets CASCADE")
                    cur.execute("DROP VIEW IF EXISTS bindingdb_summary CASCADE")
                    cur.execute("DROP VIEW IF EXISTS bindingdb_human_targets CASCADE")
                    cur.execute("DROP VIEW IF EXISTS bindingdb_stats CASCADE")
                    print("✅ Dropped existing tables and views")
                
                # 1. BindingDB molecules table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bindingdb_molecules (
                        id SERIAL PRIMARY KEY,
                        
                        -- Identifiers
                        inchi TEXT,
                        inchi_key VARCHAR(27),
                        smiles TEXT,
                        pubchem_cid BIGINT,
                        chembl_id VARCHAR(50),
                        
                        -- Molecule info
                        ligand_name TEXT,
                        molecular_weight FLOAT,
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Indexes and constraints for molecules
                cur.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_bdb_mol_inchikey_unique 
                    ON bindingdb_molecules(inchi_key) WHERE inchi_key IS NOT NULL
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bdb_mol_cid 
                    ON bindingdb_molecules(pubchem_cid) WHERE pubchem_cid IS NOT NULL
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bdb_mol_chembl 
                    ON bindingdb_molecules(chembl_id) WHERE chembl_id IS NOT NULL
                """)
                
                # 2. BindingDB targets table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bindingdb_targets (
                        id SERIAL PRIMARY KEY,
                        
                        -- Target identification
                        target_name VARCHAR(500),
                        uniprot_id VARCHAR(20),
                        gene_symbol VARCHAR(50),
                        organism VARCHAR(200),
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Indexes and constraints for targets
                cur.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_bdb_tgt_uniprot_name_unique 
                    ON bindingdb_targets(uniprot_id, target_name) WHERE uniprot_id IS NOT NULL
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bdb_tgt_gene 
                    ON bindingdb_targets(gene_symbol) WHERE gene_symbol IS NOT NULL
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bdb_tgt_organism 
                    ON bindingdb_targets(organism)
                """)
                
                # 3. BindingDB activities table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bindingdb_activities (
                        id SERIAL PRIMARY KEY,
                        
                        -- Foreign keys
                        molecule_id INTEGER REFERENCES bindingdb_molecules(id) ON DELETE CASCADE,
                        target_id INTEGER REFERENCES bindingdb_targets(id) ON DELETE CASCADE,
                        
                        -- Binding measurements (in nM)
                        ki_nm NUMERIC,
                        ic50_nm NUMERIC,
                        kd_nm NUMERIC,
                        ec50_nm NUMERIC,
                        
                        -- Computed values
                        best_value_nm NUMERIC,
                        pchembl_value NUMERIC,
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        UNIQUE(molecule_id, target_id)
                    )
                """)
                
                # Indexes for activities
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bdb_act_molecule 
                    ON bindingdb_activities(molecule_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bdb_act_target 
                    ON bindingdb_activities(target_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bdb_act_potency 
                    ON bindingdb_activities(pchembl_value DESC NULLS LAST) 
                    WHERE pchembl_value IS NOT NULL
                """)
                
                # 4. Summary view
                cur.execute("""
                    CREATE OR REPLACE VIEW bindingdb_summary AS
                    SELECT 
                        m.id as molecule_id,
                        m.inchi_key,
                        m.pubchem_cid,
                        m.chembl_id,
                        m.ligand_name,
                        
                        t.id as target_id,
                        t.target_name,
                        t.gene_symbol,
                        t.uniprot_id,
                        t.organism,
                        
                        a.ki_nm,
                        a.ic50_nm,
                        a.kd_nm,
                        a.ec50_nm,
                        a.best_value_nm,
                        a.pchembl_value
                        
                    FROM bindingdb_molecules m
                    JOIN bindingdb_activities a ON m.id = a.molecule_id
                    JOIN bindingdb_targets t ON a.target_id = t.id
                    ORDER BY a.pchembl_value DESC NULLS LAST
                """)
                
                # 5. Human targets view
                cur.execute("""
                    CREATE OR REPLACE VIEW bindingdb_human_targets AS
                    SELECT * FROM bindingdb_summary
                    WHERE organism ILIKE '%Homo sapiens%'
                """)
                
                # 6. Statistics view
                cur.execute("""
                    CREATE OR REPLACE VIEW bindingdb_stats AS
                    SELECT 
                        (SELECT COUNT(*) FROM bindingdb_molecules) as total_molecules,
                        (SELECT COUNT(*) FROM bindingdb_molecules WHERE inchi_key IS NOT NULL) as molecules_with_inchikey,
                        (SELECT COUNT(*) FROM bindingdb_molecules WHERE pubchem_cid IS NOT NULL) as molecules_with_cid,
                        (SELECT COUNT(*) FROM bindingdb_targets) as total_targets,
                        (SELECT COUNT(*) FROM bindingdb_targets WHERE organism ILIKE '%Homo sapiens%') as human_targets,
                        (SELECT COUNT(*) FROM bindingdb_activities) as total_activities,
                        (SELECT COUNT(*) FROM bindingdb_activities WHERE pchembl_value > 6) as significant_activities
                """)
                
                # Add comments
                cur.execute("COMMENT ON TABLE bindingdb_molecules IS 'BindingDB molecules - raw data from TSV file'")
                cur.execute("COMMENT ON TABLE bindingdb_targets IS 'BindingDB targets - proteins from TSV file'")
                cur.execute("COMMENT ON TABLE bindingdb_activities IS 'BindingDB binding measurements - molecule-target pairs'")
                cur.execute("COMMENT ON VIEW bindingdb_summary IS 'BindingDB summary - all molecule-target pairs'")
                cur.execute("COMMENT ON VIEW bindingdb_human_targets IS 'BindingDB human targets - filtered to Homo sapiens'")
                cur.execute("COMMENT ON VIEW bindingdb_stats IS 'BindingDB statistics - record counts and coverage'")
                
                conn.commit()
                print("✅ Tables created successfully")
                
            except psycopg2.errors.DuplicateTable:
                conn.rollback()
                print("ℹ️  Tables already exist, checking schema...")
                
                # Check and update existing table schema if needed
                _update_table_schema(cur)
                conn.commit()
                print("✅ Schema updated if needed")
            except Exception as e:
                conn.rollback()
                print(f"❌ Error creating tables: {e}")
                raise


def _update_table_schema(cur) -> None:
    """Update existing table schema to handle longer text fields."""
    try:
        # Check if ligand_name column needs to be updated to TEXT
        cur.execute("""
            SELECT data_type, character_maximum_length 
            FROM information_schema.columns 
            WHERE table_name = 'bindingdb_molecules' 
            AND column_name = 'ligand_name'
        """)
        result = cur.fetchone()
        
        if result and result[0] == 'character varying' and result[1] < 2000:
            print("🔄 Updating ligand_name column to TEXT...")
            cur.execute("ALTER TABLE bindingdb_molecules ALTER COLUMN ligand_name TYPE TEXT")
            print("✅ Updated ligand_name to TEXT")
        
        # Check if chembl_id column needs to be updated
        cur.execute("""
            SELECT data_type, character_maximum_length 
            FROM information_schema.columns 
            WHERE table_name = 'bindingdb_molecules' 
            AND column_name = 'chembl_id'
        """)
        result = cur.fetchone()
        
        if result and result[0] == 'character varying' and result[1] < 50:
            print("🔄 Updating chembl_id column to VARCHAR(50)...")
            cur.execute("ALTER TABLE bindingdb_molecules ALTER COLUMN chembl_id TYPE VARCHAR(50)")
            print("✅ Updated chembl_id to VARCHAR(50)")
            
    except Exception as e:
        print(f"⚠️  Schema update warning: {e}")


def _load_tsv(tsv_path: Path, limit: Optional[int]) -> pd.DataFrame:
    """Load BindingDB TSV file with only necessary columns."""
    usecols = [
        'Ligand InChI',
        'Ligand InChI Key',
        'Ligand SMILES', 
        'BindingDB Ligand Name',
        'PubChem CID',
        'ChEMBL ID of Ligand',
        'Target Name',
        'UniProt (SwissProt) Primary ID of Target Chain 1',
        'UniProt (SwissProt) Entry Name of Target Chain 1',
        'Target Source Organism According to Curator or DataSource',
        'Ki (nM)',
        'IC50 (nM)',
        'Kd (nM)',
        'EC50 (nM)'
    ]
    
    kwargs = {
        'sep': '\t',
        'low_memory': False,
        'usecols': lambda x: x in usecols,
    }
    
    if limit:
        kwargs['nrows'] = limit
    
    df = pd.read_csv(tsv_path, **kwargs)
    df.columns = df.columns.str.strip()
    
    return df


def _process_tsv_in_batches(tsv_path: Path, config: DatabaseConfig, 
                           batch_size: int = 10000, limit: Optional[int] = None, 
                           human_only: bool = True):
    """Process BindingDB TSV file in batches to avoid memory issues."""
    usecols = [
        'Ligand InChI',
        'Ligand InChI Key',
        'Ligand SMILES', 
        'BindingDB Ligand Name',
        'PubChem CID',
        'ChEMBL ID of Ligand',
        'Target Name',
        'UniProt (SwissProt) Primary ID of Target Chain 1',
        'UniProt (SwissProt) Entry Name of Target Chain 1',
        'Target Source Organism According to Curator or DataSource',
        'Ki (nM)',
        'IC50 (nM)',
        'Kd (nM)',
        'EC50 (nM)'
    ]
    
    print(f"📖 Processing TSV file in batches of {batch_size:,} records...")
    print(f"📁 Source: {tsv_path}")
    if human_only:
        print("🔬 Filter: Human targets only (Homo sapiens)")
    print()
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    # Initialize statistics
    stats = {
        'total_processed': 0,
        'molecules_added': 0,
        'targets_added': 0,
        'activities_added': 0,
        'skipped_no_data': 0,
        'skipped_no_structure': 0,
        'batches_processed': 0
    }
    
    # Caches for deduplication (persist across batches)
    molecule_cache = {}  # inchi_key -> molecule_id
    target_cache = {}    # (uniprot_id, target_name) -> target_id
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            # Process file in chunks
            chunk_iter = pd.read_csv(
                tsv_path,
                sep='\t',
                usecols=lambda x: x in usecols,
                chunksize=batch_size,
                low_memory=False
            )
            
            for batch_num, df_chunk in enumerate(chunk_iter, 1):
                if limit and stats['total_processed'] >= limit:
                    break
                
                # Clean column names
                df_chunk.columns = df_chunk.columns.str.strip()
                
                # Filter to human targets if requested
                if human_only:
                    organism_col = 'Target Source Organism According to Curator or DataSource'
                    if organism_col in df_chunk.columns:
                        df_chunk = df_chunk[df_chunk[organism_col].str.contains('Homo sapiens', na=False, case=False)]
                
                # Process this batch
                batch_stats = _process_batch(cur, df_chunk, molecule_cache, target_cache)
                
                # Update overall stats
                for key in batch_stats:
                    stats[key] += batch_stats[key]
                stats['batches_processed'] += 1
                stats['total_processed'] += len(df_chunk)
                
                # Commit after each batch
                conn.commit()
                
                # Progress update
                print(f"📊 Batch {batch_num}: {len(df_chunk):,} records processed, "
                      f"{batch_stats['activities_added']:,} activities added")
                
                if limit and stats['total_processed'] >= limit:
                    break
    
    return stats


def _process_batch(cur, df_batch: pd.DataFrame, molecule_cache: dict, target_cache: dict) -> dict:
    """Process a single batch of TSV data."""
    batch_stats = {
        'molecules_added': 0,
        'targets_added': 0,
        'activities_added': 0,
        'skipped_no_data': 0,
        'skipped_no_structure': 0
    }
    
    for idx, row in df_batch.iterrows():
        # Get or create molecule
        molecule_id = _upsert_molecule(cur, row, molecule_cache)
        if not molecule_id:
            batch_stats['skipped_no_structure'] += 1
            continue
        
        # Get or create target
        target_id = _upsert_target(cur, row, target_cache)
        if not target_id:
            continue
        
        # Parse binding values
        ki_nm = _parse_binding_value(row.get('Ki (nM)'))
        ic50_nm = _parse_binding_value(row.get('IC50 (nM)'))
        kd_nm = _parse_binding_value(row.get('Kd (nM)'))
        ec50_nm = _parse_binding_value(row.get('EC50 (nM)'))
        
        # Skip if no binding data
        if all(v is None for v in [ki_nm, ic50_nm, kd_nm, ec50_nm]):
            batch_stats['skipped_no_data'] += 1
            continue
        
        # Calculate best value and pChEMBL
        best_value = min([v for v in [ki_nm, ic50_nm, kd_nm] if v is not None], default=None)
        pchembl = -np.log10(best_value / 1e9) if best_value else None
        
        # Insert activity
        if _upsert_activity(cur, molecule_id, target_id, ki_nm, ic50_nm, kd_nm, ec50_nm, best_value, pchembl):
            batch_stats['activities_added'] += 1
    
    return batch_stats


def _parse_binding_value(value) -> Optional[float]:
    """Parse binding value, handling various formats like '>10000' or '<1'."""
    if pd.isna(value):
        return None
    
    if isinstance(value, str):
        value = value.strip().replace('>', '').replace('<', '').replace(',', '')
        try:
            return float(value)
        except:
            return None
    
    try:
        return float(value)
    except:
        return None


def _import_data(config: DatabaseConfig, tsv_path: Path, limit: int | None = None, human_only: bool = True, batch_size: int = 10000) -> None:
    """Import BindingDB TSV data into PostgreSQL using batch processing to avoid memory issues."""
    print()
    print("=" * 80)
    print("Importing BindingDB Data (Batch Processing)")
    print("=" * 80)
    print(f"Source: {tsv_path}")
    if human_only:
        print("Filter: Human targets only (Homo sapiens)")
    print()
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    # Use batch processing to avoid memory issues
    stats = _process_tsv_in_batches(tsv_path, config, batch_size=batch_size, limit=limit, human_only=human_only)
    
    # Print statistics
    print()
    print("=" * 80)
    print("Import Complete")
    print("=" * 80)
    print()
    print(f"Batches processed:        {stats['batches_processed']:,}")
    print(f"Total records:        {stats['total_processed']:,}")
    print(f"Activities imported:  {stats['activities_added']:,}")
    print(f"Skipped (no structure): {stats['skipped_no_structure']:,}")
    print(f"Skipped (no data):      {stats['skipped_no_data']:,}")
    print()


def _upsert_molecule(cur, row: pd.Series, cache: dict) -> Optional[int]:
    """Get or create molecule, return molecule_id."""
    inchi_key = row.get('Ligand InChI Key')
    
    # Must have InChI Key for deduplication
    if pd.isna(inchi_key) or not inchi_key.strip():
        return None
    
    inchi_key = inchi_key.strip()
    
    # Check cache
    if inchi_key in cache:
        return cache[inchi_key]
    
    # Try to find existing
    cur.execute("SELECT id FROM bindingdb_molecules WHERE inchi_key = %s", (inchi_key,))
    row_result = cur.fetchone()
    if row_result:
        mol_id = row_result[0]
        cache[inchi_key] = mol_id
        return mol_id
    
    # Insert new
    inchi = row.get('Ligand InChI')
    smiles = row.get('Ligand SMILES')
    ligand_name = row.get('BindingDB Ligand Name')
    
    # Debug: Check if we're getting concatenated names
    if pd.notna(ligand_name) and len(str(ligand_name)) > 200:
        # Check if this looks like concatenated names (contains multiple ::)
        if '::' in str(ligand_name):
            # Extract just the first name before the first ::
            first_name = str(ligand_name).split('::')[0]
            if len(first_name) < len(str(ligand_name)):
                ligand_name = first_name
    
    # Truncate ligand name if it's extremely long (keep first 2000 chars)
    if pd.notna(ligand_name):
        ligand_name = str(ligand_name)
        if len(ligand_name) > 2000:
            ligand_name = ligand_name[:2000]
    
    # Parse IDs
    pubchem_cid = row.get('PubChem CID')
    if pd.notna(pubchem_cid):
        try:
            pubchem_cid = int(float(pubchem_cid))
        except:
            pubchem_cid = None
    else:
        pubchem_cid = None
    
    chembl_id = row.get('ChEMBL ID of Ligand')
    if pd.isna(chembl_id):
        chembl_id = None
    else:
        # Truncate chembl_id if it's too long for VARCHAR(50)
        chembl_id = str(chembl_id)
        if len(chembl_id) > 50:
            print(f"Truncating chembl_id from {len(chembl_id)} to 50 chars: {chembl_id[:50]}")
            chembl_id = chembl_id[:50]
    
    try:
        cur.execute("""
            INSERT INTO bindingdb_molecules (
                inchi, inchi_key, smiles, pubchem_cid, chembl_id, ligand_name
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            str(inchi) if pd.notna(inchi) else None,
            str(inchi_key),
            str(smiles) if pd.notna(smiles) else None,
            int(pubchem_cid) if pubchem_cid is not None else None,
            str(chembl_id) if chembl_id is not None else None,
            str(ligand_name) if pd.notna(ligand_name) else None
        ))
        
        mol_id = cur.fetchone()[0]
        cache[inchi_key] = mol_id
        return mol_id
        
    except psycopg2.errors.UniqueViolation:
        # Race condition - try to fetch again
        cur.execute("SELECT id FROM bindingdb_molecules WHERE inchi_key = %s", (inchi_key,))
        row_result = cur.fetchone()
        if row_result:
            mol_id = row_result[0]
            cache[inchi_key] = mol_id
            return mol_id
        return None


def _upsert_target(cur, row: pd.Series, cache: dict) -> Optional[int]:
    """Get or create target, return target_id."""
    target_name = row.get('Target Name')
    uniprot_id = row.get('UniProt (SwissProt) Primary ID of Target Chain 1')
    entry_name = row.get('UniProt (SwissProt) Entry Name of Target Chain 1')
    organism = row.get('Target Source Organism According to Curator or DataSource')
    
    # Must have target name or uniprot ID
    if (pd.isna(target_name) or not target_name.strip()) and (pd.isna(uniprot_id) or not uniprot_id.strip()):
        return None
    
    target_name = target_name.strip() if pd.notna(target_name) else None
    uniprot_id = uniprot_id.strip() if pd.notna(uniprot_id) else None
    
    # Cache key
    cache_key = (uniprot_id, target_name)
    if cache_key in cache:
        return cache[cache_key]
    
    # Try to find existing
    if uniprot_id:
        cur.execute("""
            SELECT id FROM bindingdb_targets 
            WHERE uniprot_id = %s AND target_name = %s
        """, (uniprot_id, target_name))
    else:
        cur.execute("""
            SELECT id FROM bindingdb_targets 
            WHERE uniprot_id IS NULL AND target_name = %s
        """, (target_name,))
    
    row_result = cur.fetchone()
    if row_result:
        target_id = row_result[0]
        cache[cache_key] = target_id
        return target_id
    
    # Extract gene symbol from entry name
    gene_symbol = None
    if pd.notna(entry_name):
        gene_symbol = entry_name.split('_')[0] if '_' in entry_name else None
    
    # Insert new
    try:
        cur.execute("""
            INSERT INTO bindingdb_targets (
                target_name, uniprot_id, gene_symbol, organism
            ) VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (
            str(target_name) if target_name is not None else None,
            str(uniprot_id) if uniprot_id is not None else None,
            str(gene_symbol) if gene_symbol is not None else None,
            str(organism) if pd.notna(organism) else None
        ))
        
        target_id = cur.fetchone()[0]
        cache[cache_key] = target_id
        return target_id
        
    except psycopg2.errors.UniqueViolation:
        # Race condition - try to fetch again
        if uniprot_id:
            cur.execute("""
                SELECT id FROM bindingdb_targets 
                WHERE uniprot_id = %s AND target_name = %s
            """, (uniprot_id, target_name))
        else:
            cur.execute("""
                SELECT id FROM bindingdb_targets 
                WHERE uniprot_id IS NULL AND target_name = %s
            """, (target_name,))
        
        row_result = cur.fetchone()
        if row_result:
            target_id = row_result[0]
            cache[cache_key] = target_id
            return target_id
        return None


def _upsert_activity(cur, molecule_id: int, target_id: int, 
                     ki_nm: Optional[float], ic50_nm: Optional[float], 
                     kd_nm: Optional[float], ec50_nm: Optional[float],
                     best_value: Optional[float], pchembl: Optional[float]) -> bool:
    """Insert or update activity measurement."""
    try:
        cur.execute("""
            INSERT INTO bindingdb_activities (
                molecule_id, target_id, 
                ki_nm, ic50_nm, kd_nm, ec50_nm,
                best_value_nm, pchembl_value
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (molecule_id, target_id) DO UPDATE
            SET 
                ki_nm = LEAST(COALESCE(bindingdb_activities.ki_nm, 999999), COALESCE(EXCLUDED.ki_nm, 999999)),
                ic50_nm = LEAST(COALESCE(bindingdb_activities.ic50_nm, 999999), COALESCE(EXCLUDED.ic50_nm, 999999)),
                kd_nm = LEAST(COALESCE(bindingdb_activities.kd_nm, 999999), COALESCE(EXCLUDED.kd_nm, 999999)),
                ec50_nm = LEAST(COALESCE(bindingdb_activities.ec50_nm, 999999), COALESCE(EXCLUDED.ec50_nm, 999999)),
                best_value_nm = LEAST(COALESCE(bindingdb_activities.best_value_nm, 999999), COALESCE(EXCLUDED.best_value_nm, 999999)),
                pchembl_value = GREATEST(COALESCE(bindingdb_activities.pchembl_value, 0), COALESCE(EXCLUDED.pchembl_value, 0))
        """, (
            int(molecule_id), 
            int(target_id), 
            float(ki_nm) if ki_nm is not None else None,
            float(ic50_nm) if ic50_nm is not None else None,
            float(kd_nm) if kd_nm is not None else None,
            float(ec50_nm) if ec50_nm is not None else None,
            float(best_value) if best_value is not None else None,
            float(pchembl) if pchembl is not None else None
        ))
        return True
    except Exception as e:
        print(f"\n❌ Error inserting activity: {e}")
        return False


def _verify_import(config: DatabaseConfig) -> None:
    """Verify BindingDB import results."""
    print()
    print("=" * 80)
    print("Verification Results")
    print("=" * 80)
    print()
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            # Get statistics from view
            cur.execute("SELECT * FROM bindingdb_stats")
            stats = cur.fetchone()
            
            print(f"📊 Unique molecules:           {stats[0]:,}")
            print(f"📊 - with InChIKey:            {stats[1]:,} ({100*stats[1]/max(stats[0],1):.1f}%)")
            print(f"📊 - with PubChem CID:         {stats[2]:,} ({100*stats[2]/max(stats[0],1):.1f}%)")
            print(f"📊 Unique targets:             {stats[3]:,}")
            print(f"📊 - human targets:            {stats[4]:,} ({100*stats[4]/max(stats[3],1):.1f}%)")
            print(f"📊 Total activities:           {stats[5]:,}")
            print(f"📊 - significant (pChEMBL>6):  {stats[6]:,} ({100*stats[6]/max(stats[5],1):.1f}%)")
            
            print()
            
            # Show top molecules by number of targets
            print("Top molecules by target count:")
            print("-" * 80)
            cur.execute("""
                SELECT 
                    m.ligand_name,
                    m.pubchem_cid,
                    COUNT(DISTINCT t.id) as num_targets,
                    STRING_AGG(DISTINCT t.gene_symbol, ', ' ORDER BY t.gene_symbol) FILTER (WHERE t.gene_symbol IS NOT NULL) as genes
                FROM bindingdb_molecules m
                JOIN bindingdb_activities a ON m.id = a.molecule_id
                JOIN bindingdb_targets t ON a.target_id = t.id
                WHERE a.pchembl_value > 6
                  AND t.organism ILIKE '%Homo sapiens%'
                GROUP BY m.id, m.ligand_name, m.pubchem_cid
                HAVING COUNT(DISTINCT t.id) > 1
                ORDER BY num_targets DESC
                LIMIT 5
            """)
            
            for ligand_name, cid, num_targets, genes in cur.fetchall():
                name_display = ligand_name if ligand_name else f"CID:{cid}"
                print(f"  {name_display}")
                print(f"    └─ {num_targets} targets: {genes}")
            
            print()


def ingest_bindingdb_full(config: DatabaseConfig, raw_dir: Path, limit: int | None = None, human_only: bool = True, batch_size: int = 10000, force_recreate: bool = False) -> None:
    """
    Full BindingDB ingestion workflow - single entry point for ingest_all_postgres.py.
    
    This function follows the OpenFDA pattern:
    - Creates tables programmatically (no SQL file needed)
    - Downloads and extracts data if needed
    - Imports data into PostgreSQL
    - Verifies and reports statistics
    
    Args:
        config: Database configuration
        raw_dir: Base raw data directory (will create bindingdb subdirectory)
        limit: Limit number of records to import (for testing)
        human_only: Filter to human targets only (default: True)
    """
    print()
    print("=" * 80)
    print("BindingDB Ingestion")
    print("=" * 80)
    print()
    
    # Create bindingdb subdirectory in raw_dir (following OpenFDA/CTGov pattern)
    bindingdb_dir = Path(raw_dir) / "bindingdb"
    zip_path = bindingdb_dir / "BindingDB_All_202510_tsv.zip"
    tsv_path = bindingdb_dir / "BindingDB_All.tsv"
    
    # Step 1: Download (if needed)
    if not zip_path.exists():
        print("Step 1/4: Download")
        print("-" * 80)
        _download_tsv(zip_path)
        print()
    else:
        print(f"ℹ️  Using existing download: {zip_path}")
        print()
    
    # Step 2: Extract (if needed)
    if not tsv_path.exists():
        print("Step 2/4: Extract")
        print("-" * 80)
        _extract_tsv(zip_path, bindingdb_dir)
        print()
    else:
        print(f"ℹ️  Using existing file: {tsv_path}")
        print()
    
    # Step 3: Create tables
    print("Step 3/4: Create Tables")
    print("-" * 80)
    _create_tables(config, force_recreate=force_recreate)
    print()
    
    # Step 4: Import data
    print("Step 4/4: Import Data")
    print("-" * 80)
    _import_data(config, tsv_path, limit=limit, human_only=human_only, batch_size=batch_size)
    print()
    
    # Verify results
    _verify_import(config)
    
    print()
    print("=" * 80)
    print("✅ BindingDB ingestion complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  - View data: SELECT * FROM bindingdb_human_targets LIMIT 10;")
    print("  - Check stats: SELECT * FROM bindingdb_stats;")
    print("  - Match to interventions: python -m bioagent.data.ingest.match_bindingdb_to_interventions")
    print()


def verify_tables(config: DatabaseConfig) -> None:
    """Verify BindingDB tables exist and have correct structure."""
    print("🔍 Verifying BindingDB tables...")
    print()
    
    required_tables = [
        'bindingdb_molecules',
        'bindingdb_targets',
        'bindingdb_activities'
    ]
    
    required_views = [
        'bindingdb_summary',
        'bindingdb_human_targets',
        'bindingdb_stats'
    ]
    
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            # Check tables
            print("Tables:")
            for table in required_tables:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                """, (table,))
                exists = cur.fetchone()[0] > 0
                
                if exists:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    print(f"  ✅ {table}: {count:,} records")
                else:
                    print(f"  ❌ {table}: NOT FOUND")
            
            print()
            
            # Check views
            print("Views:")
            for view in required_views:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.views 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                """, (view,))
                exists = cur.fetchone()[0] > 0
                print(f"  {'✅' if exists else '❌'} {view}")
            
            print()
            
            # Check indexes
            print("Indexes:")
            cur.execute("""
                SELECT 
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename IN ('bindingdb_molecules', 'bindingdb_targets', 'bindingdb_activities')
                ORDER BY tablename, indexname
            """)
            
            indexes = cur.fetchall()
            for table, index, definition in indexes:
                print(f"  ✅ {table}.{index}")
            
            if not indexes:
                print("  ⚠️  No indexes found")
            
            print()


if __name__ == "__main__":
    """CLI for BindingDB ingestion (for direct execution)."""
    # Import default config
    try:
        from .config import DEFAULT_CONFIG
    except ImportError:
        from config import DEFAULT_CONFIG
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BindingDB ingestion for PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_bindingdb.py raw/                    # Full ingestion
  python build_bindingdb.py raw/ --limit 10000      # Test with 10K records
  python build_bindingdb.py raw/ --all-organisms    # Include all organisms
  python build_bindingdb.py raw/ --batch-size 5000  # Smaller batches for low memory
  python build_bindingdb.py raw/ --force-recreate   # Drop and recreate tables
  python build_bindingdb.py raw/ --verify-only      # Just verify existing data
        """
    )
    
    parser.add_argument("raw_dir", type=Path, help="Raw data directory")
    parser.add_argument("--limit", type=int, help="Limit number of records to import")
    parser.add_argument("--all-organisms", action="store_true", help="Include all organisms (default: human only)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for processing (default: 10000)")
    parser.add_argument("--force-recreate", action="store_true", help="Drop and recreate tables (WARNING: deletes all existing data)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing tables")
    
    args = parser.parse_args()
    
    if args.verify_only:
        print("🔍 Verifying BindingDB installation...")
        print()
        verify_tables(DEFAULT_CONFIG)
        print()
        _verify_import(DEFAULT_CONFIG)
    else:
        if args.force_recreate:
            print("⚠️  WARNING: --force-recreate will DELETE ALL existing BindingDB data!")
            print("This will drop and recreate all BindingDB tables and views.")
            response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Operation cancelled.")
                sys.exit(1)
            print("✅ Proceeding with force recreate...")
            print()
        
        print(f"🚀 Starting BindingDB ingestion to {args.raw_dir}")
        if args.limit:
            print(f"ℹ️  Limiting to {args.limit:,} records")
        if args.all_organisms:
            print(f"ℹ️  Including all organisms (not just human)")
        if args.force_recreate:
            print(f"🗑️  Force recreating tables (deleting existing data)")
        
        ingest_bindingdb_full(
            DEFAULT_CONFIG,
            args.raw_dir,
            limit=args.limit,
            human_only=not args.all_organisms,
            batch_size=args.batch_size,
            force_recreate=args.force_recreate
        )

