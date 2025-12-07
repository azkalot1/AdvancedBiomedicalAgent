# Mapping ClinicalTrials.gov Interventions to Molecular Identifiers

## Summary

**ClinicalTrials.gov does NOT store molecular identifiers** like SMILES, InChI, PubChem CID, or ChEMBL ID.

To enable molecule-protein binding annotation, you need to:
1. Extract intervention names from CTG database
2. Map names to external molecular databases
3. Retrieve molecular structures and identifiers
4. Link to protein binding databases

---

## Available Fields in ClinicalTrials.gov

### 1. `ctgov_interventions` Table

**Best starting point for drug names**

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer | Primary key |
| `nct_id` | varchar | Study identifier |
| `intervention_type` | varchar | DRUG, BIOLOGICAL, DEVICE, PROCEDURE, etc. |
| `name` | varchar | **Primary intervention name** |
| `description` | text | Details about the intervention |

**Example drugs:**
- Cyclophosphamide
- Lenalidomide
- Cisplatin
- Lysergic Acid Diethylamide

### 2. `ctgov_intervention_other_names` Table

**Critical for mapping - contains aliases, brand names, chemical names**

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer | Primary key |
| `nct_id` | varchar | Study identifier |
| `intervention_id` | integer | FK to ctgov_interventions |
| `name` | varchar | **Alias/brand name/chemical name** |

**Example** - Cyclophosphamide has aliases:
- (-)-Cyclophosphamide
- 2H-1,3,2-Oxazaphosphorine, 2-[bis(2-chloroethyl)amino]tetrahydro-, 2-oxide, monohydrate
- CP monohydrate
- CTX
- CYCLO-cell

**Example** - Cisplatin has aliases:
- CDDP
- Platinol
- Platinol-AQ
- cis-diamminedichloroplatinum(II)

### 3. `ctgov_browse_interventions` Table

**MeSH standardized terms - useful for drug class annotation**

| Column | Type | Description |
|--------|------|-------------|
| `nct_id` | varchar | Study identifier |
| `mesh_term` | varchar | **Standardized MeSH term** |
| `mesh_type` | varchar | mesh-list, mesh-ancestor |

**Example MeSH terms:**
- Carboplatin
- Heterocyclic Compounds, Fused-Ring
- Steroids
- Organic Chemicals

---

## External Databases for Molecular Mapping

### 1. PubChem (NIH) - FREE, BEST FOR GENERAL USE

**API**: `https://pubchem.ncbi.nlm.nih.gov/rest/pug`

**Why PubChem?**
- ✅ Free, no authentication required
- ✅ Comprehensive coverage (~111 million compounds)
- ✅ REST API easy to use
- ✅ Returns SMILES, InChI, molecular formula, etc.
- ✅ Links to bioactivity data

**Example API Calls:**

```bash
# Search by name
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/JSON"

# Search by synonym (brand name)
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/Platinol/JSON"

# Get properties by CID
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/CanonicalSMILES,MolecularFormula,MolecularWeight/JSON"

# Get bioassay data
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/assaysummary/JSON"
```

**Response Example:**
```json
{
  "PC_Compounds": [{
    "id": {
      "id": {"cid": 2244}
    },
    "props": [
      {
        "urn": {"label": "SMILES", "name": "Canonical"},
        "value": {"sval": "CC(=O)OC1=CC=CC=C1C(=O)O"}
      },
      {
        "urn": {"label": "Molecular Formula"},
        "value": {"sval": "C9H8O4"}
      }
    ]
  }]
}
```

### 2. ChEMBL (EMBL-EBI) - FREE, BEST FOR BIOACTIVITY

**API**: `https://www.ebi.ac.uk/chembl/api/data`

**Why ChEMBL?**
- ✅ Free, no authentication required
- ✅ Focus on bioactive molecules
- ✅ Extensive protein target data
- ✅ Drug-target binding data
- ✅ IC50, Ki, EC50 values

**Example API Calls:**

```bash
# Search by name
curl "https://www.ebi.ac.uk/chembl/api/data/molecule/search?q=aspirin&format=json"

# Get molecule by ChEMBL ID
curl "https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json"

# Get targets for a molecule
curl "https://www.ebi.ac.uk/chembl/api/data/activity?molecule_chembl_id=CHEMBL25&format=json"

# Get bioactivities
curl "https://www.ebi.ac.uk/chembl/api/data/activity?molecule_chembl_id=CHEMBL25&target_type=PROTEIN&format=json"
```

**Response Example:**
```json
{
  "molecule_chembl_id": "CHEMBL25",
  "molecule_structures": {
    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "standard_inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
    "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
  },
  "molecule_properties": {
    "molecular_weight": 180.16,
    "alogp": 1.23
  }
}
```

### 3. UniChem (EMBL-EBI) - FREE, CROSS-REFERENCE TOOL

**API**: `https://www.ebi.ac.uk/unichem/rest`

**Why UniChem?**
- ✅ Cross-references between 40+ chemical databases
- ✅ Map PubChem CID ↔ ChEMBL ID ↔ DrugBank ID
- ✅ Find canonical identifier

**Example:**
```bash
# Get ChEMBL ID from PubChem CID
curl "https://www.ebi.ac.uk/unichem/rest/src_compound_id/2244/22/1"
# 22 = PubChem, 1 = ChEMBL
```

### 4. DrugBank - COMPREHENSIVE BUT REQUIRES LICENSE

**API**: Requires commercial license for full access

**Why DrugBank?**
- Most comprehensive drug database
- FDA approval status
- Drug-drug interactions
- Detailed pharmacology
- ⚠️ Requires paid license for API

---

## Recommended Workflow

### Step 1: Extract Intervention Names from CTG

```sql
-- Get all drug interventions with their aliases
SELECT 
    i.id as intervention_id,
    i.nct_id,
    i.name as primary_name,
    i.intervention_type,
    array_agg(DISTINCT ion.name) FILTER (WHERE ion.name IS NOT NULL) as aliases
FROM ctgov_interventions i
LEFT JOIN ctgov_intervention_other_names ion ON i.id = ion.intervention_id
WHERE i.intervention_type IN ('DRUG', 'BIOLOGICAL')
GROUP BY i.id, i.nct_id, i.name, i.intervention_type;
```

### Step 2: Query PubChem API

```python
import requests
import time

def get_pubchem_cid(drug_name):
    """Get PubChem CID for a drug name."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/cids/JSON"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['IdentifierList']['CID'][0] if 'IdentifierList' in data else None
    except:
        return None
    
    return None

def get_pubchem_properties(cid):
    """Get molecular properties for a PubChem CID."""
    properties = [
        'CanonicalSMILES',
        'IsomericSMILES',
        'InChI',
        'InChIKey',
        'IUPACName',
        'MolecularFormula',
        'MolecularWeight'
    ]
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{','.join(properties)}/JSON"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['PropertyTable']['Properties'][0] if 'PropertyTable' in data else None
    except:
        return None
    
    return None

# Example usage
drug_name = "Cisplatin"
cid = get_pubchem_cid(drug_name)
if cid:
    properties = get_pubchem_properties(cid)
    print(f"CID: {cid}")
    print(f"SMILES: {properties.get('CanonicalSMILES')}")
    print(f"InChI Key: {properties.get('InChIKey')}")
    print(f"Formula: {properties.get('MolecularFormula')}")
```

### Step 3: Try Aliases if Primary Name Fails

```python
def map_intervention_to_pubchem(primary_name, aliases):
    """Try primary name, then aliases."""
    
    # Try primary name first
    cid = get_pubchem_cid(primary_name)
    if cid:
        return cid, primary_name
    
    # Try each alias
    for alias in aliases:
        cid = get_pubchem_cid(alias)
        if cid:
            return cid, alias
        time.sleep(0.2)  # Rate limiting
    
    return None, None
```

### Step 4: Get ChEMBL ID and Bioactivity

```python
def get_chembl_id(drug_name):
    """Get ChEMBL ID for a drug name."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug_name}&format=json"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'molecules' in data and len(data['molecules']) > 0:
                return data['molecules'][0]['molecule_chembl_id']
    except:
        return None
    
    return None

def get_protein_targets(chembl_id):
    """Get protein targets for a ChEMBL molecule."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity?molecule_chembl_id={chembl_id}&target_type=SINGLE%20PROTEIN&format=json"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            targets = []
            for activity in data.get('activities', []):
                targets.append({
                    'target_chembl_id': activity.get('target_chembl_id'),
                    'target_pref_name': activity.get('target_pref_name'),
                    'type': activity.get('standard_type'),  # IC50, Ki, etc.
                    'value': activity.get('standard_value'),
                    'units': activity.get('standard_units')
                })
            return targets
    except:
        return []
    
    return []
```

### Step 5: Store Mappings in Database

```sql
-- Create mapping table
CREATE TABLE IF NOT EXISTS intervention_molecular_mappings (
    id SERIAL PRIMARY KEY,
    intervention_id INTEGER REFERENCES ctgov_interventions(id),
    matched_name VARCHAR,  -- Which name/alias matched
    pubchem_cid BIGINT,
    chembl_id VARCHAR,
    canonical_smiles TEXT,
    isomeric_smiles TEXT,
    inchi TEXT,
    inchi_key VARCHAR,
    iupac_name TEXT,
    molecular_formula VARCHAR,
    molecular_weight FLOAT,
    mapping_source VARCHAR,  -- 'pubchem', 'chembl', 'manual'
    mapping_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(intervention_id, pubchem_cid)
);

-- Create protein targets table
CREATE TABLE IF NOT EXISTS intervention_protein_targets (
    id SERIAL PRIMARY KEY,
    intervention_id INTEGER REFERENCES ctgov_interventions(id),
    chembl_id VARCHAR,
    target_chembl_id VARCHAR,
    target_name VARCHAR,
    uniprot_id VARCHAR,
    activity_type VARCHAR,  -- IC50, Ki, EC50, etc.
    activity_value FLOAT,
    activity_units VARCHAR,
    confidence_score INTEGER,
    data_source VARCHAR DEFAULT 'chembl',
    import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert mapping
INSERT INTO intervention_molecular_mappings (
    intervention_id,
    matched_name,
    pubchem_cid,
    canonical_smiles,
    inchi_key,
    molecular_formula,
    molecular_weight,
    mapping_source
) VALUES (
    12345,
    'Cisplatin',
    5702208,
    'N.N.Cl[Pt]Cl',
    'LXZZYRPGZAFOLE-UHFFFAOYSA-L',
    'Cl2H6N2Pt',
    300.05,
    'pubchem'
);
```

---

## Complete Python Module Example

```python
"""
molecular_mapper.py - Map clinical trial interventions to molecular identifiers
"""
import requests
import psycopg2
from typing import Optional, Dict, List, Tuple
import time
from dataclasses import dataclass

@dataclass
class MolecularMapping:
    pubchem_cid: Optional[int]
    chembl_id: Optional[str]
    canonical_smiles: Optional[str]
    inchi_key: Optional[str]
    molecular_formula: Optional[str]
    molecular_weight: Optional[float]
    matched_name: str

class MolecularMapper:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.session = requests.Session()
    
    def map_intervention(self, intervention_id: int) -> Optional[MolecularMapping]:
        """Map a single intervention to molecular identifiers."""
        
        # Get intervention names
        cur = self.conn.cursor()
        cur.execute("""
            SELECT 
                i.name as primary_name,
                array_agg(DISTINCT ion.name) FILTER (WHERE ion.name IS NOT NULL) as aliases
            FROM ctgov_interventions i
            LEFT JOIN ctgov_intervention_other_names ion ON i.id = ion.intervention_id
            WHERE i.id = %s
            GROUP BY i.id, i.name
        """, (intervention_id,))
        
        row = cur.fetchone()
        if not row:
            return None
        
        primary_name, aliases = row
        all_names = [primary_name] + (aliases or [])
        
        # Try mapping with each name
        for name in all_names:
            mapping = self._try_pubchem(name)
            if mapping:
                mapping.matched_name = name
                return mapping
            time.sleep(0.2)  # Rate limiting
        
        return None
    
    def _try_pubchem(self, drug_name: str) -> Optional[MolecularMapping]:
        """Try to map via PubChem."""
        
        # Get CID
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/cids/JSON"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            cid = data['IdentifierList']['CID'][0]
        except:
            return None
        
        # Get properties
        properties = [
            'CanonicalSMILES',
            'InChIKey',
            'MolecularFormula',
            'MolecularWeight'
        ]
        
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{','.join(properties)}/JSON"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            props = data['PropertyTable']['Properties'][0]
            
            return MolecularMapping(
                pubchem_cid=cid,
                chembl_id=None,  # Could add ChEMBL lookup
                canonical_smiles=props.get('CanonicalSMILES'),
                inchi_key=props.get('InChIKey'),
                molecular_formula=props.get('MolecularFormula'),
                molecular_weight=props.get('MolecularWeight'),
                matched_name=""
            )
        except:
            return None
    
    def save_mapping(self, intervention_id: int, mapping: MolecularMapping):
        """Save mapping to database."""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO intervention_molecular_mappings (
                intervention_id,
                matched_name,
                pubchem_cid,
                canonical_smiles,
                inchi_key,
                molecular_formula,
                molecular_weight,
                mapping_source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'pubchem')
            ON CONFLICT (intervention_id, pubchem_cid) DO NOTHING
        """, (
            intervention_id,
            mapping.matched_name,
            mapping.pubchem_cid,
            mapping.canonical_smiles,
            mapping.inchi_key,
            mapping.molecular_formula,
            mapping.molecular_weight
        ))
        self.conn.commit()

# Usage
mapper = MolecularMapper(db_config)
mapping = mapper.map_intervention(12345)
if mapping:
    mapper.save_mapping(12345, mapping)
    print(f"Mapped to PubChem CID: {mapping.pubchem_cid}")
    print(f"SMILES: {mapping.canonical_smiles}")
```

---

## Performance Considerations

### Rate Limiting

- **PubChem**: 5 requests/second recommended
- **ChEMBL**: 10 requests/second recommended
- Add `time.sleep(0.2)` between requests

### Batch Processing

Process interventions in batches:

```python
def batch_map_interventions(mapper, intervention_ids, batch_size=100):
    """Map interventions in batches with progress tracking."""
    total = len(intervention_ids)
    mapped = 0
    
    for i in range(0, total, batch_size):
        batch = intervention_ids[i:i+batch_size]
        
        for intervention_id in batch:
            mapping = mapper.map_intervention(intervention_id)
            if mapping:
                mapper.save_mapping(intervention_id, mapping)
                mapped += 1
        
        print(f"Progress: {i+len(batch)}/{total} ({mapped} mapped)")
        time.sleep(1)  # Pause between batches
```

### Caching

Use database to cache mappings:

```python
def get_or_map_intervention(intervention_id):
    """Check cache first, then map if needed."""
    
    # Check if already mapped
    cur.execute("""
        SELECT pubchem_cid, canonical_smiles, inchi_key
        FROM intervention_molecular_mappings
        WHERE intervention_id = %s
        LIMIT 1
    """, (intervention_id,))
    
    if cur.rowcount > 0:
        return cur.fetchone()  # Use cached
    
    # Map and cache
    mapping = mapper.map_intervention(intervention_id)
    if mapping:
        mapper.save_mapping(intervention_id, mapping)
    
    return mapping
```

---

## Summary

### What ClinicalTrials.gov Provides

✅ **Drug names**: `ctgov_interventions.name`  
✅ **Aliases**: `ctgov_intervention_other_names.name`  
✅ **MeSH terms**: `ctgov_browse_interventions.mesh_term`

### What You Need to Add

❌ Molecular identifiers (SMILES, InChI, etc.)  
❌ PubChem CID, ChEMBL ID  
❌ Protein targets  
❌ Binding affinities

### Recommended Solution

1. **Extract** names from CTG tables
2. **Map** to PubChem (free, comprehensive)
3. **Enhance** with ChEMBL (bioactivity, targets)
4. **Store** mappings in your database
5. **Cache** results to avoid re-querying

This gives you molecule-protein binding capability while keeping the clinical trial context!




