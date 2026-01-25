#!/usr/bin/env python3
"""
Database Constants

Shared constants used across ingestion and search modules.
"""

# ----------------------------- OpenFDA Mapping Configuration -------------------------

# Mapping tables configuration: (table_name, column_name, openfda_field)
MAPPING_TABLES = [
    ("mapping_package_ndc", "package_ndc", "package_ndc"),
    ("mapping_substance_name", "substance_name", "substance_name"),
    ("mapping_rxcui", "rxcui", "rxcui"),
    ("mapping_spl_id", "spl_id", "spl_id"),
    ("mapping_spl_set_id", "spl_set_id", "spl_set_id"),
    ("mapping_brand_name", "brand_name", "brand_name"),
    ("mapping_generic_name", "generic_name", "generic_name"),
    ("mapping_product_ndc", "product_ndc", "product_ndc"),
    ("mapping_manufacturer_name", "manufacturer_name", "manufacturer_name"),
    ("mapping_application_number", "application_number", "application_number"),
    ("mapping_route", "route", "route"),
    ("mapping_dosage_form", "dosage_form", "dosage_form"),
    ("mapping_product_type", "product_type", "product_type"),
    ("mapping_upc", "upc", "upc"),
    ("mapping_unii", "unii", "unii"),
    ("mapping_nui", "nui", "nui"),
    ("mapping_pharm_class_pe", "pharm_class_pe", "pharm_class_pe"),
    ("mapping_pharm_class_epc", "pharm_class_epc", "pharm_class_epc"),
    ("mapping_pharm_class_cs", "pharm_class_cs", "pharm_class_cs"),
    ("mapping_pharm_class_moa", "pharm_class_moa", "pharm_class_moa"),
]

# ----------------------------- Common LOINC Codes -------------------------

# Common LOINC codes used in DailyMed sections
COMMON_LOINC_CODES = {
    "34066-1": "CONTRAINDICATIONS",
    "34067-9": "INDICATIONS AND USAGE", 
    "34068-7": "DOSAGE AND ADMINISTRATION",
    "34069-5": "HOW SUPPLIED/STORAGE AND HANDLING",
    "34070-3": "CONTRAINDICATIONS",
    "43678-2": "BOXED WARNING",
    "43679-0": "MECHANISM OF ACTION",
    "43680-8": "PHARMACOKINETICS",
    "34084-4": "ADVERSE REACTIONS",
    "34088-5": "OVERDOSAGE",
    "34090-1": "CLINICAL PHARMACOLOGY",
    "50741-8": "HIGHLIGHTS OF PRESCRIBING INFORMATION",
    "50742-6": "RECENT MAJOR CHANGES",
    "50743-4": "INDICATIONS AND USAGE",
    "50744-2": "DOSAGE AND ADMINISTRATION",
    "50745-9": "DOSAGE FORMS AND STRENGTHS",
}

# ----------------------------- Orange Book Constants -------------------------

# Therapeutic Equivalence Code categories
TE_CODE_CATEGORIES = {
    "A": "Therapeutically equivalent",
    "B": "Not therapeutically equivalent", 
    "AB": "Meets bioequivalence requirements",
    "AN": "Solutions and powders for aerosolization",
    "AO": "Injectable oil solutions",
    "AP": "Injectable aqueous solutions",
    "AT": "Topical products",
    "BC": "Extended release tablets, capsules, and injectables",
    "BD": "Active ingredients and dosage forms with documented bioequivalence problems",
    "BE": "Delayed release oral dosage forms",
    "BN": "Products in aerosol-nebulizer drug delivery systems",
    "BP": "Active ingredients and dosage forms with potential bioequivalence problems",
    "BR": "Suppositories or enemas",
    "BS": "Products having drug standard deficiencies",
    "BT": "Topical products with bioequivalence issues",
    "BX": "Drug products for which the data are insufficient",
}

# ----------------------------- Data Source URLs -------------------------

# Default headers for downloads to avoid bot detection
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# OpenFDA download index
OPENFDA_DOWNLOAD_INDEX = "https://api.fda.gov/download.json"

# Orange Book URL
ORANGE_BOOK_URL = "https://www.fda.gov/media/76860/download"

# ClinicalTrials.gov URLs
CTGOV_URLS = [
    "https://clinicaltrials.gov/AllPublicXML.zip",  # primary
    "https://classic.clinicaltrials.gov/AllPublicXML.zip",  # fallback
]

# DailyMed URLs
DAILYMED_PAGE = "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"
DAILYMED_BASE_URL = "https://dailymed-data.nlm.nih.gov/public-release-files/"
DAILYMED_SITE_BASE = "https://dailymed.nlm.nih.gov"

# DrugCentral URLs
DRUGCENTRAL_BASE_URL = "https://unmtid-dbs.net/download/DrugCentral/2023/"
DRUGCENTRAL_SDF_URLS = [
    "https://unmtid-shinyapps.net/download/DrugCentral/2023_03_05/structures.sdf",
    "https://unmtid-shinyapps.net/download/DrugCentral/2023_11_28/structures.sdf",
    "https://unmtid-shinyapps.net/download/DrugCentral/2024_08_05/structures.sdf",
]

# ----------------------------- ClinicalTrials.gov Constants -------------------------

# Common trial phases
TRIAL_PHASES = [
    "Early Phase 1",
    "Phase 1", 
    "Phase 1/Phase 2",
    "Phase 2",
    "Phase 2/Phase 3", 
    "Phase 3",
    "Phase 4",
    "Not Applicable",
]

# Common trial statuses
TRIAL_STATUSES = [
    "Active, not recruiting",
    "Completed", 
    "Enrolling by invitation",
    "Not yet recruiting",
    "Recruiting", 
    "Suspended",
    "Terminated",
    "Unknown status",
    "Withdrawn",
]

# ----------------------------- Search Limits -------------------------

# Default limits for search functions
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 1000
